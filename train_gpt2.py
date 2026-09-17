from dataclasses import dataclass
import torch 
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import time
import tiktoken
from torch.distributed import init_process_group, destroy_process_group
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
from hellaswag import iterate_examples, render_example 


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads but into a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.TINYGPT_SCALE_INIT=1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask but following the OpenAI/HF naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimentionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to the batch
        # nh is "number of heads", ns is "head size", C is number of channels = nh*ns
        # e.g in GPT 2 (124M), n_head=12, hs = 64 => C = 768 channels in the transformer
        qkv = self.c_attn(x)# splits the matrix into a number of matrices who'se 3rd dim will be of size n_embd or its remainder for the last one if % is not 0
        # btw in pytorch dims are always 0-indexed (1st dim is always dim=0 ect...), hence here we get 3 matrices of shape (B,T,n_embd) or less for the last one
        # and since qkv's C is 3* config.n_embd, we get proper 3 (B,T,n_embd) matrices to neatly represent our Q,K,V matrices
        q,k,v = qkv.split(self.n_embd, dim = 2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B,nh,T,hs)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B,nh,T,hs)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B,nh,T,hs)
        # attention (materializes the large (T,T) matrix for all the querys and keys)

        # btw all of the loops we are jumping through here addding a 4th dimension to batch on both batch_size and number of heads
        # is to allow to compute in parallel all attention heads using pytorch

        # btw negative indexing woks just like negative indexing for lists in python
        #attn = (q @ k.transpose(-2,-1)) * (1/math.sqrt(k.size(-1)))
        #attn = attn.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        #attn = F.softmax(attn, dim=-1)
        #y = attn @ v 

        # implementing flash attention in one kernel for better performance according to the flash attention paper

        y = F.scaled_dot_product_attention(q,k,v,is_causal=True) # magnifique

        y=y.transpose(1,2).contiguous().view(B,T,C) # re-assemble all head outputs side by side
        # this above performes the concatenation operation of all the heads outputs
        # output projection
        y=self.c_proj(y)
        return y

        # this btw follows exactly the skema of the hugging face transformer code format which should allow us to now port over all the weights from there

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd) #linear layer
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd) #2nd linear layer, merging back to n_embd
        self.c_proj.TINYGPT_SCALE_INIT=1
    def forward(self, x):  
        x = self.c_fc(x)
        x = self.gelu(x)    
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # pre-norm layernorm for attention
        self.attn = CausalSelfAttention(config) #self-attention section
        self.ln_2 = nn.LayerNorm(config.n_embd) # tanh-based approximation of GELU is here used (though today there is not significant comutational cost difference between proper GELU and tanh-based aprox of GELU, there used to be one in tensorflow which is why the tanh one was used to train GPT-2 which is what we are trying to mimic here)
        self.mlp = MLP(config) #feedforwa rd section

    def forward(self,x):
        x = x +self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x)) # both pre-norm layernom are applied here + residual connection addition for gradient flow 
        return x

@dataclass
class GPTConfig:
    block_size: int=1024 #max sequence length
    vocab_size: int=50257 # number of tokens : 50 000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    n_layer: int=12 # number of layers
    n_head: int=12 # number of heads
    n_embd: int=768 # embedding dimension
    # here hyperparams match GPT-2 (124M)


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config=config
        # let's try to reflect the gpt2 hugging face schema
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), #output embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional encoding 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # all transformer layers (attention heads, feed forwards, pre-norm linear normalizations and residual connections)
            ln_f = nn.LayerNorm(config.n_embd) # final layernorm layer
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 
         
        # weight sharing scheme 
        self.transformer.wte.weight = self.lm_head.weight 

        # init params
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'TINYGPT_SCALE_INIT'):
                std*= (2 * self.config.n_layer)**(-0.5)
            torch.nn.init.normal_(module.weight,  mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 
        

    def forward(self, idx, targets=None):
        #idx is of shape B,T
        B,T = idx.size()
        assert T <= self.config.block_size, f'Cannot forward sequence of length {T}, block size is only {self.config.block_size}'
        # forward the token and position embeddings
        pos = torch.arange(0,T, dtype=torch.long, device = idx.device) # T long row vectors
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (B,T,n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B,T,n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #B,T,vocab_size

        loss = None 
        if targets is not None: 
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn:p for pn,p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameter that is 2D will be weight decayed, otherwise no.
        # i.e all weight, tensors in matmuls + embeddings decay, all biases and layernorms don't
        decay_params = [p for n,p in param_dict.items() if p.dim()>=2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim()<2]
        optim_groups = [
            {'params': decay_params, 'weight_decay':weight_decay},
            {'params': nodecay_params, 'weight_decay':0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f'num decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters')
        print(f'num non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters')
        # Create the AdamW optimizer and use the fused version if available 
        fuse_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fuse_available and ('mps' in device or 'cuda' in device)
        print(f'using fused AdamW : {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8,fused=use_fused )
        return optimizer

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long) # converting the numpy file to a long tensor
    return ptt

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# now importing the weights :
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes,split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes 
        assert split in {'train','val'}

        # get the shard filenames 
        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root,s) for s in shards]
        self.shards = shards
        assert len(shards)>0, f'no shards found for split {split}'
        if master_process:
            print(f'found {len(shards)} shards for split {split}')

        # state init at shard 0 
        self.reset()

    
    def reset(self):
        # state, init at shard 0
        self.current_shard=0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B*self.T*self.process_rank

    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T +1] #+1 to be able to generate a y of the same size as x 
        x = (buf[:-1]).view(B,T) # inputs
        y = (buf[1:]).view(B,T) # targets
        # advance the position in the tensor
        self.current_position += B*T*self.num_processes
        # if loading the next batch would be out of bounds, advance to the next shard
        if self.current_position + (B*T*self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard+1)% len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard]) 
            self.current_position = B*T*self.process_rank 
        return x,y



# set up DDP (distributed data parallel)
# torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE

ddp = int(os.environ.get('RANK',-1)) != -1 # is this a ddp run ?
if ddp:
    # us of DDP demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now we need CUDA for ddpr"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing, ect
else:
    #vanilla non-DDP
    ddp_rank = 0
    ddp_local_rank=0
    ddp_world_size=1  
    master_process=True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    print(f'using device: {device}')

# DDP should not execute on MPS as it doesn't make any sense

# pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

# Section to batch the data -------- : 

# here since the GPT 3 paper stipulates that they trained on 0.5M token-sized batches and that scaling up our sequence length or batch 
# sizes to reach that number would cause spark a H-bomb in the GPU's memory, we'll use grad accumulation to still fit that 
# hyper parameter without changing B nor T, rather reaching 0.5M sequentially

# boolean to control compilation, not compiling has for only benefit that we get to sample text from the model through training

use_compile = False # it interferes with hellaswag benchmarking   

total_batch_size = 524288 # tokens, 2^19 to have a nice number that ~= 0.5M tokens
B=64 #micro-batch size 
T = 1024 # sequence length
assert (total_batch_size) % (B*T*ddp_world_size) == 0, "make sure total batch size is divisible by B*T*ddp_world_size" 
grad_accum_steps = total_batch_size // (B*T*ddp_world_size) #here for us it'll be 64, hence we'll only flush the gradients every 64 iterations to accumulate gradients over the 64 batch processing sessions 
if master_process:
    print(f'total desired batch size : {total_batch_size}')
    print(f'=> calculated gradient accumulation steps : {grad_accum_steps}')
train_loader = DataLoaderLite(B=B,T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B,T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
torch.set_float32_matmul_precision('high') # high uses TensorFloat 32 which is 8x more efficient
# no-op on mps

# create model
model = GPT(GPTConfig(vocab_size=50304)) # this initializes default, random model
# 50304 is a beautiful number, dividable by 2 all the way up to 128
# and though this adds more compute, turning an ugly number into a nice number did unlock a 4% bit of performance 
# and though we create tokens that don't exist in the dataset, it won't hurt since those probabilities will be driven to 
# 0 in training since they will never occur in the training data since they literally don't exist, this just allow for 
# better use of the GPU 
print(f'Detected device : {device}')
model.to(device)
if use_compile:
    model = torch.compile(model) #basicaly moves like gcc/clang and compiles to efficient ready to use asm instructions which makes the code a lot faster 
# allows for crazy optimization GPU side, to avoid chip-HBM roundtrips and gain performance.
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) #documentation is kind of unclear but it should be ddp_local_rank ig
raw_model = model.module if ddp else model
 
# cosine decay learning schedule

#------------
max_lr = 6e-4 * 3
min_lr = max_lr * 0.1
warmup_steps = 715 #matching the GPT-3 warmup schedule
max_steps = 19073
#------------
# GPT 3 paper hyperparams

def get_lr(step):
    # 1) linear warmups
    if step<warmup_steps:
        return max_lr * (step+1)/warmup_steps
    # 2) if step> lr_decay_iters, return min learning rate
    if step> max_steps:
        return min_lr
    # 3) in between use cosine decay
    decay_ratio = (step-warmup_steps) / (max_steps-warmup_steps)
    assert 0<=decay_ratio<=1
    coeff = 0.5*(1.0+math.cos(math.pi*decay_ratio)) # coef starts at 1 and goes to 0
    return min_lr + coeff * (max_lr-min_lr)

#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-8) # alternative to sgd  
# keeps buffers that seem like momentum, and sort of normalizes which performs better than vanilla sgd on nlp 

#optimize : 
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)\

# encoder 

enc = tiktoken.get_encoding('gpt2')

# create log directory
log_dir='log'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'log.txt')
with open(log_file, 'w') as f: # open for writing to clear the file
    pass


for step in range(max_steps):
    t0 = time.time()
    # once in a while, evaluate our validation loss
    if step%2500==0 or step==max_steps-1:   
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps=20
            for _ in range(val_loss_steps):
                x,y = val_loader.next_batch() 
                x,y = x.to(device), y.to(device) 
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16): #for better efficiency 
                    logits, loss = model(x.to(device),y.to(device)) #only changes those to float16  
                loss = loss / val_loss_steps # normalizer we would have lost otherwise which would have rendered our gradients off. 
                val_loss_accum += loss.detach() # detaching the tensor from the gradient graph
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f'validation loss : {val_loss_accum.item():.4f}')

                if step > 0 and (step % 5000 == 0 or step==max_steps-1):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    # you might also want to add optimizer.state_dict() and
                    # rng seeds etc., if you wanted to more exactly resume training
                    torch.save(checkpoint, checkpoint_path)   

    # once in a while, evaluate hellaswag
    if (step % 250 == 0 or step == max_steps-1) and not use_compile:
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            #only process examples where i%ddp_world_size = ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm+= int (pred_norm==label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device = device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device = device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm/num_total
        if master_process:
            print(f'Hellaswag accuracy : {num_correct_norm}/{num_total}={acc_norm:.4f}')
            with open(log_file, 'a') as f:
                f.write(f'{step} hellaswag : {acc_norm:.4f}\n')


    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or step==(max_steps-1))   and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    # training loop:
    model.train()
    optimizer.zero_grad() # zero out the gradients
    loss_accum=0.0
    for micro_step in range(grad_accum_steps): # accumulate for 32 iterations 
        x,y = train_loader.next_batch() 
        x,y = x.to(device), y.to(device) 
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16): #for better efficiency 
            logits, loss = model(x.to(device),y.to(device)) #only changes those to float16  
        loss = loss/grad_accum_steps # normalizer we would have lost otherwise which would have rendered our gradients off. 
        loss_accum +=loss.detach() # detaching the tensor from the gradient graph 
        if ddp:
            model.require_backward_grad_sync = (micro_step == (grad_accum_steps-1)) # should only turn on on the last step to avoid synchronization at every micro_step while avoiding to use the pytorch context manager abstraction here
        loss.backward() #generate and +='s the gradients
    if ddp:
         dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) #averages loss_accum on all ranks such that master_process and all other ranks hold the mean of loss_accum
    # nudge accordingly later after the 32 accumulations (0.5M tokens got processed )
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr # setting the learning rate in the optimizer for all parameter groups
    optimizer.step() # adamW optimize on th enew gradients
    if device.startswith('cuda'): torch.cuda.synchronize()
    elif device.startswith('mps'): torch.mps.synchronize()
    t1 = time.time()  
    dt = t1-t0
    tokens_processed = train_loader.B*train_loader.T*grad_accum_steps*ddp_world_size
    tokens_per_sec = (tokens_processed)/dt
    if master_process:
        print(f'step {step} | loss : {loss_accum.item():.6f} | norm : {norm:.4f} | lr : {lr:.4e} | dt: {dt:.2f}s | tok/sec : {tokens_per_sec:.2f}')
        with open(log_file,'a') as f:
            f.write(f'{step} train {loss_accum.item():.6f}\n')
 
if ddp:
    destroy_process_group()

import sys; sys.exit(0)
