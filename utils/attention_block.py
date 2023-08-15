import pandas as pd
from collections import Counter
import sys
import numpy as np
from matplotlib.pylab import plt
pd.options.mode.chained_assignment = None
from IPython.display import Image, display
from tokenizers import ByteLevelBPETokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F
import ast
from dataclasses import dataclass
from contextlib import nullcontext
import pickle, dill
from unidecode import unidecode

# Attention Block
path_to_data = '../data/'

@dataclass
class GPTConfig:
    block_size: int = 32
    vocab_size: int = 1800
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 512
    dropout: float = 0.2
    batch_size: int = 32
    temperature=1.0 
    top_k=20
    
config = dill.load(open(path_to_data + 'scratch_reddit_gpt_config.pickle','rb'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = ByteLevelBPETokenizer(path_to_data + 'redditTok-vocab.json', path_to_data +'redditTok-merges.txt')
pad_token_id = tokenizer.token_to_id('</s>')

class Head(nn.Module):
    """ self-attention """

    def __init__(self, head_size, rank=32, alpha=1):
        super().__init__()
        self.key   = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.alpha = 1
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # input of size  (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)  # decoder only, forget the future 
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) # (B, T, T)
        
        # add LoRA 
#         rank = 8
#         W_A = nn.Parameter(torch.empty(C, self.rank)) # LoRA weight A
#         W_B = nn.Parameter(torch.empty(self.rank, T))
        
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
#         out += (W_A @ W_B)*self.alpha
        return out

## Multihead Attention 

# Several self attention layers working in parallel. The result from each self attention layer are then concatenated and projected to the desired embedding dimension. 

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # concatenated over the embedding dimension (B, T , hs * num_heads) 
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        out = self.dropout(self.proj(out))
        return out

    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head # head size is determined by how many heads and how long the embedding we want 
        self.sa = MultiHeadAttention(n_head, head_size) # this many attention layer with head size output 
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd) # layer norm (normalizing features not examples)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x)) # residual layer
        return x
    
class GPTLanguageModel(nn.Module):

    def __init__(self,config):
        
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table    = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=pad_token_id)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        # Multi-head attention blocks
        self.blocks = nn.Sequential(*[Block(config.n_embd, n_head=config.n_head) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd) # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)    # (B,T,C)
        x = self.ln_f(x)      # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            mask = (targets != torch.tensor(pad_token_id)) # padding mask
            loss = F.cross_entropy(logits, targets, reduction='none')
            loss *= mask
            loss  = loss.mean()
            
        return logits, loss
        
    
    def generate(self, idx, max_new_tokens,  temperature=0.5, top_k=None, skip_special_tokens=True):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]/temperature # becomes (B, C)
            # apply softmax to get probabilities
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs  = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)            
            # append sampled index to the running sequence
            if idx_next[0][0].detach().numpy() == end_token_id:
                break
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx

