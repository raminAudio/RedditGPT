import pandas as pd
from collections import Counter
import sys
import numpy as np
from matplotlib.pylab import plt
pd.options.mode.chained_assignment = None
from IPython.display import Image, display
import torch
import torch.nn as nn
from torch.nn import functional as F
import ast
from dataclasses import dataclass
from contextlib import nullcontext
import pickle, dill
from unidecode import unidecode

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path_to_data = '../data/'
eval_iters = 50

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

train_data = pickle.load(open(path_to_data + 'train_data.pickle','rb'))
val_data = pickle.load(open(path_to_data + 'valid_data.pickle','rb'))

# Training Helper
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data), (config.batch_size,)) # randomy pick a data sample
    x = torch.stack([data[i][0:-1] for i in ix])
    y = torch.stack([data[i][1:] for i in ix]) # label is shift by 1
    x, y = x.to(device), y.to(device)
    return x, y