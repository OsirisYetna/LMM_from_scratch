"""
Single Block Transformer Model
Step 4: Complete Transformer Block (Attention + Feed-Forward)

This model introduces the full transformer block architecture:
- Multi-head attention for communication between tokens
- Feed-forward network for individual token processing
- Layer normalization and residual connections
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# Reproducibility
torch.manual_seed(1337)

# Hyperparameters
batch_size = 32
block_size = 64        
max_iters = 3000
learning_rate = 3e-4   
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
eval_iters = 100
n_embd = 128          
n_head = 4            
dropout = 0.2        

def load_data(file_path="../input.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded data from {file_path}")
    return text

# Tokenizer setup
text = load_data()
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): 
    return [stoi[c] for c in s]

def decode(L): 
    return "".join([itos[i] for i in L])

def prepare_data(text, encode_fn):
    data = torch.tensor(encode_fn(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(split, train_data, val_data):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """One head of self-attention"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand dimension by 4x
            nn.ReLU(),                      # Non-linearity
            nn.Linear(4 * n_embd, n_embd),  # Project back to original dimension
            nn.Dropout(dropout),            # Regularization
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Complete Transformer block: communication followed by computation"""
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # Self-attention
        self.ffwd = FeedForward(n_embd)                  # Feed-forward
        self.ln1 = nn.LayerNorm(n_embd)                  # Layer norm 1
        self.ln2 = nn.LayerNorm(n_embd)                  # Layer norm 2
        
    def forward(self, x):
        # Pre-norm architecture with residual connections
        x = x + self.sa(self.ln1(x))      # Self-attention with residual
        x = x + self.ffwd(self.ln2(x))    # Feed-forward with residual
        return x

class SingleBlockTransformer(nn.Module):
    """Transformer with a single transformer block"""
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Single transformer block
        self.transformer_block = TransformerBlock(n_embd, n_head)
        
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb  # (B,T,C)
        
        # Apply single transformer block
        x = self.transformer_block(x)  # (B,T,C)
        
        # Final processing
        x = self.ln_f(x)  # Final layer norm
        logits = self.lm_head(x)  # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def train_model():
    print("=== SINGLE BLOCK TRANSFORMER ===")
    print("Key innovation: Complete transformer block architecture")
    print("- Multi-head self-attention for token communication")
    print("- Feed-forward network for individual token processing")
    print("- Layer normalization (pre-norm architecture)")
    print("- Residual connections for training stability")
    print("- All components working together in one block\n")
    
    train_data, val_data = prepare_data(text, encode)
    
    print(f"Vocab size: {vocab_size}")
    print(f"Training data length: {len(train_data):,} tokens")
    print(f"Context length: {block_size}")
    print(f"Embedding dimension: {n_embd}")
    print(f"Number of attention heads: {n_head}")
    print(f"Feed-forward expansion factor: 4x")
    
    model = SingleBlockTransformer()
    model = model.to(device)
    
    print(f"Number of parameters: {sum(p.numel