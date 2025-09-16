"""
Single Head Attention Model
Step 2: From Bigram to Single Head Self-Attention

This model introduces the core attention mechanism with just one attention head.
Key addition: Self-attention allows each position to look at all previous positions.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# Reproducibility
torch.manual_seed(1337)

# Hyperparameters
batch_size = 16  # Number of sequences processed in parallel
block_size = 64   # Maximum context length for predictions
max_iters = 5000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
eval_iters = 500
n_embd = 128  

def load_data(file_path="input.txt"):
    """Load and prepare text data"""
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

class SingleHead(nn.Module):
    """Single head of self-attention"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B,T,head_size)
        k = self.key(x)    # (B,T,head_size)
        
        # Compute attention scores with scaling
        wei = q @ k.transpose(-2, -1) * (C**-0.5)  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        # Apply to values
        v = self.value(x)  # (B,T,head_size)
        out = wei @ v      # (B,T,head_size)
        return out

class SingleHeadAttentionModel(nn.Module):
    """Language model with single head self-attention"""
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = SingleHead(n_embd)  # Single attention head
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Get embeddings
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        
        # Apply self-attention
        x = self.sa_head(x)  # (B,T,C)
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
    print("=== SINGLE HEAD ATTENTION MODEL ===")
    print("Key innovation: Self-attention mechanism")
    print("- Each position can attend to all previous positions")
    print("- Single attention head processes the entire sequence")
    print("- Position embeddings added for sequence understanding\n")
    
    train_data, val_data = prepare_data(text, encode)
    
    print(f"Vocab size: {vocab_size}")
    print(f"Training data length: {len(train_data):,} tokens")
    print(f"Context length: {block_size}")
    
    model = SingleHeadAttentionModel()
    model = model.to(device)
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"\nStarting training on {device}...")
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        xb, yb = get_batch('train', train_data, val_data)
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print("\nTraining completed!")
    return model

if __name__ == "__main__":
    model = train_model()
    
    # Interactive mode
    print("\nInteractive mode (type 'quit' to exit):")
    while True:
        prompt = input("\nPrompt: ").strip()
        if prompt.lower() == 'quit':
            break
        if prompt:
            try:
                context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=100)
                result = decode(generated[0].tolist())
                print(f"Generated: {result}")
            except Exception as e:
                print(f"Error: {e}")