"""
Transformer Language Model Implementation - Version Corrigée
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# Reproducibility
torch.manual_seed(1337)

# Hyperparameters optimized for CPU
batch_size = 16        
block_size = 64      
max_iters = 1000     
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 200  
eval_iters = 50 
n_embd = 128        
n_head = 4           
n_layer = 3           
dropout = 0.2

def load_data(file_path="input.txt"):
    """Load and prepare text data"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded data from {file_path}")
    return text

# Tokenizer
text = load_data()
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): 
    return [stoi[c] for c in s]

def decode(L): 
    return "".join([itos[i] for i in L])

def prepare_data(text, encode_fn):
    """Prepare training and validation data"""
    data = torch.tensor(encode_fn(text), dtype=torch.long)
    
    # Train/validation split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data

def get_batch(split, train_data, val_data):
    """Generate a small batch of data"""
    data = train_data if split == 'train' else val_data
    
    # Random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Context and targets
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Estimate loss on train and validation sets"""
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
        q = self.query(x)  # (B,T,head_size)
        k = self.key(x)    # (B,T,head_size)
        
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (C**-0.5)  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Apply to values
        v = self.value(x)  # (B,T,head_size)
        out = wei @ v      # (B,T,head_size)
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
    """Simple linear layer followed by a non-linearity"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   
        x = x + self.ffwd(self.ln2(x))  
        return x

class TransformerLanguageModel(nn.Module):
    """Transformer Language Model"""
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
 
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embeddings
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        
        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
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
        """Generate new tokens given a context"""
        for _ in range(max_new_tokens):
            # Crop context to last block_size tokens
            idx_cond = idx[:, -block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)  # ✅ Correction: utiliser idx_cond
            
            # Focus on last time step
            logits = logits[:, -1, :]  # (B,C)
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

def train_model():
    """Main training function"""
    print("Loading data...")
    train_data, val_data = prepare_data(text, encode)
    
    print(f"Vocab size: {vocab_size}")
    print(f"Training data length: {len(train_data):,} tokens")
    print(f"Validation data length: {len(val_data):,} tokens")
    
    # Initialize model
    model = TransformerLanguageModel()
    model = model.to(device)
    
    # Print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"\nStarting training on {device}...")
    for iter in range(max_iters):
        
        # Evaluate loss periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample batch and train
        xb, yb = get_batch('train', train_data, val_data)
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\nTraining completed!")
    
    # Generate sample text
    print("\nGenerating sample text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=200)[0].tolist())
    print("Generated text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
    
    return model

if __name__ == "__main__":
    # Run training
    model = train_model()
    
    # Interactive generation
    print("\nInteractive mode - Enter some text to continue (or 'quit' to exit):")
    while True:
        user_input = input("\nPrompt: ").strip()
        if user_input.lower() == 'quit':
            break
        if user_input:
            try:
                context = torch.tensor([encode(user_input)], dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=100)
                result = decode(generated[0].tolist())
                print(f"Generated: {result}")
            except Exception as e:
                print(f"Error: {e}")