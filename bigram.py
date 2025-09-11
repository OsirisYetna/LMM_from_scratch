"""
Bigram Language Model Implementation
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# Reproducibility
torch.manual_seed(1337)

# Hyperparameters
batch_size = 32  # Number of sequences processed in parallel
block_size = 8   # Maximum context length for predictions
max_iters = 20000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 1000
eval_iters = 200

# Data loading
def load_data(file_path="input.txt"):
    """Load and prepare text data"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def create_tokenizer(text):
    """Create character-level tokenizer"""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(s): 
        return [stoi[c] for c in s]
    
    def decode(L): 
        return "".join([itos[i] for i in L])
    
    return encode, decode, vocab_size

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
    
    # Random starting positions to avoid overfitting
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

class BigramLanguageModel(nn.Module):
    """Simple Bigram Language Model"""
    
    def __init__(self, vocab_size):
        super().__init__()
        # Token embedding table: each token maps to a vocab_size dimensional vector
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass
        idx and targets are both (B,T) tensor of integers
        """
        # Get token embeddings
        logits = self.token_embedding_table(idx)  # (B,T,C)
        
        if targets is None:
            loss = None
        else:
            # Reshape for cross entropy loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens given a context
        idx is (B, T) array of indices in the current context
        """
        for _ in range(max_new_tokens):
            # Get predictions
            logits, loss = self(idx)
            
            # Focus only on the last time step (bigram model)
            logits = logits[:, -1, :]  # (B, C)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        
        return idx

def train_model():
    """Main training function"""
    print("Loading data...")
    text = load_data()
    encode, decode, vocab_size = create_tokenizer(text)
    train_data, val_data = prepare_data(text, encode)
    
    print(f"Vocab size: {vocab_size}")
    print(f"Training data length: {len(train_data):,} tokens")
    print(f"Validation data length: {len(val_data):,} tokens")
    
    # Initialize model
    model = BigramLanguageModel(vocab_size)
    model = model.to(device)
    
    # Print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("\nStarting training...")
    for iter in range(max_iters):
        
        # Evaluate loss periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = get_batch('train', train_data, val_data)

        # Evaluate the loss
        logits, loss = model(xb, yb)
        
        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\nTraining completed!")
    
    # Generate some text
    print("\nGenerating sample text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    print(generated_text)
    
    return model, encode, decode

def main():
    """Main function"""
    print(f"Using device: {device}")
    print("="*50)
    
    model, encode, decode = train_model()
    
    # Interactive generation
    print("\n" + "="*50)
    print("Model training complete! You can now generate text.")
    print("Type 'quit' to exit, or enter a prompt to generate text from it.")
    
    while True:
        prompt = input("\nEnter prompt (or 'quit'): ")
        if prompt.lower() == 'quit':
            break
        
        try:
            # Encode the prompt
            context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            
            # Generate text
            generated = model.generate(context, max_new_tokens=100)
            generated_text = decode(generated[0].tolist())
            
            print(f"\nGenerated: {generated_text}")
            
        except KeyError as e:
            print(f"Error: Character '{e}' not in vocabulary")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()