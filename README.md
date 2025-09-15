# Large Language Models from Scratch

This project explores the implementation of language models from the ground up, starting with a simple bigram model and progressing to a complete **decoder-only transformer**. The goal is to deeply understand the theory behind LLMs and gain practical experience in implementing them.

## Project Goals

- **Understand the theory**: Master the fundamental concepts of language models
- **Practical implementation**: Code architectures from scratch using PyTorch
- **Educational progression**: Move from simple (bigram) to complex (transformer)
- **Detailed documentation**: Explain every component and concept

## Input - Training Data

For training, we use the Tiny Shakespeare dataset (1M character-level token), a ~1 MB collection of several Shakespeare plays. It provides a rich but compact corpus of classical English text, making it ideal for experimenting with character-level language models. The goal is to train a model to generate new passages in a Shakespeare-like style.

# Large Language Models from Scratch

This project explores the implementation of language models from the ground up, starting with a simple bigram model and progressing to a complete **decoder-only transformer**. The goal is to deeply understand the theory behind LLMs and gain practical experience in implementing them.

## Project Goals

- **Understand the theory**: Master the fundamental concepts of language models
- **Practical implementation**: Code architectures from scratch using PyTorch
- **Educational progression**: Move from simple (bigram) to complex (transformer)
- **Detailed documentation**: Explain every component and concept

## Input - Training Data

For training, we use the Tiny Shakespeare dataset (1M character-level tokens), a ~1 MB collection of several Shakespeare plays. It provides a rich but compact corpus of classical English text, making it ideal for experimenting with character-level language models. The goal is to train a model to generate new passages in a Shakespeare-like style.

## Project Content

### 1. Bigram Model (`models/bigram.py`)
**The starting point** – A simple character-level model that predicts the next character based solely on the previous character.

**Purpose:**  
Introduce the basics of language modeling and autoregressive generation in the simplest possible setup.

**Key concepts covered:**
- Character-level tokenization
- Token embeddings
- Cross-entropy loss
- Text generation through sampling

**Parameters:**
- Embedding dimension: 32
- Context window: 8 characters
- ~10K parameters

---

### 2. Intermediate Transformer Block (`models/single_head_attention.py`, `models/multiple_head_attention.py`, `models/single_block_transformer.py`)
**The bridge to modern transformers** – Introduces the core components of transformer architectures in a simplified.

**Key concepts covered:**
- **Single-Head Self-Attention**: Learn how the model weighs the importance of each token relative to others.
- **Multi-Head Self-Attention**: Capture different types of relationships simultaneously.
- **Feed-Forward Networks**: Enrich token representations independently.
- **Residual Connections & Layer Normalization**: Stabilize and improve training of deeper models.
- **Modular Transformer Block**: Combines attention, feed-forward, residuals, and normalization into one unit.
- **Causal Masking**: Ensures autoregressive generation.

**Learning goals:**
- Understand attention mechanisms and multi-head benefits.
- Experiment with a modular transformer before scaling up.

**Parameters (CPU-optimized):**
- Embedding dimension: 64
- Number of heads: 2–4
- Number of layers: 1–2
- Context window: 32 tokens
- ~100–200K parameters

---

### 3. Decoder-Only Transformer (`models/decoder_transformer.py`)
**The modern architecture** – A GPT-style decoder-only transformer that generates text autoregressively using causal attention.

**Key concepts implemented:**
- Multi-Head Self-Attention
- Positional Embeddings
- Feed-Forward Networks
- Layer Normalization
- Residual Connections
- Causal Masking

**Why decoder-only?**  
Unlike the original Transformer, which uses both encoder and decoder, GPT-style models rely solely on the decoder. This allows the model to generate text one token at a time while attending only to previous tokens.

**Current Parameters (CPU-optimized):**
- Embedding dimension: 128
- Number of heads: 4
- Number of layers: 3
- Context window: 64 tokens
- ~850K parameters

## Learning Resources

This implementation was built following:
- **[Attention is all you need](https://arxiv.org/abs/1706.03762)** - The seminal transformer paper by Vaswani et al.
- **[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6618s&ab_channel=AndrejKarpathy)** - Andrej Karpathy's

## Getting Started

### Prerequisites
```bash
pip install torch
```

### Running the Models - demo
In my macbook I have python3 but it's depends on what setting you have in your computer (python or python3 ...)

1. **Bigram Model:**
```bash
python3 models/bigram_model.py
```

2. **Single Block Transformer:**
```bash
python3 models/single_block_transformer.py
```

3. **Transformer Model:**
```bash
python3 models/transformer_model.py
```

## Results & Performance

### Bigram Model
- **Simple but limited**: Only considers the immediate previous character
- **Fast training**: Converges quickly on CPU
- **Baseline performance**: Good for understanding fundamentals but very limited in generation quality

### Transformer Model
- **Context awareness**: Can attend to all previous tokens in the sequence
- **Better generation quality**: More coherent and contextual text. The quality depends on hyperparameters and training duration.
With a reasonnable training with a CPU, we notice the english structure but it is still not english.
- **Scalable architecture**: Foundation for larger models


## Possible Improvements

### Short-term Enhancements
- **GPU Training**: use a GPU to train the model deeper and faster
- **Hyperparameter Scaling**: 
  - Increase embedding dimensions (384, 512, 768...)
  - More attention heads (8, 12, 16...)
  - Deeper networks (6, 12, 24+ layers)
  - Larger context windows (256, 512, 1024+ tokens)
- **Better Tokenization**: Move from character-level to subword tokenization (BPE, SentencePiece)
- **Regularization**: Experiment with dropout rates, weight decay

### Advanced Extensions
- **Encoder-Decoder Transformer**: Implement the full transformer architecture for tasks like translation, assistance...
- **Attention Variants**: Explore different attention mechanisms (sparse attention, local attention)
- **Model Scaling**: Experiment with larger architectures (GPT-2, GPT-3 scale)
- **Fine-tuning Capabilities**: Add support for task-specific fine-tuning

## Architecture Details

### Decoder-Only vs Full Transformer
```
Full Transformer (Original Paper):
Input → Encoder → Decoder → Output

Decoder-Only (This Implementation):
Input → Decoder → Output
```

The decoder-only architecture is simpler but just as powerful for language generation tasks, which is why it's used in models like GPT, PaLM, and LLaMA.

### Key Components Explained
- **Causal Self-Attention**: Each token can only see previous tokens
- **Multi-Head Attention**: Multiple attention patterns learned in parallel  
- **Position Embeddings**: Learned representations of token positions
- **Layer Norm**: Applied before (pre-norm) each sub-layer
- **Residual Connections**: Help gradient flow in deep networks
