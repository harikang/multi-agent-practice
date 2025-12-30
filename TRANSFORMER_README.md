# Transformer Implementation: "Attention is All You Need"

A complete, production-ready PyTorch implementation of the Transformer architecture from the seminal paper ["Attention is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762).

## 🌟 Features

- **Complete Implementation**: All core components including Multi-Head Attention, Positional Encoding, and Feed-Forward Networks
- **Production Ready**: Clean, modular code with comprehensive documentation and error handling
- **Well Tested**: Extensive test suite covering all components and edge cases
- **Educational**: Clear documentation explaining implementation decisions and mathematical foundations
- **Flexible**: Easy to extend for various tasks (classification, sequence-to-sequence, etc.)

## 📁 Project Structure

```
├── transformer.py          # Core Transformer implementation
├── transformer_example.py  # Usage examples and demonstrations
├── test_transformer.py     # Comprehensive test suite
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone or download the files
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from transformer import Transformer

# Create a Transformer model
model = Transformer(
    vocab_size=10000,    # Size of vocabulary
    d_model=512,         # Model dimension
    num_heads=8,         # Number of attention heads
    num_layers=6,        # Number of encoder layers
    d_ff=2048,          # Feed-forward dimension
    max_len=1000,       # Maximum sequence length
    dropout=0.1         # Dropout probability
)

# Create input (batch_size=2, seq_len=10)
input_ids = torch.randint(0, 10000, (2, 10))

# Forward pass
output = model(input_ids)  # Shape: (2, 10, 512)
print(f"Output shape: {output.shape}")
```

### Classification Example

```python
from transformer_example import TransformerClassifier

# Create classifier
classifier = TransformerClassifier(
    vocab_size=10000,
    num_classes=5,
    d_model=256,
    num_heads=8,
    num_layers=4
)

# Input and forward pass
input_ids = torch.randint(0, 10000, (4, 50))  # batch_size=4, seq_len=50
logits = classifier(input_ids)  # Shape: (4, 5)

# Get predictions
predictions = torch.argmax(logits, dim=1)
```

## 🏗️ Architecture Components

### 1. Multi-Head Attention

The core of the Transformer, implementing the scaled dot-product attention mechanism:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Key Features:**
- Multiple attention heads running in parallel
- Scaled dot-product attention with optional masking
- Proper weight initialization using Xavier uniform

### 2. Positional Encoding

Sinusoidal positional encodings to inject position information:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 3. Feed-Forward Network

Position-wise feed-forward network with ReLU activation:

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

### 4. Layer Normalization & Residual Connections

Each sub-layer includes:
- Residual connection: `LayerNorm(x + Sublayer(x))`
- Proper normalization for training stability

## 🔧 Implementation Decisions

### 1. **Pre-Layer Norm vs Post-Layer Norm**
- Implemented **pre-layer norm** for better gradient flow and training stability
- More commonly used in modern implementations

### 2. **Weight Initialization**
- Xavier uniform initialization for linear layers
- Proper scaling of embeddings by `√d_model`

### 3. **Attention Masking**
- Flexible masking support for padding and causal attention
- Efficient implementation using large negative values

### 4. **Dropout Placement**
- Applied to attention weights, feed-forward networks, and embeddings
- Configurable dropout rate for all components

### 5. **Memory Efficiency**
- Proper tensor reshaping to minimize memory allocation
- Efficient attention computation

## 📊 Usage Examples

Run the comprehensive example script:

```bash
python transformer_example.py
```

This will demonstrate:

1. **Multi-Head Attention Mechanism**: Core attention computation
2. **Positional Encoding Visualization**: Sinusoidal patterns
3. **Full Transformer**: Complete model usage
4. **Training Example**: Simple training loop
5. **Attention Analysis**: Attention weight visualization

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_transformer.py
```

The test suite includes:
- **Unit tests** for all components
- **Shape verification** for all tensor operations
- **Mathematical property tests** (e.g., positional encoding correctness)
- **Integration tests** for the complete model
- **Performance benchmarks**

## 📈 Performance

The implementation is optimized for:
- **Memory efficiency**: Minimal tensor copying
- **Computational efficiency**: Vectorized operations
- **Training stability**: Proper initialization and normalization

Typical performance on a modern GPU:
- **Small model** (d_model=256): ~100-200 sequences/sec
- **Base model** (d_model=512): ~50-100 sequences/sec
- **Large model** (d_model=1024): ~20-50 sequences/sec

## 🔬 Mathematical Foundation

### Scaled Dot-Product Attention

The attention mechanism computes a weighted sum of values based on compatibility between queries and keys:

1. **Compute attention scores**: `scores = QK^T / √d_k`
2. **Apply softmax**: `weights = softmax(scores)`
3. **Compute output**: `output = weights × V`

### Multi-Head Attention

Multiple attention functions are computed in parallel:

1. **Project to multiple subspaces**: `Q_i = XW_Q^i`, `K_i = XW_K^i`, `V_i = XW_V^i`
2. **Compute attention for each head**: `head_i = Attention(Q_i, K_i, V_i)`
3. **Concatenate and project**: `MultiHead = Concat(head_1, ..., head_h)W_O`

## 🎯 Extensions and Applications

This implementation can be easily extended for:

### 1. **Text Classification**
```python
# Add classification head
classifier = nn.Linear(d_model, num_classes)
pooled_output = transformer_output.mean(dim=1)  # Global average pooling
logits = classifier(pooled_output)
```

### 2. **Sequence-to-Sequence Tasks**
```python
# Add decoder layers for full seq2seq
# Implement cross-attention between encoder and decoder
```

### 3. **Language Modeling**
```python
# Add causal masking for autoregressive generation
# Add language modeling head
lm_head = nn.Linear(d_model, vocab_size)
```

### 4. **Custom Attention Patterns**
```python
# Modify attention masks for specific patterns
# Implement sparse attention for long sequences
```

## 📚 References

1. **Original Paper**: Vaswani, A., et al. "Attention is all you need." NIPS 2017.
2. **The Illustrated Transformer**: Jay Alammar's excellent blog post
3. **The Annotated Transformer**: Harvard NLP's line-by-line implementation
4. **Transformers from Scratch**: Peter Bloem's detailed tutorial

## 🤝 Contributing

This implementation is designed to be educational and production-ready. Contributions welcome for:

- Performance optimizations
- Additional attention variants (sparse, local, etc.)
- More comprehensive examples
- Documentation improvements

## 📄 License

MIT License - Feel free to use this implementation for research, education, or production use.

## 🔍 Key Implementation Details

### Memory Management
- Efficient tensor operations with minimal copying
- Proper gradient flow for deep networks
- Configurable sequence length limits

### Numerical Stability
- Proper attention score scaling
- Layer normalization for stable training
- Gradient clipping recommendations

### Flexibility
- Modular design for easy customization
- Support for different model sizes
- Easy integration with existing PyTorch workflows

---

**Note**: This implementation focuses on the encoder portion of the Transformer, which is widely used for tasks like BERT, classification, and representation learning. The decoder implementation can be added similarly following the same architectural principles.