"""
Example usage of the Transformer implementation.

This script demonstrates how to use the Transformer model for:
1. Text classification
2. Sequence representation learning
3. Attention visualization

Author: CodeImplementer
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from transformer import Transformer, MultiHeadAttention, PositionalEncoding
from typing import Dict, List, Tuple


class TransformerClassifier(nn.Module):
    """
    A complete Transformer-based text classifier.
    
    This combines the Transformer encoder with a classification head
    for tasks like sentiment analysis, document classification, etc.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super(TransformerClassifier, self).__init__()
        
        self.transformer = Transformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            x (torch.Tensor): Input token ids of shape (batch_size, seq_len)
            mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes)
        """
        # Get transformer output
        transformer_output = self.transformer(x, mask)
        
        # Global average pooling over sequence dimension
        if mask is not None:
            mask = mask.squeeze(1).squeeze(1)  # (batch_size, seq_len)
            lengths = mask.sum(dim=1, keepdim=True).float()  # (batch_size, 1)
            pooled = (transformer_output * mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            pooled = transformer_output.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


def create_sample_data(vocab_size: int, seq_len: int, batch_size: int, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create sample data for demonstration.
    
    Args:
        vocab_size (int): Size of vocabulary
        seq_len (int): Sequence length
        batch_size (int): Batch size
        num_classes (int): Number of classes
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Input tokens, labels, and masks
    """
    # Create random token sequences
    inputs = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    # Create random labels
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Create masks (simulate padding)
    masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
    for i in range(batch_size):
        # Randomly set some positions to False to simulate padding
        pad_start = torch.randint(seq_len // 2, seq_len, (1,)).item()
        masks[i, pad_start:] = False
    
    return inputs, labels, masks


def demonstrate_attention_mechanism():
    """Demonstrate the core attention mechanism with simple examples."""
    print("=" * 60)
    print("MULTI-HEAD ATTENTION DEMONSTRATION")
    print("=" * 60)
    
    # Create a simple multi-head attention module
    d_model = 64
    num_heads = 8
    seq_len = 10
    batch_size = 2
    
    attention = MultiHeadAttention(d_model, num_heads)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = attention(x, x, x)  # Self-attention
    
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in attention.parameters())}")
    print()


def demonstrate_positional_encoding():
    """Visualize positional encoding patterns."""
    print("=" * 60)
    print("POSITIONAL ENCODING DEMONSTRATION")
    print("=" * 60)
    
    d_model = 128
    max_len = 100
    
    pos_encoding = PositionalEncoding(d_model, max_len, dropout=0.0)
    
    # Create dummy input
    dummy_input = torch.zeros(max_len, 1, d_model)
    encoded = pos_encoding(dummy_input)
    
    # Extract positional encodings
    pe = encoded[:, 0, :].detach().numpy()
    
    print(f"Positional encoding shape: {pe.shape}")
    
    # Plot some dimensions
    plt.figure(figsize=(12, 8))
    
    # Plot first few dimensions
    for i in range(0, min(8, d_model), 2):
        plt.subplot(2, 4, i//2 + 1)
        plt.plot(pe[:50, i], label=f'sin(dim={i})')
        plt.plot(pe[:50, i+1], label=f'cos(dim={i+1})')
        plt.legend()
        plt.title(f'Dimensions {i} and {i+1}')
        plt.xlabel('Position')
        plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('positional_encoding.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Positional encoding plot saved as 'positional_encoding.png'")
    print()


def demonstrate_full_transformer():
    """Demonstrate the full transformer on a classification task."""
    print("=" * 60)
    print("FULL TRANSFORMER DEMONSTRATION")
    print("=" * 60)
    
    # Model parameters
    vocab_size = 1000
    num_classes = 5
    d_model = 256
    num_heads = 8
    num_layers = 4
    d_ff = 1024
    seq_len = 128
    batch_size = 4
    
    # Create model
    model = TransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample data
    inputs, labels, masks = create_sample_data(vocab_size, seq_len, batch_size, num_classes)
    
    # Create proper masks for transformer
    attention_masks = model.transformer.create_padding_mask(inputs)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Attention mask shape: {attention_masks.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(inputs, attention_masks)
        predictions = torch.argmax(logits, dim=1)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Predictions: {predictions.tolist()}")
    print(f"True labels: {labels.tolist()}")
    print()


def train_simple_example():
    """Train a simple transformer on synthetic data."""
    print("=" * 60)
    print("TRAINING EXAMPLE")
    print("=" * 60)
    
    # Simple parameters for quick training
    vocab_size = 100
    num_classes = 3
    d_model = 128
    num_heads = 4
    num_layers = 2
    seq_len = 32
    batch_size = 8
    num_epochs = 5
    
    # Create model
    model = TransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        # Generate batch
        inputs, labels, _ = create_sample_data(vocab_size, seq_len, batch_size, num_classes)
        attention_masks = model.transformer.create_padding_mask(inputs)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(inputs, attention_masks)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.4f}")
    
    print("Training completed!")
    print()


def analyze_attention_weights():
    """Analyze and visualize attention weights."""
    print("=" * 60)
    print("ATTENTION WEIGHTS ANALYSIS")
    print("=" * 60)
    
    # Create a simple example
    d_model = 64
    num_heads = 4
    seq_len = 8
    batch_size = 1
    
    # Modified attention class to return weights
    class AttentionWithWeights(MultiHeadAttention):
        def forward(self, query, key, value, mask=None):
            batch_size, seq_len, _ = query.shape
            
            # Linear projections and reshape for multi-head attention
            Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            
            # Apply scaled dot-product attention
            attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
            
            # Concatenate heads and put through final linear layer
            attention_output = attention_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.d_model
            )
            
            output = self.w_o(attention_output)
            
            return output, attention_weights
    
    attention = AttentionWithWeights(d_model, num_heads)
    
    # Create sample input representing a simple sentence
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Get output and attention weights
    output, weights = attention(x, x, x)
    
    print(f"Attention weights shape: {weights.shape}")  # (batch, heads, seq_len, seq_len)
    
    # Average attention weights across heads
    avg_weights = weights[0].mean(dim=0)  # (seq_len, seq_len)
    
    # Visualize attention matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(avg_weights.detach().numpy(), cmap='Blues')
    plt.colorbar()
    plt.title('Average Attention Weights')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.savefig('attention_weights.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Attention weights visualization saved as 'attention_weights.png'")
    print()


def main():
    """Run all demonstrations."""
    print("TRANSFORMER IMPLEMENTATION DEMONSTRATION")
    print("Based on 'Attention is All You Need' (Vaswani et al., 2017)")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_attention_mechanism()
    demonstrate_positional_encoding()
    demonstrate_full_transformer()
    train_simple_example()
    analyze_attention_weights()
    
    print("All demonstrations completed!")
    print("\nKey files generated:")
    print("- positional_encoding.png: Visualization of positional encodings")
    print("- attention_weights.png: Attention weights heatmap")


if __name__ == "__main__":
    main()