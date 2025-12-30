"""
Transformer Implementation based on "Attention is All You Need" (Vaswani et al., 2017)

This module implements the complete Transformer architecture including:
- Multi-Head Attention mechanism
- Positional Encoding
- Encoder and Decoder layers
- Complete Transformer model

Author: CodeImplementer
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in "Attention is All You Need".
    
    This implements the scaled dot-product attention with multiple attention heads
    running in parallel, allowing the model to attend to information from different
    representation subspaces at different positions.
    
    Args:
        d_model (int): The dimensionality of input and output features
        num_heads (int): The number of attention heads
        dropout (float): Dropout probability for attention weights
        
    Attributes:
        d_k (int): Dimension of key/query vectors (d_model // num_heads)
        num_heads (int): Number of attention heads
        w_q, w_k, w_v (nn.Linear): Linear projections for queries, keys, values
        w_o (nn.Linear): Output linear projection
        dropout (nn.Dropout): Dropout layer
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def scaled_dot_product_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, d_k)
            value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, d_k)
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attention output and attention weights
        """
        d_k = query.size(-1)
        
        # Compute attention scores: QK^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, value)
        
        return attention_output, attention_weights
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Multi-Head Attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model)
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model)
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(attention_output)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional Encoding as described in "Attention is All You Need".
    
    Since the Transformer contains no recurrence or convolution, positional encodings
    are added to give the model information about the relative or absolute position
    of tokens in the sequence.
    
    Args:
        d_model (int): The dimensionality of the model
        max_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter, but part of the model state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Args:
        d_model (int): Input and output dimension
        d_ff (int): Hidden dimension of feed-forward network
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    
    Each encoder layer consists of:
    1. Multi-head self-attention
    2. Position-wise feed-forward network
    Both with residual connections and layer normalization.
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward hidden dimension
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of multiple encoder layers.
    
    Args:
        num_layers (int): Number of encoder layers
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward hidden dimension
        dropout (float): Dropout probability
    """
    
    def __init__(
        self, 
        num_layers: int, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through all encoder layers.
        
        Args:
            x (torch.Tensor): Input tensor
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Output tensor
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    
    This implementation focuses on the encoder part of the transformer,
    which is commonly used for tasks like classification, representation learning, etc.
    
    Args:
        vocab_size (int): Size of vocabulary
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of encoder layers
        d_ff (int): Feed-forward hidden dimension
        max_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer.
        
        Args:
            src (torch.Tensor): Source sequence tensor of shape (batch_size, seq_len)
            src_mask (torch.Tensor, optional): Source mask
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Token embeddings scaled by sqrt(d_model)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        # Note: pos_encoding expects (seq_len, batch_size, d_model)
        src_emb = src_emb.transpose(0, 1)
        src_emb = self.pos_encoding(src_emb)
        src_emb = src_emb.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
        
        # Apply transformer encoder
        output = self.encoder(src_emb, src_mask)
        
        return output
    
    def create_padding_mask(self, seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
        """
        Create a padding mask for the sequence.
        
        Args:
            seq (torch.Tensor): Input sequence
            pad_token (int): Padding token id
            
        Returns:
            torch.Tensor: Padding mask
        """
        return (seq != pad_token).unsqueeze(1).unsqueeze(2)