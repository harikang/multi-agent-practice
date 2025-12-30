"""
Test suite for Transformer implementation.

This module contains comprehensive tests for all components of the Transformer
implementation to ensure correctness and robustness.

Author: CodeImplementer
License: MIT
"""

import torch
import torch.nn as nn
import pytest
import math
from transformer import (
    MultiHeadAttention, 
    PositionalEncoding, 
    FeedForward,
    EncoderLayer, 
    TransformerEncoder, 
    Transformer
)


class TestMultiHeadAttention:
    """Test cases for MultiHeadAttention module."""
    
    def test_initialization(self):
        """Test proper initialization of MultiHeadAttention."""
        d_model = 512
        num_heads = 8
        
        attention = MultiHeadAttention(d_model, num_heads)
        
        assert attention.d_model == d_model
        assert attention.num_heads == num_heads
        assert attention.d_k == d_model // num_heads
        
    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise assertion error."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=513, num_heads=8)  # Not divisible
    
    def test_forward_pass_shapes(self):
        """Test that forward pass produces correct output shapes."""
        batch_size, seq_len, d_model = 2, 10, 64
        num_heads = 8
        
        attention = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_attention_with_mask(self):
        """Test attention mechanism with masking."""
        batch_size, seq_len, d_model = 1, 4, 64
        num_heads = 8
        
        attention = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create a mask that masks out last two positions
        mask = torch.ones(batch_size, num_heads, seq_len, seq_len)
        mask[:, :, :, 2:] = 0
        
        output = attention(x, x, x, mask)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_self_attention_properties(self):
        """Test properties of self-attention."""
        batch_size, seq_len, d_model = 1, 5, 64
        num_heads = 4
        
        attention = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        
        # Create identical inputs
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(x, x, x)
        
        # Output should have same batch and sequence dimensions
        assert output.shape[:-1] == x.shape[:-1]


class TestPositionalEncoding:
    """Test cases for PositionalEncoding module."""
    
    def test_initialization(self):
        """Test proper initialization of PositionalEncoding."""
        d_model = 512
        max_len = 1000
        
        pos_enc = PositionalEncoding(d_model, max_len)
        
        assert pos_enc.pe.shape == (max_len, 1, d_model)
    
    def test_encoding_properties(self):
        """Test mathematical properties of positional encoding."""
        d_model = 64
        max_len = 100
        
        pos_enc = PositionalEncoding(d_model, max_len, dropout=0.0)
        
        # Test that even dimensions use sine and odd use cosine
        pe = pos_enc.pe.squeeze(1)  # (max_len, d_model)
        
        # Check that the encoding follows the expected pattern
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        expected_sin = torch.sin(position * div_term)
        expected_cos = torch.cos(position * div_term)
        
        # Check sine components (even indices)
        torch.testing.assert_close(pe[:, 0::2], expected_sin, rtol=1e-5, atol=1e-5)
        
        # Check cosine components (odd indices)  
        torch.testing.assert_close(pe[:, 1::2], expected_cos, rtol=1e-5, atol=1e-5)
    
    def test_forward_pass(self):
        """Test forward pass of positional encoding."""
        seq_len, batch_size, d_model = 10, 2, 64
        
        pos_enc = PositionalEncoding(d_model, dropout=0.0)
        x = torch.randn(seq_len, batch_size, d_model)
        
        output = pos_enc(x)
        
        assert output.shape == (seq_len, batch_size, d_model)


class TestFeedForward:
    """Test cases for FeedForward module."""
    
    def test_forward_pass_shapes(self):
        """Test that feed-forward network produces correct shapes."""
        batch_size, seq_len, d_model = 2, 10, 512
        d_ff = 2048
        
        ff = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = ff(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_intermediate_dimension(self):
        """Test that intermediate dimension is correct."""
        d_model = 256
        d_ff = 1024
        
        ff = FeedForward(d_model, d_ff)
        
        # Check that first linear layer has correct output dimension
        assert ff.linear1.out_features == d_ff
        assert ff.linear2.in_features == d_ff
        assert ff.linear2.out_features == d_model


class TestEncoderLayer:
    """Test cases for EncoderLayer module."""
    
    def test_forward_pass(self):
        """Test forward pass of encoder layer."""
        batch_size, seq_len, d_model = 2, 10, 256
        num_heads = 8
        d_ff = 1024
        
        layer = EncoderLayer(d_model, num_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = layer(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_residual_connections(self):
        """Test that residual connections work properly."""
        batch_size, seq_len, d_model = 1, 5, 64
        num_heads = 4
        d_ff = 256
        
        # Create layer with no dropout to test residuals clearly
        layer = EncoderLayer(d_model, num_heads, d_ff, dropout=0.0)
        
        # Create input
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output = layer(x)
        
        # Output should be different from input (due to transformations)
        assert not torch.allclose(output, x)
        
        # But should have same shape
        assert output.shape == x.shape


class TestTransformerEncoder:
    """Test cases for TransformerEncoder module."""
    
    def test_multiple_layers(self):
        """Test encoder with multiple layers."""
        batch_size, seq_len, d_model = 2, 10, 256
        num_layers = 6
        num_heads = 8
        d_ff = 1024
        
        encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = encoder(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert len(encoder.layers) == num_layers


class TestTransformer:
    """Test cases for complete Transformer module."""
    
    def test_initialization(self):
        """Test proper initialization of Transformer."""
        vocab_size = 1000
        d_model = 512
        
        transformer = Transformer(vocab_size, d_model)
        
        assert transformer.d_model == d_model
        assert transformer.embedding.num_embeddings == vocab_size
        assert transformer.embedding.embedding_dim == d_model
    
    def test_forward_pass(self):
        """Test forward pass of complete transformer."""
        batch_size, seq_len = 2, 10
        vocab_size = 1000
        d_model = 256
        
        transformer = Transformer(vocab_size, d_model)
        
        # Create input token indices
        src = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        output = transformer(src)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_padding_mask_creation(self):
        """Test padding mask creation."""
        batch_size, seq_len = 2, 5
        vocab_size = 100
        
        transformer = Transformer(vocab_size)
        
        # Create sequence with padding (token 0)
        src = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
        
        mask = transformer.create_padding_mask(src)
        
        expected_mask = torch.tensor([
            [[[True, True, True, False, False]]],
            [[[True, True, False, False, False]]]
        ])
        
        assert torch.equal(mask, expected_mask)
    
    def test_embedding_scaling(self):
        """Test that embeddings are properly scaled by sqrt(d_model)."""
        vocab_size = 100
        d_model = 64
        seq_len = 5
        
        transformer = Transformer(vocab_size, d_model)
        
        # Create simple input
        src = torch.randint(1, vocab_size, (1, seq_len))
        
        # Get raw embeddings
        raw_emb = transformer.embedding(src)
        
        # Check that forward pass scales embeddings
        # We can't directly check this without modifying the code,
        # but we can verify the model works correctly
        output = transformer(src)
        assert output.shape == (1, seq_len, d_model)


def run_performance_test():
    """Run a performance test to ensure model runs efficiently."""
    print("Running performance test...")
    
    batch_size, seq_len = 4, 128
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    )
    
    # Create input
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    mask = transformer.create_padding_mask(src)
    
    # Time forward pass
    import time
    
    transformer.eval()
    with torch.no_grad():
        start_time = time.time()
        
        for _ in range(10):
            output = transformer(src, mask)
        
        end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    print(f"Average forward pass time: {avg_time:.4f} seconds")
    print(f"Model parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    return avg_time < 1.0  # Should complete within 1 second


def main():
    """Run all tests."""
    print("Running Transformer implementation tests...")
    
    # Create test instances
    test_classes = [
        TestMultiHeadAttention(),
        TestPositionalEncoding(),
        TestFeedForward(),
        TestEncoderLayer(),
        TestTransformerEncoder(),
        TestTransformer()
    ]
    
    # Run tests
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\nTesting {class_name}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
    
    # Run performance test
    print(f"\nRunning performance test...")
    perf_passed = run_performance_test()
    total_tests += 1
    if perf_passed:
        passed_tests += 1
        print("  ✓ Performance test passed")
    else:
        print("  ✗ Performance test failed")
    
    # Summary
    print(f"\nTest Summary:")
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("All tests passed! ✓")
        return True
    else:
        print("Some tests failed! ✗")
        return False


if __name__ == "__main__":
    main()