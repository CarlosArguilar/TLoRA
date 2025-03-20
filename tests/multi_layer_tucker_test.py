import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tlora.tensor_factorization.multi_layer_tucker import MultiLayerTucker  # Adjust the import path accordingly

def test_gradient_propagation():
    hidden_size = 768
    num_layers = 12
    # You can pass an integer rank (which will be expanded to a tuple) for simplicity
    rank = 4  
    model = MultiLayerTucker(hidden_size, rank, num_layers)
    
    # Set all parameters to ones for deterministic behavior.
    for param in model.parameters():
        param.data.fill_(1.0)
    
    # Dummy forward pass (using layer 0) and loss computation.
    delta_q, delta_k, delta_v = model(0)
    loss = sum(delta.sum() for delta in (delta_q, delta_k, delta_v))
    
    # Backward pass.
    loss.backward()
    
    # Verify that every parameter received a non-zero gradient.
    for param in model.parameters():
        assert param.grad is not None, f"Gradient for parameter with shape {param.shape} is None!"
        assert torch.any(param.grad != 0), f"All gradients for parameter with shape {param.shape} are zero!"

def test_zero_mode_factors_zeroes_q_delta():
    hidden_size = 768
    num_layers = 12
    rank = (3, 8, 8, 8)  # r_layer, r_mode, r_row, r_col
    model = MultiLayerTucker(hidden_size, rank, num_layers)
    
    # Fill all parameters with ones.
    for param in model.parameters():
        param.data.fill_(1.0)
    
    # Zero out the first row of mode_factors corresponding to Q.
    model.mode_factors.data[0] = 0
    
    # Forward pass for layer 0.
    delta_q, delta_k, delta_v = model(0)
    
    # Verify that delta_q is completely zero.
    assert torch.allclose(delta_q, torch.zeros_like(delta_q), atol=1e-8), \
        "Zeroing mode_factors for Q did not zero delta_q!"
    
    # Verify that delta_k and delta_v remain non-zero.
    assert not torch.allclose(delta_k, torch.zeros_like(delta_k), atol=1e-8), \
        "delta_k is unexpectedly zero!"
    assert not torch.allclose(delta_v, torch.zeros_like(delta_v), atol=1e-8), \
        "delta_v is unexpectedly zero!"

def test_zero_layer_factors_zeroes_layer_delta():
    hidden_size = 768
    num_layers = 12
    rank = (3, 8, 8, 8)
    model = MultiLayerTucker(hidden_size, rank, num_layers)
    
    # Set all parameters to ones.
    for param in model.parameters():
        param.data.fill_(1.0)
    
    # Zero out the layer_factors for layer 0.
    model.layer_factors.data[0] = 0
    
    # Forward pass for layer 0: the entire delta should become zero.
    delta_q, delta_k, delta_v = model(0)
    assert torch.allclose(delta_q, torch.zeros_like(delta_q), atol=1e-8), \
        "Zeroing layer_factors for layer 0 did not zero delta_q!"
    assert torch.allclose(delta_k, torch.zeros_like(delta_k), atol=1e-8), \
        "Zeroing layer_factors for layer 0 did not zero delta_k!"
    assert torch.allclose(delta_v, torch.zeros_like(delta_v), atol=1e-8), \
        "Zeroing layer_factors for layer 0 did not zero delta_v!"
    
    # Forward pass for a different layer (e.g. layer 1) to verify that the deltas are non-zero.
    delta_q1, delta_k1, delta_v1 = model(1)
    assert not torch.allclose(delta_q1, torch.zeros_like(delta_q1), atol=1e-8), \
        "delta_q for layer 1 is unexpectedly zero!"
    assert not torch.allclose(delta_k1, torch.zeros_like(delta_k1), atol=1e-8), \
        "delta_k for layer 1 is unexpectedly zero!"
    assert not torch.allclose(delta_v1, torch.zeros_like(delta_v1), atol=1e-8), \
        "delta_v for layer 1 is unexpectedly zero!"

def test_multiple_mode_factors_zeroing():
    hidden_size = 768
    num_layers = 12
    rank = (3, 8, 8, 8)
    model = MultiLayerTucker(hidden_size, rank, num_layers)
    
    # Set all parameters to ones.
    for param in model.parameters():
        param.data.fill_(1.0)
    
    # Zero out the first two rows of mode_factors (corresponding to Q and K).
    model.mode_factors.data[:2] = 0
    
    # Forward pass for layer 0.
    delta_q, delta_k, delta_v = model(0)
    
    # Verify that both delta_q and delta_k are completely zero.
    assert torch.allclose(delta_q, torch.zeros_like(delta_q), atol=1e-8), \
        "Zeroing mode_factors for Q did not zero delta_q!"
    assert torch.allclose(delta_k, torch.zeros_like(delta_k), atol=1e-8), \
        "Zeroing mode_factors for K did not zero delta_k!"
    
    # Verify that delta_v is non-zero.
    assert not torch.allclose(delta_v, torch.zeros_like(delta_v), atol=1e-8), \
        "delta_v is unexpectedly zero!"
