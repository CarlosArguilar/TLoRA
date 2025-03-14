
import pytest
import torch

from tlora.tensor_factorization.htftucker import HTFTuckerFactorizedTensor

def test_gradient_propagation():
    hidden_size = 768
    num_heads = 12
    rank = 4
    model = HTFTuckerFactorizedTensor(hidden_size, rank, num_heads)

    for param in model.parameters():
        param.data.fill_(1.0)  # Set to ones for deterministic behavior
    
    # Dummy forward pass and loss computation
    deltas = model()
    loss = sum(delta.sum() for delta in deltas)  # Arbitrary scalar loss
    
    # Backward pass
    loss.backward()
    
    # Verify gradients exist and are non-zero
    for param in model.parameters():
        assert param.grad is not None, f"Gradient for {param.shape} is None!"
        assert torch.any(param.grad != 0), f"All gradients for {param.shape} are zero!"

def test_zero_mode_factors_affect_deltas():
    hidden_size = 768
    num_heads = 12
    head_dim = hidden_size // num_heads  # 64
    rank = (3, 8, 8, 8)  # group_rank=3 (for Q/K/V), others=8
    
    model = HTFTuckerFactorizedTensor(hidden_size, rank, num_heads)
    
    # Freeze all parameters to avoid random initialization effects
    for param in model.parameters():
        param.data.fill_(1.0)  # Set to ones for deterministic behavior
    
    # Zero out mode_factors for the first head (index 0)
    model.mode_factors.data[0] = 0
    
    # Forward pass
    delta_q, delta_k, delta_v = model()
    
    # Check if the first head_dim columns (64) are zero for all deltas
    for delta in [delta_q, delta_k, delta_v]:
        # Slice the first head_dim columns (64) across all rows
        assert torch.allclose(
            delta[:head_dim, :], 
            torch.zeros_like(delta[:head_dim, :]),
            atol=1e-8
        ), "Zeroing mode_factors did not zero the expected head columns!"
        
        # Verify other columns are non-zero
        assert not torch.allclose(
            delta[head_dim:, :], 
            torch.zeros_like(delta[head_dim:, :]),
            atol=1e-8
        ), "Unexpected zero columns outside the first head!"

def test_zero_group_factors_zeroes_group_delta():
    hidden_size = 768
    num_heads = 12
    rank = (3, 8, 8, 8)
    
    model = HTFTuckerFactorizedTensor(hidden_size, rank, num_heads)
    
    # Initialize all factors to ones for deterministic behavior
    for param in model.parameters():
        param.data.fill_(1.0)
    
    # Zero group_factors for Q (index 0)
    model.group_factors.data[0] = 0
    
    # Forward pass
    delta_q, delta_k, delta_v = model()
    
    # Verify delta_q is all zeros
    assert torch.allclose(
        delta_q, 
        torch.zeros_like(delta_q),
        atol=1e-8
    ), "Zeroing group_factors for Q did not zero delta_q!"
    
    # Verify delta_k and delta_v are non-zero
    assert not torch.allclose(delta_k, torch.zeros_like(delta_k)), "delta_k is unexpectedly zero!"
    assert not torch.allclose(delta_v, torch.zeros_like(delta_v)), "delta_v is unexpectedly zero!"

def test_multiple_heads_zeroing():
    hidden_size = 768
    num_heads = 12
    head_dim = hidden_size // num_heads
    rank = (3, 8, 8, 8)
    
    model = HTFTuckerFactorizedTensor(hidden_size, rank, num_heads)
    
    # Set all factors to ones
    for param in model.parameters():
        param.data.fill_(1.0)
    
    # Zero mode_factors for the first two heads (indices 0 and 1)
    model.mode_factors.data[:2] = 0
    
    delta_q, delta_k, delta_v = model()
    
    # First 2*head_dim columns (128) should be zero
    for delta in [delta_q, delta_k, delta_v]:
        assert torch.allclose(
            delta[:2*head_dim, :], 
            torch.zeros_like(delta[:2*head_dim, :]),
            atol=1e-8
        ), "Zeroing first two heads did not zero columns 0-127!"
        
        # Columns beyond 128 should be non-zero
        assert not torch.allclose(
            delta[2*head_dim:, :], 
            torch.zeros_like(delta[2*head_dim:, :]),
            atol=1e-8
        ), "Columns beyond 128 are unexpectedly zero!"