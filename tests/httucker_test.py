
import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tlora.tensor_factorization.httucker import HTTuckerFactorizedTensor

def test_gradient_propagation_httucker():
    hidden_size = 768
    num_heads = 12
    rank = 4  # When rank is an int, it will be converted to (4, 4, 4)
    model = HTTuckerFactorizedTensor(hidden_size, rank, num_heads)

    # Set all parameters to ones for deterministic behavior
    for param in model.parameters():
        param.data.fill_(1.0)
    
    # Dummy forward pass and loss computation
    outputs = model()  # Returns a tuple: (delta_q, delta_k, delta_v)
    loss = sum(output.sum() for output in outputs)  # Arbitrary scalar loss
    
    # Backward pass
    loss.backward()
    
    # Verify that gradients exist and are non-zero for each parameter
    for param in model.parameters():
        assert param.grad is not None, f"Gradient for parameter with shape {param.shape} is None!"
        assert torch.any(param.grad != 0), f"All gradients for parameter with shape {param.shape} are zero!"


def test_zero_mode_factors_affect_deltas_httucker():
    hidden_size = 768
    num_heads = 12
    head_dim = hidden_size // num_heads  # 64
    rank = (3, 8, 8)  # Rank tuple for (r_mode, r_row, r_col)
    
    groups = ['q', 'k', 'v']
    
    for group in groups:
        model = HTTuckerFactorizedTensor(hidden_size, rank, num_heads)
        
        # Initialize all parameters to ones for deterministic behavior.
        for param in model.parameters():
            param.data.fill_(1.0)
        
        # Zero out the mode factors for the first head in the current group.
        getattr(model, f"{group}mode_factors").data[0] = 0
        
        # Forward pass: get outputs for Q, K, V.
        delta_q, delta_k, delta_v = model()
        outputs = {'q': delta_q, 'k': delta_k, 'v': delta_v}
        
        # For the current group, the first head's block (first head_dim rows) should be zero.
        assert torch.allclose(
            outputs[group][:head_dim, :],
            torch.zeros_like(outputs[group][:head_dim, :]),
            atol=1e-8
        ), f"Zeroing {group}mode_factors for head 0 did not zero the expected block in delta_{group}!"
        
        # Also, ensure that for the current group the other rows are non-zero.
        assert not torch.allclose(
            outputs[group][head_dim:, :],
            torch.zeros_like(outputs[group][head_dim:, :]),
            atol=1e-8
        ), f"Non-head0 block in delta_{group} is unexpectedly zero!"
        
        # Verify that the other groups (not modified) are entirely non-zero.
        for other_group in groups:
            if other_group == group:
                continue
            assert not torch.allclose(
                outputs[other_group],
                torch.zeros_like(outputs[other_group]),
                atol=1e-8
            ), f"delta_{other_group} is unexpectedly zero when zeroing {group}!"

def test_multiple_heads_zeroing_httucker():
    hidden_size = 768
    num_heads = 12
    head_dim = hidden_size // num_heads  # 64
    rank = (3, 8, 8)
    
    groups = ['q', 'k', 'v']
    
    for group in groups:
        model = HTTuckerFactorizedTensor(hidden_size, rank, num_heads)
        
        # Set all parameters to ones for deterministic behavior.
        for param in model.parameters():
            param.data.fill_(1.0)
        
        # Zero out the mode factors for the first two heads in the current group.
        getattr(model, f"{group}mode_factors").data[:2] = 0
        
        # Forward pass.
        delta_q, delta_k, delta_v = model()
        outputs = {'q': delta_q, 'k': delta_k, 'v': delta_v}
        
        # For the current group, the first two heads (first 2*head_dim rows) should be zero.
        assert torch.allclose(
            outputs[group][:2*head_dim, :],
            torch.zeros_like(outputs[group][:2*head_dim, :]),
            atol=1e-8
        ), f"Zeroing the first two heads in {group}mode_factors did not zero the expected block in delta_{group}!"
        
        # Check that the remaining rows in the current group's output are non-zero.
        assert not torch.allclose(
            outputs[group][2*head_dim:, :],
            torch.zeros_like(outputs[group][2*head_dim:, :]),
            atol=1e-8
        ), f"Remaining block in delta_{group} is unexpectedly zero!"
        
        # Verify that the other groups (not modified) are entirely non-zero.
        for other_group in groups:
            if other_group == group:
                continue
            assert not torch.allclose(
                outputs[other_group],
                torch.zeros_like(outputs[other_group]),
                atol=1e-8
            ), f"delta_{other_group} is unexpectedly zero when zeroing {group}!"