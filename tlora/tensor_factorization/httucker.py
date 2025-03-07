import torch.nn as nn
import torch.nn.functional as F
import torch

from tlora.tensor_factorization import FactorizedTensor

class HTTuckerFactorizedTensor(FactorizedTensor, factorization_type="httucker"):
    def __init__(self, hidden_size: int, rank, num_heads: int = 12):
        # Process rank: if int, convert to a tuple; if tuple, verify length.
        if isinstance(rank, int):
            rank = (rank, rank, rank)
        elif isinstance(rank, tuple):
            if len(rank) != 3:
                raise ValueError("Rank tuple must contain 3 values: (r_mode, r_row, r_col)")
        else:
            raise ValueError("rank must be either an int or a tuple of 3 ints")
        
        # Call parent constructor with the processed rank tuple
        super().__init__(hidden_size, rank)
        self.hidden_size = hidden_size
        
        r_mode, r_row, r_col = rank
        
        # Define groups to avoid code duplication
        self.groups = ['q', 'k', 'v']
        for group in self.groups:
            # Create and register the core tensor for each group
            setattr(self, f"{group}core", nn.Parameter(torch.zeros(r_mode, r_row, r_col)))
            
            # Create and register the factor matrices for each group
            setattr(self, f"{group}mode_factors", nn.Parameter(torch.zeros(num_heads, r_mode)))
            setattr(self, f"{group}row_factors", nn.Parameter(torch.zeros(self.hidden_size, r_row)))
            setattr(self, f"{group}col_factors", nn.Parameter(torch.zeros(self.hidden_size // num_heads, r_col)))
            
            # Orthogonal initialization for the factor matrices (but not the core tensor)
            nn.init.orthogonal_(getattr(self, f"{group}mode_factors"))
            nn.init.orthogonal_(getattr(self, f"{group}row_factors"))
            nn.init.orthogonal_(getattr(self, f"{group}col_factors"))
    
    def _reconstruct(self, group: str):
        """Helper method to compute reconstruction for a given group (q, k, or v)."""
        core = getattr(self, f"{group}core")
        mode_factors = getattr(self, f"{group}mode_factors")
        row_factors = getattr(self, f"{group}row_factors")
        col_factors = getattr(self, f"{group}col_factors")
        
        # Perform tensor contraction using Einstein summation
        result = torch.einsum("mrc, qm, hr, kc -> hkq", 
                                core, mode_factors, row_factors, col_factors)
        # Reshape to combine the output dimensions
        return result.reshape(self.hidden_size, self.hidden_size)
    
    def forward(self):
        """Reconstruction: core[m,r,c] x mode[m] x row[r] x col[c] for Q, K, V"""
        outputs = tuple(self._reconstruct(group) for group in self.groups)
        return outputs  # Returns a tuple: (deltas_q, deltas_k, deltas_v)
