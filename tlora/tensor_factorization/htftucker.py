import torch.nn as nn
import torch.nn.functional as F
import torch

from tlora.tensor_factorization import FactorizedTensor

import torch.nn as nn
import torch.nn.functional as F
import torch

from tlora.tensor_factorization import FactorizedTensor

class HTFTuckerFactorizedTensor(FactorizedTensor, factorization_type="htftucker"):
    def __init__(self, hidden_size: int, rank, num_heads: int = 12):
        # Process rank into a tuple of 3 integers
        if isinstance(rank, int):
            rank = (rank, rank, rank)
        elif isinstance(rank, tuple) and len(rank) != 4:
            raise ValueError("Rank must be an int or a tuple of 4 ints")
        
        super().__init__(hidden_size, rank)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        r_group, r_mode, r_row, r_col = rank
        
        # Define a single core tensor with an extra dimension for groups (Q, K, V)
        self.core = nn.Parameter(torch.zeros(r_group, r_mode, r_row, r_col))  # Shape: [3, r_mode, r_row, r_col]
        
        # Shared factor matrices across Q/K/V
        self.group_factors = nn.Parameter(torch.zeros(3, r_group))      # Shape: [3, r_group]
        self.mode_factors = nn.Parameter(torch.zeros(num_heads, r_mode))      # Shape: [num_heads, r_mode]
        self.row_factors = nn.Parameter(torch.zeros(hidden_size, r_row))      # Shape: [hidden_size, r_row]
        self.col_factors = nn.Parameter(torch.zeros(hidden_size // num_heads, r_col))  # Shape: [head_dim, r_col]

        # Orthogonal initialization for factors
        nn.init.orthogonal_(self.group_factors)
        nn.init.orthogonal_(self.mode_factors)
        nn.init.orthogonal_(self.row_factors)
        nn.init.orthogonal_(self.col_factors)

    def forward(self):
        # Compute all three deltas (Q, K, V) in one einsum
        # Shape of `result`: [3, num_heads, head_dim, hidden_size]
        result = torch.einsum(
            "gmrc, dg, qm, hr, kc -> dqkh", 
            self.core,             
            self.group_factors,      
            self.mode_factors,          
            self.row_factors,           
            self.col_factors            
        )

        delta_q = result[0].reshape(self.hidden_size, self.hidden_size)
        delta_k = result[1].reshape(self.hidden_size, self.hidden_size)
        delta_v = result[2].reshape(self.hidden_size, self.hidden_size)
        
        return delta_q, delta_k, delta_v