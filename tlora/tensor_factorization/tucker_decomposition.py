import torch.nn as nn
import torch.nn.functional as F
import torch

from tlora.tensor_factorization import FactorizedTensor

class Tucker2FactorizedTensor(FactorizedTensor, factorization_type="tucker2"):
    """Tucker2 Decomposition with shared core tensor"""
    def __init__(self, hidden_size: int, rank: int):
        super().__init__(hidden_size, rank)
        
        # Tucker parameters
        self.core = nn.Parameter(torch.zeros(3, rank, rank))  # [Q,K,V] x R x R
        self.U = nn.Parameter(torch.zeros(hidden_size, rank))
        self.V = nn.Parameter(torch.zeros(hidden_size, rank))

        # Initialize with orthogonal matrices and leave core as zeros
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)

    def forward(self):
        """Efficient Tucker2 reconstruction using mode products"""
        # Core x U x V for each matrix
        deltas = torch.einsum("kab,ia,jb->kij", self.core, self.U, self.V)
        return deltas[0], deltas[1], deltas[2]
    
class Tucker3FactorizedTensor(FactorizedTensor, factorization_type="tucker3"):
    """Tucker-3 Decomposition with explicit dimension labels.
    
    The `rank` parameter can be either:
      - an integer, in which case all three ranks are set to that value, or
      - a tuple of three integers (r_mode, r_row, r_col).
    """
    def __init__(self, hidden_size: int, rank):
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
        r_mode, r_row, r_col = rank
        
        # Core tensor (compressed representation)
        self.core = nn.Parameter(torch.zeros(r_mode, r_row, r_col))
        
        # Factor matrices
        self.mode_factors = nn.Parameter(torch.zeros(3, r_mode))  # Q/K/V mode
        self.row_factors = nn.Parameter(torch.zeros(hidden_size, r_row))  # Matrix rows
        self.col_factors = nn.Parameter(torch.zeros(hidden_size, r_col))  # Matrix columns

        # Orthogonal initialization for the factor matrices
        nn.init.orthogonal_(self.mode_factors)
        nn.init.orthogonal_(self.row_factors)
        nn.init.orthogonal_(self.col_factors)

    def forward(self):
        """Reconstruction: core[m,r,c] x mode[m] x row[r] x col[c]"""
        deltas = torch.einsum(
            "mrc, qm, hr, kc -> qhk",
            self.core,
            self.mode_factors,
            self.row_factors,
            self.col_factors
        )
        return deltas[0], deltas[1], deltas[2]  # Return Q, K, V matrices
