import torch.nn as nn
import torch.nn.functional as F
import torch

from tlora.tensor_factorization import FactorizedTensor

class TuckerFactorizedTensor(FactorizedTensor, factorization_type="tucker"):
    """Tucker Decomposition with shared core tensor"""
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
        """Efficient Tucker reconstruction using mode products"""
        # Core x U x V for each matrix
        deltas = torch.einsum("kab,ia,jb->kij", self.core, self.U, self.V)
        return deltas[0], deltas[1], deltas[2]