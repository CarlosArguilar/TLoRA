import torch.nn as nn
import torch

from tlora.tensor_factorization import FactorizedTensor


class CPFactorizedTensor(FactorizedTensor, factorization_type="cp"):
    """Canonical Polyadic (CP) Decomposition"""
    def __init__(self, hidden_size: int, rank: int):
        super().__init__(hidden_size, rank)
        
        # Shared factors for Q/K/V with separate weights per matrix
        self.A = nn.Parameter(torch.zeros(3, rank))  # [Q,K,V] x rank
        self.B = nn.Parameter(torch.zeros(hidden_size, rank))
        self.C = nn.Parameter(torch.zeros(hidden_size, rank))

        # Init with scaled normal distribution
        nn.init.normal_(self.A, std=1.0 / (rank ** 0.5))
        nn.init.normal_(self.B, std=1.0 / (hidden_size ** 0.5))

    def forward(self):
        """Efficient batched CP reconstruction using einsum"""
        deltas = torch.einsum("kr,ir,jr->kij", self.A, self.B, self.C)
        return deltas[0], deltas[1], deltas[2]