import torch.nn as nn
import torch

from tlora.tensor_factorization import FactorizedTensor

class LoRAFactorizedTensor(FactorizedTensor, factorization_type="lora"):
    """LoRA Decomposition with separate low-rank pairs for Q, K, V"""
    def __init__(self, hidden_size: int, rank: int):
        super().__init__(hidden_size, rank)
        
        # LoRA parameters for each matrix (Q, K, V)
        self.A_q = nn.Parameter(torch.Tensor(hidden_size, rank))
        self.B_q = nn.Parameter(torch.Tensor(hidden_size, rank))
        self.A_k = nn.Parameter(torch.Tensor(hidden_size, rank))
        self.B_k = nn.Parameter(torch.Tensor(hidden_size, rank))
        self.A_v = nn.Parameter(torch.Tensor(hidden_size, rank))
        self.B_v = nn.Parameter(torch.Tensor(hidden_size, rank))
        
        # Initialize A with small random values and B with zeros
        nn.init.normal_(self.A_q, std=1/self.rank)
        nn.init.normal_(self.A_k, std=1/self.rank)
        nn.init.normal_(self.A_v, std=1/self.rank)
        nn.init.zeros_(self.B_q)
        nn.init.zeros_(self.B_k)
        nn.init.zeros_(self.B_v)
    
    def forward(self):
        """Compute low-rank updates using outer products"""
        delta_q = self.A_q @ self.B_q.T  # (hidden_size, hidden_size)
        delta_k = self.A_k @ self.B_k.T
        delta_v = self.A_v @ self.B_v.T
        
        return delta_q, delta_k, delta_v