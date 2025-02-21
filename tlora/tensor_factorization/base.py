import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Type, Dict, Tuple

class FactorizedTensor(ABC, nn.Module):
    """Base class for factorized tensor adapters with factory pattern"""
    _registry: Dict[str, Type['FactorizedTensor']] = {}
    
    def __init_subclass__(cls, factorization_type: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if factorization_type is not None:
            cls._registry[factorization_type] = cls

    @classmethod
    def create(cls, factorization_type: str, hidden_size: int, rank: int, **kwargs) -> 'FactorizedTensor':
        """Factory method to create appropriate factorization instance"""
        if factorization_type not in cls._registry:
            raise ValueError(f"Unsupported factorization: {factorization_type}. "
                             f"Available: {list(cls._registry.keys())}")
        return cls._registry[factorization_type](hidden_size, rank, **kwargs)

    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank

    @abstractmethod
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return delta matrices for Q, K, V"""
        pass