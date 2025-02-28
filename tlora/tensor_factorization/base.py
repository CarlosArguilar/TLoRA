import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Type, Dict, Tuple, Union

class FactorizedTensor(ABC, nn.Module):
    """Base class for factorized tensor adapters with factory pattern"""
    _registry: Dict[str, Type['FactorizedTensor']] = {}
    
    def __init_subclass__(cls, factorization_type: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if factorization_type is not None:
            cls._registry[factorization_type] = cls

    @classmethod
    def create(cls, factorization_type: str, hidden_size: int, rank: Union[int, Tuple[int, int, int]], **kwargs) -> 'FactorizedTensor':
        """Factory method to create appropriate factorization instance
        
        Parameters:
            factorization_type (str): The type of factorization.
            hidden_size (int): The hidden size of the attention mechanism.
            rank (Union[int, Tuple[int, int, int]]): Either an integer (all ranks are the same or only 1 rank)
                                                     or a tuple of three integers.
        """
        if factorization_type not in cls._registry:
            raise ValueError(f"Unsupported factorization: {factorization_type}. "
                             f"Available: {list(cls._registry.keys())}")
        
        return cls._registry[factorization_type](hidden_size, rank, **kwargs)

    def __init__(self, hidden_size: int, rank: Union[int, Tuple[int, int, int]]):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank

    @abstractmethod
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return delta matrices for Q, K, V"""
        pass