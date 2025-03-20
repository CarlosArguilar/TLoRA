import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Type, Dict, Tuple, Union, Optional

class FactorizedTensor(ABC, nn.Module):
    """Base class for factorized tensor adapters with factory pattern"""
    _registry: Dict[str, Type['FactorizedTensor']] = {}
    # The cache now stores a tuple (instance, next_id) for multi_layer types.
    _cache: Dict[Tuple[str, int, Union[int, Tuple[int, ...]]], Tuple['FactorizedTensor', int]] = {}

    def __init_subclass__(cls, factorization_type: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if factorization_type is not None:
            cls._registry[factorization_type] = cls

    @classmethod
    def create(cls, factorization_type: str, hidden_size: int, rank: Union[int, Tuple[int, ...]], **kwargs) -> Tuple['FactorizedTensor', Optional[int]]:
        """
        Factory method to create appropriate factorization instance.

        For factorization types including "multi_layer", the same instance is reused.
        In that case, the method returns a tuple (instance, id), where id is incremented
        by 1 on each call.
        """
        if factorization_type not in cls._registry:
            raise ValueError(f"Unsupported factorization: {factorization_type}. Available: {list(cls._registry.keys())}")

        if "multi_layer" in factorization_type:
            # Build a key based on the factorization type, hidden_size, and rank
            key = (factorization_type, hidden_size, rank)
            if key not in cls._cache:
                instance = cls._registry[factorization_type](hidden_size, rank, **kwargs)
                cls._cache[key] = (instance, 0)
            print(f'Creating multi layer factorization with key: {factorization_type}, {hidden_size}, {rank}, id:{cls._cache[key][1]}')
            instance, current_id = cls._cache[key]
            ret_id = current_id
            # Increment the counter for the next call
            cls._cache[key] = (instance, current_id + 1)
            return instance, ret_id
        else:
            # If not multi layer the layer id is None
            return cls._registry[factorization_type](hidden_size, rank, **kwargs), None

    def __init__(self, hidden_size: int, rank: Union[int, Tuple[int, ...]]):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank

    @abstractmethod
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return delta matrices for Q, K, V"""
        pass
