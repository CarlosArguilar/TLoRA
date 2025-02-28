import torch
from torchvision import datasets
from transformers import ViTImageProcessorFast
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Type, Optional

processor = ViTImageProcessorFast.from_pretrained("google/vit-base-patch16-224")

class DatasetFactory(ABC):
    """Base class for dataset factories with automatic registration"""
    _registry: Dict[str, Type['DatasetFactory']] = {}
    
    def __init_subclass__(cls, dataset_name: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if dataset_name is not None:
            cls._registry[dataset_name.lower()] = cls
            
    def __init__(self, 
                 root: str = "./data",
                 download: bool = True,
                 validation_split: Optional[float] = None):
        self.root = root
        self.download = download
        self.validation_split = validation_split
        
    @classmethod
    def create(cls, 
               name: str,
               **kwargs) -> Tuple[torch.utils.data.Dataset, ...]:
        """Factory method to create dataset splits"""
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(cls._registry.keys())}")
            
        instance = cls._registry[name](**kwargs)
        return instance.get_splits()
    
    @abstractmethod
    def get_splits(self) -> Tuple[torch.utils.data.Dataset, ...]:
        """Return dataset splits (train, test) or (train, val, test)"""
        pass



# Usage example
if __name__ == "__main__":
    # Get basic splits
    train, test = DatasetFactory.create("cifar10", root="./custom_data")
    
    # Get splits with validation
    train, val, test = DatasetFactory.create("cifar100", validation_split=0.2)
    
    print(f"CIFAR-10: {len(train)} train, {len(test)} test")
    print(f"CIFAR-100: {len(train)} train, {len(val)} val, {len(test)} test")