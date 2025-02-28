import torch
from transformers import ViTImageProcessorFast
from typing import Tuple, Dict, Callable
from functools import wraps

# Global processor for image transformations
processor = ViTImageProcessorFast.from_pretrained("google/vit-base-patch16-224")

# Dataset registry dictionary
_dataset_registry: Dict[str, Callable] = {}

def register_dataset(func: Callable) -> Callable:
    """Decorator to automatically register dataset creation functions"""
    _dataset_registry[func.__name__] = func
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def create_dataset(
    name: str,
    root: str = "./data",
    **kwargs
) -> Tuple[torch.utils.data.Dataset, ...]:
    """
    Main factory function to create datasets
    
    Args:
        name: Name of registered dataset (e.g., 'cifar10')
        root: Root directory for data storage
        **kwargs: Dataset-specific parameters (e.g., validation_split)
        
    Returns:
        Tuple of datasets (train, test) or (train, val, test)
    """
    name = name.lower()
    if name not in _dataset_registry:
        available = list(_dataset_registry.keys())
        raise ValueError(f"Dataset '{name}' not found. Available: {available}")
    
    return _dataset_registry[name](root=root, **kwargs)

# Example usage
if __name__ == "__main__":
    # Basic usage
    train, test = create_dataset("cifar10", root="./custom_data")
    
    # With validation split
    train, val, test = create_dataset("cifar100", validation_split=0.2)
    
    print(f"CIFAR-10: {len(train)} train, {len(test)} test")
    print(f"CIFAR-100: {len(train)} train, {len(val)} val, {len(test)} test")