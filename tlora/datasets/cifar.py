from tlora.datasets.datasets import register_dataset, processor
from typing import Tuple, Optional
from torchvision import datasets
import torch

@register_dataset
def cifar10(
    root: str = "./data",
    download: bool = True,
    validation_split: Optional[float] = None
) -> Tuple[torch.utils.data.Dataset, ...]:
    """
    Returns: (train, test) or (train, val, test) if validation_split is specified
    """
    transform = lambda img: processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
    
    train = datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
    test = datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
    
    if validation_split:
        train, val = torch.utils.data.random_split(train, [1 - validation_split, validation_split])
        return train, val, test
        
    return train, test

@register_dataset
def cifar100(
    root: str = "./data",
    download: bool = True,
    validation_split: Optional[float] = None
) -> Tuple[torch.utils.data.Dataset, ...]:
    """Same interface as cifar10 but for CIFAR-100"""
    transform = lambda img: processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
    
    train = datasets.CIFAR100(root=root, train=True, download=download, transform=transform)
    test = datasets.CIFAR100(root=root, train=False, download=download, transform=transform)
    
    if validation_split:
        train, val = torch.utils.data.random_split(train, [1 - validation_split, validation_split])
        return train, val, test
        
    return train, test