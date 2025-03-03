from tlora.datasets.datasets import processor, DatasetFactory
from typing import Tuple, Optional
from torchvision import datasets
import torch


class FGVCAircraftDataset(DatasetFactory, dataset_name="fgvc_aircraft"):
    """FGVC-Aircraft implementation following factory pattern"""
    
    def get_splits(self) -> Tuple[int, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        transform = lambda img: processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
        
        train = datasets.FGVCAircraft(
            root=self.root,
            split="train",  # Using a string-based split identifier
            download=self.download,
            transform=transform
        )
        
        test = datasets.FGVCAircraft(
            root=self.root,
            split="test",
            download=self.download,
            transform=transform
        )

        val = datasets.FGVCAircraft(
            root=self.root,
            split="val",
            download=self.download,
            transform=transform
        )
        
        num_classes = len(train.classes)
        return num_classes, train, val, test