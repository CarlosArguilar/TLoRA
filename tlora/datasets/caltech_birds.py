from tlora.datasets.datasets import processor, DatasetFactory
from typing import Tuple
import torch
from datasets import load_dataset

class CaltechUCSDBirdsDataset(DatasetFactory, dataset_name="caltech_birds"):
    """Caltech-UCSD Birds-200-2011 dataset implementation following the factory pattern.
    
    Each example in this dataset consists of an image, a label, and a bounding box.
    The original splits are 'train' (5,994 instances) and 'test' (5,794 instances).
    Here, we further split the training set into train and validation subsets.
    """
    
    def get_splits(self) -> Tuple[int, torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        # Define a transform that processes the image using the given processor.
        transform = lambda x: processor(x["image"], return_tensors="pt")["pixel_values"].squeeze(0)
        
        # Load the dataset from Hugging Face.
        dataset = load_dataset("bentrevett/caltech-ucsd-birds-200-2011", cache_dir=self.root)
        
        # Apply a transformation that returns only the "image" and "label" fields.
        train_dataset = dataset["train"].with_transform(lambda x: {"image": transform(x), "label": x["label"]})
        test_dataset  = dataset["test"].with_transform(lambda x: {"image": transform(x), "label": x["label"]})
        
        # Retrieve the number of classes from the label feature.
        num_classes = len(dataset["train"].features["label"].names)
        
        return num_classes, train_dataset, None, test_dataset