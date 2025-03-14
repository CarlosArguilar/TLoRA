from tlora.datasets.datasets import processor, DatasetFactory
from typing import Tuple, Optional
import torch
from datasets import load_dataset

class CaltechUCSDBirdsDataset(DatasetFactory, dataset_name="caltech_birds"):
    """Caltech-UCSD Birds-200-2011 dataset implementation following the factory pattern.
    
    Each example in this dataset consists of an image, a label, and a bounding box.
    We ignore the bounding box. The original splits are 'train' (5,994 instances) and 'test' (5,794 instances).
    """
    
    def get_splits(self) -> Tuple[int, torch.utils.data.Dataset, Optional[torch.utils.data.Dataset], torch.utils.data.Dataset]:
        # Define a transform that processes the image using the provided processor.
        transform = lambda x: processor(x["image"], return_tensors="pt")["pixel_values"].squeeze(0)
        
        # Load the dataset from Hugging Face.
        dataset = load_dataset("bentrevett/caltech-ucsd-birds-200-2011", cache_dir=self.root)
        
        # Apply a transformation that returns a dict with "image" and "label".
        train_dataset = dataset["train"].with_transform(lambda x: {"image": transform(x), "label": x["label"]})
        test_dataset  = dataset["test"].with_transform(lambda x: {"image": transform(x), "label": x["label"]})
        
        # Set the format so that __getitem__ returns a tuple (image, label) instead of a dict.
        train_dataset.set_format("torch", columns=["image", "label"], output_all_columns=False)
        test_dataset.set_format("torch", columns=["image", "label"], output_all_columns=False)
        
        # Retrieve the number of classes from the label feature.
        num_classes = len(dataset["train"].features["label"].names)
        
        return num_classes, train_dataset, None, test_dataset