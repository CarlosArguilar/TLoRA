from typing import Tuple, Optional
import torch
from tlora.datasets.datasets import processor, DatasetFactory

class CaltechUCSDBirdsDataset(DatasetFactory, dataset_name="caltech_birds"):
    """Caltech-UCSD Birds-200-2011 dataset implementation following factory pattern"""

    def get_splits(self) -> Tuple[int, torch.utils.data.Dataset, Optional[torch.utils.data.Dataset], torch.utils.data.Dataset]:
        from datasets import load_dataset

        # Load the Hugging Face dataset
        dataset_dict = load_dataset("bentrevett/caltech-ucsd-birds-200-2011")

        # A simple transform using your `processor` to get an image tensor
        transform = lambda img: processor(img, return_tensors="pt")["pixel_values"].squeeze(0)

        # Create a thin wrapper to transform Hugging Face dataset entries into (image, label) tuples
        class HFToTorchDataset(torch.utils.data.Dataset):
            def __init__(self, hf_dataset, transform):
                self.hf_dataset = hf_dataset
                self.transform = transform

            def __len__(self):
                return len(self.hf_dataset)

            def __getitem__(self, idx):
                example = self.hf_dataset[idx]
                image = example["image"]
                label = example["label"]
                if self.transform:
                    image = self.transform(image)
                return (image, label)

        # Build train/test splits
        train_hf = dataset_dict["train"]
        test_hf = dataset_dict["test"]

        train_ds = HFToTorchDataset(train_hf, transform)
        test_ds = HFToTorchDataset(test_hf, transform)

        # Number of classes (should be 200)
        num_classes = len(train_hf.features["label"].names)

        val_ds = None

        return num_classes, train_ds, val_ds, test_ds
