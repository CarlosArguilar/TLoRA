from typing import Tuple
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import ViTForImageClassification
from torchvision import datasets, transforms

from tlora.utils import replace_attention_layers
from tlora.modified_layers import ModifiedViTSdpaSelfAttention

torch.backends.cudnn.benchmark = True

def parse_args():
    """Parse command-line arguments with default values"""
    parser = argparse.ArgumentParser(description='Train ViT with TLoRA')
    
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Input batch size for training (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=20,
                       help='Number of epochs to train (default: 20)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                       help='Weight decay (default: 1e-2)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers for data loading (default: 4)')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed (default: 123)')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                       help='Path to load checkpoint (default: None)')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Get available compute device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_datasets() -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Create CIFAR-10 datasets with appropriate transforms"""
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return (
        datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform),
        datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    )

def create_model(device: torch.device, args) -> ViTForImageClassification:
    """Create and configure ViT model with LoRA modifications"""
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    
    # Replace attention layers
    model = replace_attention_layers(model, ModifiedViTSdpaSelfAttention)
    
    # Replace final classifier
    model.classifier = nn.Linear(768, 10)

    if args.checkpoint_path:
        print(f"Loading model from checkpoint: {args.checkpoint_path}")

        # Load the checkpoint
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # Load the state dict into the model
        model.load_state_dict(checkpoint)

    
    # Freeze all parameters except factorization and classifier
    for param in model.parameters():
        param.requires_grad = False
    
    for layer in model.vit.encoder.layer:
        if hasattr(layer.attention.attention, 'factorization'):
            for param in layer.attention.attention.factorization.parameters():
                param.requires_grad = True

    model.classifier.requires_grad_(True)

    return model.to(device)

def print_trainable_params(model: nn.Module):
    """Print ratio of trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")

def train_epoch(model: nn.Module, loader: DataLoader, 
                optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
                criterion: nn.Module, scaler: GradScaler, device: torch.device) -> float:
    """Single training epoch"""
    model.train()
    total_loss = 0.0
    
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        with autocast(device_type=(str(model.device))):
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        total_loss += loss.item() * inputs.size(0)
    
    scheduler.step()
    return total_loss / len(loader.dataset)

@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, 
             criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Model evaluation"""
    model.eval()
    total_loss = 0.0
    correct = 0
    
    for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        with autocast(device_type=(str(model.device))):
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
        
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
    
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def main():
    args = parse_args()

    # Initialization
    set_seed(args.seed)
    device = get_device()
    
    # Data loading
    train_set, test_set = create_datasets()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size*2,
                            num_workers=args.num_workers, pin_memory=True)
    
    # Model setup
    model = create_model(device, args)
    print_trainable_params(model)
    
    # Optimization setup
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    best_acc = 0.0
    
    # Training loop
    for epoch in range(1, args.num_epochs+1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, 
                                criterion, scaler, device)
        
        # Evaluation
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

if __name__ == "__main__":
    main()