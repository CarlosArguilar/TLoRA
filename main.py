from typing import Tuple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import ViTForImageClassification
from torchvision import datasets, transforms

from tlora.utils import replace_attention_layers, parse_args
from tlora.modified_layers import ModifiedViTSdpaSelfAttention
from tlora.datasets import DatasetFactory
from tlora.utils.checkpoint import save_checkpoint, load_checkpoint

torch.backends.cudnn.benchmark = True

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_model(device: torch.device, args, num_classes: int) -> ViTForImageClassification:
    """Create and configure ViT model with LoRA modifications"""
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    
    # Replace attention layers
    model = replace_attention_layers(model, ModifiedViTSdpaSelfAttention, args.factorization, args.rank)
    
    # Replace final classifier
    model.classifier = nn.Linear(768, num_classes)
    
    # Freeze all parameters except factorization and classifier
    for param in model.parameters():
        param.requires_grad = False
    
    for layer in model.vit.encoder.layer:
        if hasattr(layer.attention.attention, 'factorization'):
            for param in layer.attention.attention.factorization.parameters():
                param.requires_grad = True

    model.classifier.requires_grad_(True)

    model.to(device)

    if args.compile_model:
        print('Compiling model')
        model = torch.compile(model)

    return model

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

    print(f'Last LR: {scheduler.get_last_lr()[0]: .2e}')
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

def main(**kwargs):
    set_seed(kwargs.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    num_classes, train_set, test_set = DatasetFactory.create(kwargs.dataset)
    train_loader = DataLoader(train_set, batch_size=kwargs.batch_size, shuffle=True,
                             num_workers=kwargs.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=kwargs.batch_size*2,
                            num_workers=kwargs.num_workers, pin_memory=True)
    
    # Model setup
    model = create_model(device, kwargs, num_classes)
    print_trainable_params(model)
    
    # Optimization setup
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=kwargs.learning_rate,
        weight_decay=kwargs.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=kwargs.num_epochs, eta_min=kwargs.eta_min)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # Training state initialization
    start_epoch = 1
    best_acc = 0.0
    
    # Checkpoint loading
    if kwargs.checkpoint_path:
        start_epoch, best_acc = load_checkpoint(
            model, optimizer, scheduler, scaler, device, kwargs.checkpoint_path
        )
        print(f"Resuming training from epoch {start_epoch} with best acc {best_acc*100:.2f}%")

    # Training loop
    for epoch in range(start_epoch, kwargs.num_epochs+1):
        print(f"\nEpoch {epoch}/{kwargs.num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, 
                                criterion, scaler, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        # Update and save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_acc=best_acc,
                args=kwargs,
            )

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

    return best_acc

if __name__ == "__main__":
    args = parse_args()
    main(**args)