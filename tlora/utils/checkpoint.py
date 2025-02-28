from __future__ import annotations
import torch

def get_checkpoint_name(args) -> str:
    """Generate standardized checkpoint name based on training parameters"""
    # Handle rank (could be single int or tuple)
    rank_str = "_".join(map(str, args.rank)) if isinstance(args.rank, (tuple, list)) else str(args.rank)
    
    # Clean factorization name for filename safety
    factorization = args.factorization.lower().replace("-", "")
    
    # Clean dataset name
    dataset = args.dataset.lower().replace("/", "_")
    
    return f"vit_{factorization}_rank{rank_str}_{dataset}_best.pth"

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
                   scaler: GradScaler, epoch: int, best_acc: float, args, filename: str = None):
    """Save training checkpoint containing only trainable parameters and training state"""
    # Collect trainable parameters

    if filename is None:
        filename = get_checkpoint_name(args)

    trainable_params = {name for name, param in model.named_parameters() if param.requires_grad}
    trainable_state_dict = {k: v for k, v in model.state_dict().items() if k in trainable_params}
    
    checkpoint = {
        'trainable_model_state_dict': trainable_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc,
        'args': vars(args)
    }
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint to {filename}")

def load_checkpoint(model: nn.Module, 
                    optimizer: optim.Optimizer, 
                    scheduler: optim.lr_scheduler._LRScheduler,
                    scaler: GradScaler, 
                    device: torch.device, 
                    filename: str) -> Tuple[int, float]:
    """Load training checkpoint and return updated training state"""
    print(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location=device)

    # Identify trainable parameters in the *current* model
    trainable_params = {name for name, param in model.named_parameters() if param.requires_grad}
    
    # Get full current state_dict of the model
    model_state_dict = model.state_dict()

    # Update only the trainable parameters from the checkpoint
    for name, param in checkpoint['trainable_model_state_dict'].items():
        if name in trainable_params:
            model_state_dict[name] = param

    # Load updated state_dict back into the model
    model.load_state_dict(model_state_dict)

    # Load optimizer, scheduler, scaler states
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint['epoch'], checkpoint['best_acc']