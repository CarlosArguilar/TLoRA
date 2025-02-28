import argparse

def parse_rank(s):
    try:
        if ',' in s:
            parts = s.split(',')
            if len(parts) != 3:
                raise ValueError("Rank tuple must have exactly 3 integers")
            return tuple(int(p.strip()) for p in parts)
        else:
            return int(s)
    except Exception as e:
        raise argparse.ArgumentTypeError("Rank must be an integer or a comma-separated tuple of 3 integers")


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
    parser.add_argument('--factorization', type=str, default='cp',
                       help='Method of tensor factorization (default: cp)')
    parser.add_argument('--rank', type=parse_rank, default=8,
                        help='rank param for tensor factorization. \
                            Provide a single integer (e.g., 8) or a tuple as comma separated values (e.g., "8,8,8").')
    parser.add_argument('--compile-model', action='store_true',
                        help='Enable model compilation (default: False)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset to be used.')
    return parser.parse_args()