import optuna
from main import main  # <-- Import the main function from your training file

def objective(trial):
    """
    Objective function called by each Optuna trial.
    Samples hyperparameters and calls the main() function from
    the training script, returning the best accuracy.
    """

    # Suggest values for the hyperparameters you want to search
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    
    # For rank, pick an integer range you want to explore, e.g., [2..32]
    rank = trial.suggest_int("rank", 2, 32, step=2)

    # Now define a dictionary for all the args you want to pass to `main`.
    # Include both the hyperparameters you're tuning and any fixed arguments.
    training_args = {
        "seed": 123,
        "batch_size": 128,
        "num_epochs": 6,       # Or fewer if you want faster trials
        "num_workers": 4,
        "dataset": "cifar10",
        "factorization": "cp",
        "compile_model": False,
        "checkpoint_path": None,
        
        # Hyperparameters from Optuna
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "rank": rank,
        
        # If using CosineAnnealingLR with a minimum LR:
        "eta_min": 1e-6,
    }

    # Call the training script’s main function
    # which we modified to return best_acc
    best_acc = main(**training_args)

    # We want to maximize accuracy, so return the accuracy
    return best_acc

if __name__ == "__main__":
    # For reproducible search results, you can pass a seed.
    # But note that Optuna itself sometimes uses random sampling.
    # study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=123))
    
    # Or you can skip the seed in the sampler for more variability:
    study = optuna.create_study(direction="maximize")
    
    # Number of trials to run - adjust based on your compute budget
    study.optimize(objective, n_trials=10)
    
    print("Best trial:")
    print("  Value (accuracy):", study.best_trial.value)
    print("  Params:", study.best_trial.params)
