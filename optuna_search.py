import optuna
from main import main
class CustomEpochPruner(optuna.pruners.BasePruner):
    def __init__(self):
        # Define the epochs and their minimum acceptable validation accuracy.
        self.criteria = { # At epoch {key}, prune if val < {value}%
            10: 0.85, 
            25: 0.865, 
            35: 0.87,
        }

    def prune(self, study, trial):
        # Iterate over each specified epoch and threshold.
        for epoch, threshold in self.criteria.items():
            if epoch in trial.intermediate_values:
                if trial.intermediate_values[epoch] < threshold:
                    return True
        return False

def make_pruning_callback(trial):
    def pruning_callback(metric_value, epoch):
        trial.report(metric_value, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return pruning_callback


def objective(trial):
    """
    Objective function called by each Optuna trial.
    Samples hyperparameters and calls the main() function from
    the training script, returning the best accuracy.
    """
    
    # For rank, pick an integer range you want to explore, e.g., [2..32]
    rank1 = trial.suggest_categorical("rank1", [1, 2, 3, 4, 5, 8, 10, 12, 16, 24, 32])
    rank2 = trial.suggest_categorical("rank2", [1, 2, 3, 4, 5, 8, 10, 12, 16, 24, 32])
    rank3 = trial.suggest_categorical("rank3", [1, 2, 3, 4, 5, 8, 10, 12, 16, 24, 32])
    rank4 = trial.suggest_categorical("rank4", [1, 2, 3, 4, 5, 8, 10, 12, 16, 24, 32])

    # Now define a dictionary for all the args you want to pass to `main`.
    # Include both the hyperparameters you're tuning and any fixed arguments.
    training_args = {
        "seed": 123,
        "batch_size": 128,
        "num_epochs": 50,       # Or fewer if you want faster trials
        "num_workers": 2,
        "dataset": "fgvc_aircraft",
        "factorization": "tucker3",
        "compile_model": False,
        "checkpoint_path": None,
        
        # Hyperparameters from Optuna
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "rank": (rank1, rank2, rank3, rank4),
        
        # If using CosineAnnealingLR with a minimum LR:
        "eta_min": 5e-5,

        "pruning_callback": make_pruning_callback(trial),
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
    study = optuna.create_study(
        direction="maximize", 
        storage="sqlite:///study_params.db", 
        study_name="study_params", 
        load_if_exists=True,
        pruner=CustomEpochPruner(),
    )
    
    # Number of trials to run - adjust based on your compute budget
    study.optimize(objective, n_trials=50)
    
    print("Best trial:")
    print("  Value (accuracy):", study.best_trial.value)
    print("  Params:", study.best_trial.params)
