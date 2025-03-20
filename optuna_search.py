import optuna
from main import main

class CustomEpochPruner(optuna.pruners.BasePruner):
    def __init__(self):
        # Define the epochs and their minimum acceptable validation accuracy.
        self.criteria = {  # At epoch {key}, prune if val < {value}%
            25: 0.85, 
            35: 0.87,
        }

    def prune(self, study, trial):
        # Iterate over each specified epoch and threshold.
        for epoch, threshold in self.criteria.items():
            if epoch in trial.intermediate_values:
                if trial.intermediate_values[epoch] < threshold:
                    return True
        return False

class PruningCallback:
    def __init__(self, trial):
        self.trial = trial

    def __call__(self, metric_value, epoch):
        self.trial.report(metric_value, epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

def objective(trial):
    """
    Objective function called by each Optuna trial.
    Samples hyperparameters and calls the main() function from
    the training script, returning the best accuracy.
    """
    # For rank, pick an integer range you want to explore.
    rank1 = trial.suggest_categorical("rank1", [1, 2, 3, 4, 5, 8, 10, 12, 16, 24, 32])
    rank2 = trial.suggest_categorical("rank2", [1, 2, 3, 4, 5, 8, 10, 12, 16, 24, 32])
    rank3 = trial.suggest_categorical("rank3", [1, 2, 3, 4, 5, 8, 10, 12, 16, 24, 32])
    rank4 = trial.suggest_categorical("rank4", [1, 2, 3, 4, 5, 8, 10, 12, 16, 24, 32])

    # Define the arguments to pass to main.
    training_args = {
        "seed": 123,
        "batch_size": 128,
        "num_epochs": 50,       # Or fewer if you want faster trials
        "num_workers": 2,
        "dataset": "caltech_birds",
        "factorization": "htftucker",
        "compile_model": False,
        "checkpoint_path": None,
        # Hyperparameters from Optuna
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "rank": (rank1, rank2, rank3, rank4),
        # If using CosineAnnealingLR with a minimum LR:
        "eta_min": 5e-5,
        # Pass the top-level pruning callback instance.
        "pruning_callback": PruningCallback(trial),
    }

    # Call the training script’s main function (which should call pruning_callback appropriately)
    best_acc = main(**training_args)

    # We want to maximize accuracy, so return the accuracy
    return best_acc

if __name__ == "__main__":
    # Create an Optuna study with the custom pruner.
    study = optuna.create_study(
        direction="maximize", 
        storage="sqlite:///study_params.db", 
        study_name="study_params", 
        load_if_exists=True,
        pruner=CustomEpochPruner(),
    )
    
    # Number of trials to run – adjust based on your compute budget.
    study.optimize(objective, n_trials=50)
    
    print("Best trial:")
    print("  Value (accuracy):", study.best_trial.value)
    print("  Params:", study.best_trial.params)
