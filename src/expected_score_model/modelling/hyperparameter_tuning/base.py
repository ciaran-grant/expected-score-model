import optuna
from sklearn.metrics import log_loss

class BaseHyperparameterTuner:
    
    def __init__(self, training_data, response, param_grid):
        self.training_data = training_data
        self.response = response
        self.param_grid = param_grid

    def objective(self, trial):
        pass
    
    def get_objective_function(self):
        return self.objective

    def tune_hyperparameters(self):

        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.objective, n_trials=self.param_grid.trials)

        print("Number of finished trials: ", len(self.study.trials))
        print("Best trial:")
        trial = self.study.best_trial

        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return self.study

    def get_best_params(self):
        return self.study.best_params