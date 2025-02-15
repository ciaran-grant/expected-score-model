from expected_score_model.modelling.hyperparameter_tuning.base import BaseHyperparameterTuner
from expected_score_model.modelling.supermodel.supernn import MulticlassNeuralNetwork
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class HyperparameterTuner(BaseHyperparameterTuner):
    
    def __init__(self, training_data, response, param_grid):
        self.training_data = training_data
        self.response = response
        self.param_grid = param_grid

    def objective(self, trial):
        
        train_x, valid_x, train_y, valid_y = train_test_split(self.training_data, self.response, test_size=self.param_grid.validation_size)

        # Convert the data to PyTorch tensors
        X_train_tensor = torch.tensor(train_x, dtype=torch.float32)
        y_train_tensor = torch.tensor(train_y.values, dtype=torch.long)

        # Create a data loader for batching the data
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.param_grid.batch_size, shuffle=True)
        
        # Convert the data to PyTorch tensors
        X_val_tensor = torch.tensor(valid_x, dtype=torch.float32)
        y_val_tensor = torch.tensor(valid_y.values, dtype=torch.long)

        # Create a data loader for batching the data
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.param_grid.batch_size, shuffle=True)
        
        # Define the hyperparameters to tune
        hidden_size = trial.suggest_categorical("hidden_size", self.param_grid.hidden_size)
        learning_rate = trial.suggest_float("learning_rate",
                                        self.param_grid.learning_rate_min, 
                                        self.param_grid.learning_rate_max, 
                                        log=True)
        # num_layers = trial.suggest_int("num_layers",
        #                                self.param_grid.num_layers_min,
        #                                self.param_grid.num_layers_max)
        # dropout_rate = trial.suggest_float("dropout_rate",
        #                                self.param_grid.dropout_rate_min,
        #                                self.param_grid.dropout_rate_max)

        # Create the model with the suggested hyperparameters
        input_size = X_train_tensor.shape[1]
        output_size = len(np.unique(self.response))
        criterion = nn.CrossEntropyLoss()
        model = MulticlassNeuralNetwork(input_size, hidden_size, output_size, criterion)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        for _ in range(self.param_grid.num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate the model on the validation set
        with torch.no_grad():
            model.eval()
            output = model(X_val_tensor)
            val_loss = criterion(output, y_val_tensor)

        return val_loss  # Return the validation loss as the objective value
    
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