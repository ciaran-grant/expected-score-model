import optuna
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import train_test_split

from expected_score_model.modelling.hyperparameter_tuning.base import BaseHyperparameterTuner

class HyperparameterTuner(BaseHyperparameterTuner):
    
    def __init__(self, training_data, response, param_grid):
        self.training_data = training_data
        self.response = response
        self.param_grid = param_grid

    def objective(self, trial):

        train_x, valid_x, train_y, valid_y = train_test_split(self.training_data, self.response, test_size=self.param_grid.validation_size)
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dvalid = xgb.DMatrix(valid_x, label=valid_y)

        param = {
            "verbosity": self.param_grid.verbosity,
            'objective': self.param_grid.error,
            'num_class': self.param_grid.num_class,
            # maximum depth of the tree, signifies complexity of the tree.
            "max_depth" : trial.suggest_int("max_depth",
                                            self.param_grid.max_depth_min,
                                            self.param_grid.max_depth_max,
                                            step=self.param_grid.max_depth_step),
            # minimum child weight, larger the term more conservative the tree.
            "min_child_weight" : trial.suggest_int("min_child_weight", 
                                                self.param_grid.min_child_weight_min,
                                                self.param_grid.min_child_weight_max,
                                                step=self.param_grid.min_child_weight_step),
            "eta" : trial.suggest_float("eta",
                                        self.param_grid.eta_min, 
                                        self.param_grid.eta_max, 
                                        log=True),
            # defines how selective algorithm is.
            "gamma" : trial.suggest_float("gamma", 
                                        self.param_grid.gamma_min, 
                                        self.param_grid.gamma_max, 
                                        log=True),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda",
                                        self.param_grid.lambda_min,
                                        self.param_grid.lambda_max, 
                                        log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 
                                        self.param_grid.alpha_min,
                                        self.param_grid.alpha_max,
                                        log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 
                                            self.param_grid.subsample_min, 
                                            self.param_grid.subsample_max),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree",
                                                    self.param_grid.colsample_bytree_min, 
                                                    self.param_grid.colsample_bytree_max),
        }        

        bst = xgb.train(param, dtrain)
        
        if self.param_grid.error == "reg:squarederror":
            preds = bst.predict(dvalid)
            return mean_squared_error(preds, valid_y, squared=False)
        if self.param_grid.error == "binary:logistic":
            probas = bst.predict(dvalid)
            return log_loss(valid_y, probas)
        if self.param_grid.error == "multi:softprob":
            probas = bst.predict(dvalid)
            return log_loss(valid_y, probas)
