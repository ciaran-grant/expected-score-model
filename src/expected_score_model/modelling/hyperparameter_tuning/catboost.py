import optuna
import catboost
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

        param = {
            "depth" : trial.suggest_int("depth",
                                        self.param_grid.max_depth_min,
                                        self.param_grid.max_depth_max,
                                        step=self.param_grid.max_depth_step),
            "min_child_samples" : trial.suggest_int("min_child_samples", 
                                                self.param_grid.min_child_weight_min,
                                                self.param_grid.min_child_weight_max,
                                                step=self.param_grid.min_child_weight_step),
            "eta" : trial.suggest_float("eta",
                                        self.param_grid.eta_min, 
                                        self.param_grid.eta_max, 
                                        log=True),
            "reg_lambda": trial.suggest_float("reg_lambda",
                                        self.param_grid.lambda_min,
                                        self.param_grid.lambda_max, 
                                        log=True),
            "subsample": trial.suggest_float("subsample", 
                                            self.param_grid.subsample_min, 
                                            self.param_grid.subsample_max),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel",
                                                    self.param_grid.colsample_bytree_min, 
                                                    self.param_grid.colsample_bytree_max),
        } 

        cat_features = list(train_x.select_dtypes(include='object').columns)

        if self.param_grid.error == "RMSE":
            cb = catboost.CatBoostRegressor(
                iterations=self.param_grid.iterations,
                depth = param['depth'],
                min_child_samples = param['min_child_samples'],
                eta = param['eta'],
                reg_lambda = param['reg_lambda'],
                colsample_bylevel = param['colsample_bylevel'],
                cat_features=cat_features,
                loss_function=self.param_grid.error, 
                eval_metric=self.param_grid.error,
                verbose = self.param_grid.verbosity
            )
            cb.fit(train_x, train_y)
            preds = cb.predict(valid_x)
            return mean_squared_error(preds, valid_y, squared=False)

        if self.param_grid.error in ["Logloss", "MultiClass"]:
            cb = catboost.CatBoostClassifier(
                iterations=self.param_grid.iterations,
                depth = param['depth'],
                min_child_samples = param['min_child_samples'],
                eta = param['eta'],
                reg_lambda = param['reg_lambda'],
                colsample_bylevel = param['colsample_bylevel'],
                cat_features=cat_features,
                loss_function=self.param_grid.error, 
                eval_metric=self.param_grid.error,
                verbose = self.param_grid.verbosity
            )
            cb.fit(train_x, train_y)
            probas = cb.predict_proba(valid_x)
            return log_loss(valid_y, probas)

