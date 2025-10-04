import pandas as pd
import numpy as np
import warnings
import catboost
import joblib
from catboost import CatBoostClassifier
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from expected_score_model.evaluation.model_evaluator import ModelEvaluator
from expected_score_model.evaluation.classification_evaluator import ClassificationModelEvaluator
from expected_score_model.evaluation.regression_evaluator import RegressionModelEvaluator
from expected_score_model.evaluation.catboost_evaluator import CatBoostEvaluator
from expected_score_model.preprocessing.preprocessor import ExpectedScorePreprocessor
from expected_score_model.preprocessing.response import create_expected_score_response
from expected_score_model.modelling.hyperparameter_tuning.catboost import HyperparameterTuner

warnings.filterwarnings('ignore')

# Load the data
chains = pd.read_csv('/Users/ciaran/Documents/Projects/AFL/git-repositories/expected-score-model/data/AFL_Chains.csv')
chains = create_expected_score_response(chains)

# Preprocessing
xs_preproc = ExpectedScorePreprocessor()
X_shots, y_shots = xs_preproc.fit_transform(chains, chains['result'])

# Feature Selection
selected_features = ['distance', 'angle', 'set_shot', 'distance_since_last_action']
X_shots_selected = X_shots[selected_features]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_shots_selected, y_shots, test_size=0.2, random_state=42)

# Hyperparameter Tuning
@dataclass
class OptunaCatBoostParamGrid:
    trials: int = 10
    iterations: int = 1000
    error: str = 'MultiClass'
    eta_min: float = 0.01
    eta_max: float = 1.0
    max_depth_min: int = 2
    max_depth_max: int = 10
    max_depth_step: int = 1
    min_child_weight_min: int = 2
    min_child_weight_max: int = 100
    min_child_weight_step: int = 1
    lambda_min: float = 0.0001
    lambda_max: float = 10
    subsample_min: float = 0.2
    subsample_max: float = 0.9
    subsample_step: float = 0.05
    colsample_bytree_min: float = 0.2
    colsample_bytree_max: float = 0.9
    colsample_bytree_step: float = 0.05
    validation_size: float = 0.2
    verbosity: int = 100
catboost_param_grid = OptunaCatBoostParamGrid()

cb_tuner = HyperparameterTuner(training_data=X_train, response = y_train, param_grid=catboost_param_grid)
cb_tuner.tune_hyperparameters()
best_params = cb_tuner.get_best_params()

# Fitting Model
cat_features = list(X_train.select_dtypes(include='object').columns)
model = CatBoostClassifier(
        iterations=catboost_param_grid.iterations,
        depth = best_params['depth'],
        min_child_samples = best_params['min_child_samples'],
        eta = best_params['eta'],
        reg_lambda = best_params['reg_lambda'],
        colsample_bylevel = best_params['colsample_bylevel'],
        cat_features=cat_features,
        loss_function=catboost_param_grid.error, 
        eval_metric=catboost_param_grid.error,
        verbose = catboost_param_grid.verbosity
)
model.fit(X_train, y_train)

# Export Model
joblib.dump(model, '/Users/ciaran/Documents/Projects/AFL/git-repositories/expected-score-model/model_outputs/models/catboost_model.joblib')