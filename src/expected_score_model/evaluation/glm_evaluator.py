import shap
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.linear_model import LogisticRegression
from expected_score_model.evaluation.model_evaluator import ModelEvaluator
import matplotlib.pyplot as plt

class LogisticEvaluator:
    
    def __init__(self, 
                 lr_model: LogisticRegression,
                 evaluator: ModelEvaluator):
        """ Given a LogistcRegression model and response type."""
        
        if not (isinstance(lr_model, LogisticRegression)):
            raise ValueError("lr_model must be a LogisticRegression object")
        else:
            self.lr_model = lr_model

        if (not isinstance(evaluator, ModelEvaluator)) & (not issubclass(evaluator, ModelEvaluator)):
            raise ValueError("evaluator must be a ModelEvaluator-type object")
        else:
            self.evaluator = evaluator(model = lr_model)
    
    def get_feature_importance(self) -> plt.Axes:
        """
        Gets the feature importance of the Logistic Regression model.

        Returns:
            pd.DataFrame: The coeffients for the Logistic Regression.
        """
        
        # Get the coefficients from the lr model
        coefficients = self.lr_model.coef_

        # Get the feature names from the model
        feature_names = self.lr_model.feature_names_in_

        # Create a dataframe of the coefficients
        coefficients_df = pd.DataFrame(coefficients, columns=feature_names)

        # Transpose the dataframe for better readability
        coefficients_df = coefficients_df.T

        # # Rename the column to 'Coefficient'
        coefficients_df.columns = [f'Coefficient_{x}' for x in self.lr_model.classes_]

        # # Reset the index
        coefficients_df = coefficients_df.reset_index()

        # # Rename the index column to 'Feature'
        coefficients_df = coefficients_df.rename(columns={'index': 'Feature'})
        
        return coefficients_df
    
    def plot_shap_beeswarm(self, X: np.ndarray, y: np.ndarray, specified_class: Optional[int] = None, max_display: int = 10) -> plt.Figure:
        """
        Plots a SHAP beeswarm plot for the CatBoost model.

        Args:
            X (np.ndarray): The feature data to compute SHAP values for.
            y (np.ndarray): The target data to compute SHAP values for.
            specified_class (Optional[int]): The class index to plot for multi-class models. Defaults to None.
            max_display (int): The maximum number of features to display. Defaults to 10.

        Returns:
            plt.Figure: The figure object of the SHAP beeswarm plot.
        """
        
        explainer = shap.Explainer(self.lr_model, X, feature_names=self.lr_model.feature_names_in_)
        shap_values = explainer(X)
        
        if (len(self.lr_model.classes_) > 1) & (specified_class is not None):
            ind = list(self.lr_model.classes_).index(specified_class)
            return shap.plots.beeswarm(shap_values[:, :, ind], max_display=max_display)
        else:
            return shap.plots.beeswarm(shap_values, max_display=max_display)
