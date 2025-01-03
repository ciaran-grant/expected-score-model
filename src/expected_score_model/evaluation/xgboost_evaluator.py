import shap
import xgboost as xgb
import numpy as np
from typing import Tuple, Optional
from expected_score_model.evaluation.model_evaluator import ModelEvaluator
import matplotlib.pyplot as plt

class XGBoostEvaluator:
    
    def __init__(self, 
                 xgb_model: xgb.XGBClassifier or xgb.XGBRegressor,
                 evaluator: ModelEvaluator):
        """ Given a XGBoost model and response type."""
        
        if not (isinstance(xgb_model, (xgb.XGBClassifier, xgb.XGBRegressor))):
            raise ValueError("xgb_model must be a xgb.XGBClassifier or xgb.XGBRegressor object")
        else:
            self.xgb_model = xgb_model

        if (not isinstance(evaluator, ModelEvaluator)) & (not issubclass(evaluator, ModelEvaluator)):
            raise ValueError("evaluator must be a ModelEvaluator-type object")
        else:
            self.evaluator = evaluator(model = xgb_model)
    
    def plot_feature_importance(self) -> plt.Axes:
        """
        Plots the feature importance of the CatBoost model.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes objects of the plot.
        """
        
        ax = xgb.plot_importance(self.xgb_model)
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Feature')
        ax.set_title('XGBoost Feature Importance')
        
        return ax
    
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
        
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer(X, y)
        
        if (len(self.xgb_model.classes_) > 1) & (specified_class is not None):
            ind = list(self.xgb_model.classes_).index(specified_class)
            return shap.plots.beeswarm(shap_values[..., ind], max_display=max_display)
        else:
            return shap.plots.beeswarm(shap_values, max_display=max_display)
