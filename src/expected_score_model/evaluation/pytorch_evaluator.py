import shap
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from expected_score_model.evaluation.model_evaluator import ModelEvaluator
import matplotlib.pyplot as plt

class PyTorchEvaluator:
    
    def __init__(self, 
                 pytorch_model: nn.Module,
                 evaluator: ModelEvaluator):
        """ Given a PyTorch model and response type."""
        
        if (not isinstance(type(pytorch_model), nn.Module)) & (not issubclass(type(pytorch_model), nn.Module)):
            raise ValueError("pytorch_model must be a PyTorch model or subclas of nn.Module")
        else:
            self.pytorch_model = pytorch_model
            
        if (not isinstance(evaluator, ModelEvaluator)) & (not issubclass(evaluator, ModelEvaluator)):
            raise ValueError("evaluator must be a ModelEvaluator-type object")

        self.evaluator = evaluator(model = self.pytorch_model)
    
    def plot_feature_importance(self) -> Tuple[plt.Figure, plt.Axes]:
        pass
    
    def plot_shap_beeswarm(self, X: np.ndarray, specified_class: Optional[int] = None, max_display: int = 10, features = None) -> plt.Figure:
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
        
        explainer = shap.DeepExplainer(self.pytorch_model, X)
        shap_values = explainer.shap_values(X)
        
        if (len(self.pytorch_model.classes_) > 1) & (specified_class is not None):
            ind = list(self.pytorch_model.classes_).index(specified_class)
            return shap.summary_plot(shap_values[ind], X, feature_names=features, max_display=max_display)
        else:
            return shap.summary_plot(shap_values, X, feature_names=features, max_display=max_display)
