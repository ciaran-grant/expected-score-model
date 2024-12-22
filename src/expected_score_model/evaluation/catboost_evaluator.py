import shap
import catboost
import numpy as np
from typing import Tuple, Optional
from expected_score_model.evaluation.model_evaluator import ModelEvaluator
import matplotlib.pyplot as plt

class CatBoostEvaluator:
    
    def __init__(self, 
                 catboost_model: catboost.CatBoost,
                 evaluator: ModelEvaluator):
        """ Given a CatBoost model and response type."""
        
        if not isinstance(catboost_model, catboost.CatBoost):
            raise ValueError("catboost_model must be a CatBoost model")
        else:
            self.catboost_model = catboost_model
            
        if (not isinstance(evaluator, ModelEvaluator)) & (not issubclass(evaluator, ModelEvaluator)):
            raise ValueError("evaluator must be a ModelEvaluator-type object")

        self.evaluator = evaluator(model = self.catboost_model)
    
    def plot_feature_importance(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the feature importance of the CatBoost model.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes objects of the plot.
        """
        
        # Assuming you have a trained CatBoost model called 'model'
        feature_importance = self.catboost_model.get_feature_importance()

        # Get the feature names from your dataset
        feature_names = self.catboost_model.feature_names_

        # Sort the feature importance values and feature names in descending order
        sorted_indices = np.argsort(feature_importance)[::-1]
        sorted_feature_importance = feature_importance[sorted_indices]
        sorted_feature_names = [feature_names[i] for i in sorted_indices]

        # Plot the feature importance using fig, ax
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(sorted_feature_importance)), sorted_feature_importance, align='center')
        ax.set_yticks(range(len(sorted_feature_importance)))
        ax.set_yticklabels(sorted_feature_names)
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Feature')
        ax.set_title('CatBoost Feature Importance')
        
        return fig, ax
    
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
        
        explainer = shap.TreeExplainer(self.catboost_model)
        shap_values = explainer(X, y)
        
        if (len(self.catboost_model.classes_) > 1) & (specified_class is not None):
            ind = list(self.catboost_model.classes_).index(specified_class)
            return shap.plots.beeswarm(shap_values[..., ind], max_display=max_display)
        else:
            return shap.plots.beeswarm(shap_values, max_display=max_display)
