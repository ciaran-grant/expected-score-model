import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from typing import Tuple, Optional

class ModelEvaluator:
    """
    A class used to evaluate a model's performance by comparing actual and predicted values.

    Attributes:
    ----------
    model : object
        The model to be evaluated.
    """
    
    def __init__(self, model):
        self.model = model

    def averages(self, actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
        """
        Calculates the mean of actual and predicted values.

        Parameters:
        ----------
        actual : pandas.Series
            The actual values.
        predicted : pandas.Series
            The predicted values.

        Returns:
        -------
        dict
            A dictionary containing the mean of actual and predicted values.
        """
        return {
            'actual': actual.mean(),
            'predicted': predicted.mean()
        }

    def _get_feature_plot_data(self, actual: pd.Series, predicted: pd.Series, feature: pd.Series) -> pd.DataFrame:
        """
        Creates a DataFrame for plotting feature data.

        Parameters:
        ----------
        actual : pandas.Series
            The actual values.
        predicted : pandas.Series
            The predicted values.
        feature : pandas.Series
            The feature values.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame containing actual, predicted, and feature values.
        """
        return pd.DataFrame({
            'actual':actual,
            'predicted':predicted,
            'feature':feature
            }
        )

    def _group_feature(self, data: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
        """
        Groups a feature into bins.

        Parameters:
        ----------
        data : pandas.DataFrame
            The DataFrame containing the feature to be binned.
        bins : int, optional
            The number of bins to create (default is 10).

        Returns:
        -------
        pandas.DataFrame
            The DataFrame with the binned feature.
        """
        _, bin_edges = pd.qcut(data['feature'], q=bins, retbins=True, duplicates='drop')
        labels = [f'({bin_edges[i]}, {bin_edges[i + 1]}]' for i in range(len(bin_edges) - 1)]
        data['feature'] = pd.cut(data['feature'], bins=bin_edges, labels=labels, include_lowest=True)
        return data

    def _aggregate_actual_expected(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates actual and predicted values by feature.

        Parameters:
        ----------
        data : pandas.DataFrame
            The DataFrame containing actual, predicted, and feature values.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame with aggregated actual, predicted, and exposure values by feature.
        """
        return data.groupby('feature', observed=True).agg(
            actual=('actual', 'mean'),
            predicted=('predicted', 'mean'),
            exposure=('actual', 'size'),
        ).reset_index()

    def _aggregate_feature_plot_data(self, actual: pd.Series, predicted: pd.Series, feature: pd.Series) -> pd.DataFrame:
        """ Aggregates actual, expected and comparison columns by specified feature.
            For numeric continuous features, creates bins.

        Args:
            feature (str): Feature to plot.

        Returns:
            Dataframe: Aggregated data by feature.
        """
        plot_data = self._get_feature_plot_data(actual, predicted, feature)
        if is_numeric_dtype(feature) & (len(np.unique(feature)) > 50):
            plot_data = self._group_feature(plot_data)

        return self._aggregate_actual_expected(plot_data)
    
    def plot_feature_ave(self, actual: pd.Series, predicted: pd.Series, feature: pd.Series, feature_name: Optional[str] = None) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """ Plots Actual v Predicted for feature.

        Args:
            feature (Str): Feature to plot.
        """
        
        feature_plot_data = self._aggregate_feature_plot_data(actual, predicted, feature)

        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax2 = ax1.twinx()

        ax1.bar(feature_plot_data['feature'],feature_plot_data['exposure'], alpha = 0.5)
        ax2.plot(feature_plot_data['feature'], feature_plot_data['actual'], label = "Actual", color = "r")
        ax2.plot(feature_plot_data['feature'], feature_plot_data['predicted'], label = "Predicted", color = "green")

        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)

        ax1.set_ylabel('Exposure', fontsize=14)
        ax2.set_ylabel('Actuals', fontsize=14)

        ax2.legend()
        if feature_name is not None:
            ax1.set_xlabel(feature_name)
            fig.suptitle(f"Actual v Predicted: {feature_name}", fontsize=20)
        return fig, (ax1, ax2)
        