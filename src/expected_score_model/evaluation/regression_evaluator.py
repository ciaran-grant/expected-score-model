from .model_evaluator import ModelEvaluator
import pandas as pd
import seaborn as sns
from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

class RegressionModelEvaluator(ModelEvaluator):
    
    @staticmethod
    def plot_ave(actual: pd.Series, predictions: pd.Series) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the actual vs predicted values for a regression model.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predictions : pd.Series
            The predicted values.

        Returns:
        -------
        Tuple[plt.Figure, plt.Axes]
            The figure and axes of the plot.
        """
        fig, ax = plt.subplots()
        ax.scatter(actual, predictions, edgecolors=(0, 0, 0))
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=4)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted')
        return fig, ax
        
    @staticmethod
    def plot_distribution(actual: pd.Series, predictions: pd.Series) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the distribution of actual vs. predicted values for a regression model.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predictions : pd.Series
            The predicted values.

        Returns:
        -------
        Tuple[plt.Figure, plt.Axes]
            The figure and axes of the plot.
        """
                
        fig, ax = plt.subplots()
        
        sns.kdeplot(actual, fill=True, color="r", ax=ax)
        sns.kdeplot(predictions, fill=True, color="b", ax=ax)
        ax.legend(labels=["Actual", "Expected"])
        
        return fig, ax
    
    @staticmethod
    def plot_decile(actual: pd.Series, predictions: pd.Series) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the actual vs. predicted values by decile.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predictions : pd.Series
            The predicted values.

        Returns:
        -------
        Tuple[plt.Figure, plt.Axes]
            The figure and axes of the plot.
        """
        plot_data = pd.DataFrame({
            'actual': actual,
            'predictions': predictions
        })
        
        # Create a new column 'predictions_decile' with decile values
        plot_data['predictions_decile'] = pd.qcut(plot_data['predictions'], 10, labels=False)

        # Group the dataframe by 'predictions_decile' and calculate the average of 'actual' and 'predictions'
        decile_avg = plot_data.groupby('predictions_decile').agg({'actual': 'mean', 'predictions': 'mean'})

        # Plot the average actual and prediction
        fig, ax = plt.subplots()
        decile_avg['actual'].plot(ax=ax, color='r')
        decile_avg['predictions'].plot(ax=ax, color='b')
        
        ax.set_ylim(0, 6)
        ax.set_xlabel('Decile')
        ax.set_ylabel('Average')
        ax.legend(['Actual', 'Prediction'])
        
        return fig, ax
        
    
    @staticmethod
    def get_mae(actual: pd.Series, predictions: pd.Series) -> float:
        """
        Returns the mean absolute error (MAE) for regression.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predictions : pd.Series
            The predicted values.

        Returns:
        -------
        float
            The mean absolute error between the actual and predicted values.
        """
        return mean_absolute_error(actual, predictions)
    
    @staticmethod
    def get_mse(actual: pd.Series, predictions: pd.Series) -> float:
        """
        Returns the mean squared error (MSE) for regression.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predictions : pd.Series
            The predicted values.

        Returns:
        -------
        float
            The mean squared error between the actual and predicted values.
        """
        return mean_squared_error(actual, predictions)

    @staticmethod
    def get_r2_score(actual: pd.Series, predictions: pd.Series) -> float:
        """
        Returns the R-squared score for regression.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predictions : pd.Series
            The predicted values.

        Returns:
        -------
        float
            The R-squared score between the actual and predicted values.
        """
        return r2_score(actual, predictions)

    