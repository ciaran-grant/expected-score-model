from .model_evaluator import ModelEvaluator
from sklearn.metrics import (
    log_loss, 
    brier_score_loss, 
    confusion_matrix, 
    roc_curve, 
    roc_auc_score, 
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score
    )
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import pandas as pd
from typing import Union


class ClassificationModelEvaluator(ModelEvaluator):
    
    def __init__(self, model, classification_type='binary'):
        """
        Initializes the ClassificationModelEvaluator with a model and classification type.

        Parameters:
        ----------
        model : object
            The classification model to be evaluated.
        classification_type : str, optional
            The type of classification ('binary' or 'multi') (default is 'binary').

        Raises:
        ------
        ValueError
            If classification_type is not 'binary' or 'multi'.
        """
        super().__init__(model)
        if classification_type not in ['binary', 'multi']:
            raise ValueError("classification_type must be either 'binary' or 'multi'")
        else:
            self.classification_type = classification_type
        
    def get_log_loss(self, actual: pd.Series, predictions: pd.Series) -> Union[float, dict[str, float]]:
        """
        Calculates the log loss for classification.

        Parameters:
        ----------
        actual : array-like
            The actual values.
        predictions : array-like
            The predicted values.

        Returns:
        -------
        float or dict
            The log loss for binary classification or a dictionary of log losses for each class in multi-class classification.
        """
        if self.classification_type == 'binary':
            return log_loss(actual, predictions)
        if self.classification_type == 'multi':
            multi_class_log_loss = [log_loss(actual[:, i], predictions[:, i]) for i in range(len(self.model.classes_))]

            return {
                f'class_{c}': multi_class_log_loss[i]
                for i, c in enumerate(self.model.classes_)
            }

    def get_brier_score_loss(self, actual: pd.Series, predictions: pd.Series) -> Union[float, dict[str, float]]:
        """
        Calculates the Brier score loss for classification.

        Parameters:
        ----------
        actual : array-like
            The actual values.
        predictions : array-like
            The predicted values.

        Returns:
        -------
        float or dict
            The Brier score loss for binary classification or a dictionary of Brier score losses for each class in multi-class classification.
        """

        if self.classification_type == 'binary':
            return brier_score_loss(actual, predictions)
        if self.classification_type == 'multi':
            multi_class_brier_score = [brier_score_loss(actual[:, i], predictions[:, i]) for i in range(len(self.model.classes_))]

            return {
                f'class_{c}': multi_class_brier_score[i]
                for i, c in enumerate(self.model.classes_)
            }
            
    def display_calibration_curve(self, actual: pd.Series, predictions: pd.Series, nbins: int = 10) -> CalibrationDisplay:
        """
        Displays the calibration curve for the classification model.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predictions : pd.Series
            The predicted probabilities.

        Returns:
        -------
        CalibrationDisplay
            The calibration display object.
        """
        if self.classification_type == 'binary':
            return CalibrationDisplay.from_predictions(actual, predictions, n_bins=nbins)
        if self.classification_type == 'multi':
            return [CalibrationDisplay.from_predictions(actual[:, i], predictions[:, i], n_bins=nbins, name = self.model.classes_[i]) for i in range(len(self.model.classes_))]

    # Binary and Multi Classification Metrics
    @staticmethod
    def get_confusion_matrix(actual_label: pd.Series, predicted_label: pd.Series) -> ConfusionMatrixDisplay:
        """
        Generates and displays the confusion matrix for the classification model.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predicted_label : pd.Series
            The predicted values.

        Returns:
        -------
        ConfusionMatrixDisplay
            The confusion matrix display object.
        """
        return confusion_matrix(actual_label, predicted_label)
    
    @staticmethod
    def display_confusion_matrix(actual_label: pd.Series, predicted_label: pd.Series) -> ConfusionMatrixDisplay:
        """
        Generates and displays the confusion matrix for the classification model.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predicted_label : pd.Series
            The predicted values.

        Returns:
        -------
        ConfusionMatrixDisplay
            The confusion matrix display object.
        """
        return ConfusionMatrixDisplay.from_predictions(actual_label, predicted_label, cmap="Blues", normalize="all")
    
    # Binary Classification Metrics
    @staticmethod
    def get_roc_curve(actual_label: pd.Series, predicted_label: pd.Series) -> RocCurveDisplay:
        """
        Generates and displays the ROC curve for the classification model.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predicted_label : pd.Series
            The predicted probabilities.

        Returns:
        -------
        RocCurveDisplay
            The ROC curve display object.
        """
        fpr, tpr, thresholds = roc_curve(actual_label, predicted_label)
        return fpr, tpr, thresholds
    
    @staticmethod
    def get_auc_score(actual_label: pd.Series, predicted_label: pd.Series) -> float:
        """
        Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predicted_label : pd.Series
            The predicted probabilities.

        Returns:
        -------
        float
            The ROC AUC score.
        """
        return roc_auc_score(actual_label, predicted_label)
    
    @staticmethod
    def plot_roc_curve(actual_label: pd.Series, predicted_label: pd.Series) -> RocCurveDisplay:
        """
        Plots the ROC curve for the classification model.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predicted_label : pd.Series
            The predicted probabilities.

        Returns:
        -------
        RocCurveDisplay
            The ROC curve display object.
        """
        return RocCurveDisplay.from_predictions(actual_label, predicted_label)
     
    @staticmethod
    def plot_prauc_curve(actual_label: pd.Series, predicted_label: pd.Series) -> PrecisionRecallDisplay:
        """
        Plots the Precision-Recall AUC curve for the classification model.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predicted_label : pd.Series
            The predicted probabilities.

        Returns:
        -------
        PrecisionRecallDisplay
            The Precision-Recall AUC curve display object.
        """
        return PrecisionRecallDisplay.from_predictions(actual_label, predicted_label) 
           
    @staticmethod
    def get_accuracy(actual_label: pd.Series, predicted_label: pd.Series) -> float:
        """
        Calculates the accuracy score for the classification model.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predicted_label : pd.Series
            The predicted values.

        Returns:
        -------
        float
            The accuracy score.
        """
        return accuracy_score(actual_label, predicted_label)
    
    @staticmethod
    def get_recall(actual_label: pd.Series, predicted_label: pd.Series) -> float:
        """
        Calculates the recall score for the classification model.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predicted_label : pd.Series
            The predicted values.

        Returns:
        -------
        float
            The recall score.
        """
        return recall_score(actual_label, predicted_label)
    
    @staticmethod
    def get_precision(actual_label: pd.Series, predicted_label: pd.Series) -> float:
        """
        Calculates the precision score for the classification model.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predicted_label : pd.Series
            The predicted values.

        Returns:
        -------
        float
            The precision score.
        """
        return precision_score(actual_label, predicted_label)

    @staticmethod
    def get_f1_score(actual_label: pd.Series, predicted_label: pd.Series) -> float:
        """
        Calculates the F1 score for the classification model.

        Parameters:
        ----------
        actual : pd.Series
            The actual values.
        predicted_label : pd.Series
            The predicted values.

        Returns:
        -------
        float
            The F1 score.
        """
        return f1_score(actual_label, predicted_label, average="binary")
