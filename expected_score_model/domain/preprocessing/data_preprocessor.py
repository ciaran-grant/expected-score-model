# from total_points_model.domain.preprocessing.preprocessing_functions import 
from expected_score_model.domain.contracts.modelling_data_contract import ModellingDataContract
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np

from expected_score_model.domain.contracts.mappings import Mappings

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """ Preprocessing class and functions for training total game score model.
    """
    
    def __init__(self, Mappings, rolling_dict):
        """ Specify mappings and rolling average columns to create.

        Args:
            Mappings (Mappings): Mappings object specifying mapping and transformations.
            rolling_dict (Dict): Dictionary specifying columns and types of rolling average columns.
        """
        self.ModellingDataContract = ModellingDataContract
       
        
    def fit(self, X):
        """ Fits preprocessor to training data.
            Learns expected columns and mean imputations. 

        Args:
            X (Dataframe): Training dataframe to fit preprocessor to.

        Returns:
            self: Preprocessor learns expected colunms and means to impute.
        """
        
        X_copy = X.copy()
        
        # Feature Engineering
        
        # Feature Grouping and Mappings
        
        # Apply self mappings
        
        self.modelling_cols = ModellingDataContract.modelling_feature_list + self.rolling_cols_list

        # Keep only modelling columns
        self.modelling_cols = ModellingDataContract.modelling_feature_list + self.rolling_cols_list
        X_copy = X_copy[self.modelling_cols]
        
        # Fitting to training data
        self.train_set_means = X_copy.mean()
        self.expected_dummy_cols = list(pd.get_dummies(X_copy))
                        
        return self
    
    def transform(self, X):
        """ Applies transformations and preprocessing steps to dataframe.

        Args:
            X (Dataframe): Training or unseen data to transform.

        Returns:
            Dataframe: Transformed data with modelling columns and no missing values.
        """
        
        # Feature Engineering
        
        # Feature Grouping and Mappings
        
        # Apply self mappings
        
        # Keep only modelling columns
        self.modelling_cols = ModellingDataContract.modelling_feature_list + self.rolling_cols_list
        X = X[[ModellingDataContract.ID_COL] + self.modelling_cols]
        
        # Applying transformations
        X = X.fillna(self.train_set_means)        
        X_dummies = pd.get_dummies(X[self.modelling_cols])
        for col in list(self.expected_dummy_cols):
            if col not in list(X_dummies):
                X_dummies[col] = 0

        return X_dummies[self.expected_dummy_cols]
    