from expected_score_model.domain.contracts.modelling_data_contract import ModellingDataContract
from expected_score_model.domain.preprocessing.preprocessing import expected_score_feature_engineering
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """ Preprocessing class and functions for training total game score model.
    """
    
    def __init__(self, set_shot, model_response):
        """ Specify mappings and rolling average columns to create.

        Args:
            Mappings (Mappings): Mappings object specifying mapping and transformations.
            rolling_dict (Dict): Dictionary specifying columns and types of rolling average columns.
        """
        self.ModellingDataContract = ModellingDataContract
        self.set_shot = set_shot
        self.model_response = model_response
       
        
    def fit(self):
        """ Fits preprocessor to training data.
            Learns expected columns and mean imputations. 

        Args:
            X (Dataframe): Training dataframe to fit preprocessor to.

        Returns:
            self: Preprocessor learns expected colunms and means to impute.
        """
        
        # Keep only modelling columns
        if self.set_shot:
            if self.model_response == "Goal":
                self.modelling_cols = ModellingDataContract.feature_list_set_goal
            if self.model_response == "Behind":
                self.modelling_cols = ModellingDataContract.feature_list_set_behind
            if self.model_response == "Miss":
                self.modelling_cols = ModellingDataContract.feature_list_set_miss
        else:
            if self.model_response == "Goal":
                self.modelling_cols = ModellingDataContract.feature_list_open_goal
            if self.model_response == "Behind":
                self.modelling_cols = ModellingDataContract.feature_list_open_behind
            if self.model_response == "Miss":
                self.modelling_cols = ModellingDataContract.feature_list_open_miss
        return self
    
    def transform(self, X):
        """ Applies transformations and preprocessing steps to dataframe.

        Args:
            X (Dataframe): Training or unseen data to transform.

        Returns:
            Dataframe: Transformed data with modelling columns and no missing values.
        """
        
        # Feature Engineering
        X[['ballUp', 'centreBounce', 'kickIn', 'possGain', 'throwIn']] = pd.get_dummies(X['Initial_State'])
        X = expected_score_feature_engineering(X)
        
        # Shots
        X_shots = X[X['Shot_At_Goal'] == True]
        
        # Set or Open
        X_shots['Set_Shot'] = X_shots['Event_Type1'].apply(lambda x: ("Mark" in x) or ("Free" in x))
        X_shots = X_shots[X_shots['Set_Shot']==self.set_shot]

        # Modelling Features
        X_shots = X_shots[self.modelling_cols]
        
        return X_shots
    