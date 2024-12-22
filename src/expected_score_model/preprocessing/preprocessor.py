import numpy as np

class ExpectedScorePreprocessor:
    """
    A class used to preprocess data for an Expected Score Model.

    Methods
    -------
    feature_engineering(X)
        Adds new features to the dataset based on existing columns.
    
    filter_data(X, y=None)
        Filters the dataset to include only rows where 'Shot_At_Goal' is not null.
    
    fit(X, y=None)
        Fits the preprocessor to the data by performing feature engineering and filtering.
    
    transform(X, y=None)
        Transforms the data by applying feature engineering and filtering, and returns the new features.
    
    fit_transform(X, y)
        Fits the preprocessor to the data and then transforms it.
    """

    def feature_engineering(self, X):
        X_copy = X.copy(deep=True)
        
        X_copy['initial_state'] = X_copy['Initial_State']
        
        for col in ['Description', 'x', 'y', 'Period_Duration']:
            for i in range(4):
                X_copy[f'{col.lower()}_{i}'] = X_copy[col].shift(i) if i > 0 else X_copy[col]

        X_copy['time_since_last_action'] = X_copy['period_duration_0'] - X_copy['period_duration_1']
        X_copy['distance_since_last_action'] = ((X_copy['x_1'] - X_copy['x_0'])**2 + (X_copy['y_1'] - X_copy['y_0'])**2)**0.5
        
        X_copy['chain_duration'] = X_copy['Period_Duration'] - X_copy['Period_Duration_Chain_Start']
        
        X_copy['distance_to_goal_x'] = X_copy['Venue_Length']/2 - X_copy['x_0']
        X_copy['distance_to_middle_y'] = abs(X_copy['y_0'])
        X_copy['distance'] = (X_copy['distance_to_goal_x']**2 + X_copy['distance_to_middle_y']**2)**0.5
        X_copy['angle'] = np.arctan2(X_copy['distance_to_middle_y'], X_copy['distance_to_goal_x'])
        X_copy['angle_degrees'] = np.degrees(X_copy['angle'])

        X_copy['visible_goal_angle'] = (6.4*X_copy['distance_to_goal_x']) / (X_copy['distance_to_goal_x']**2 + X_copy['distance_to_middle_y']**2-(6.4/2)**2)
        X_copy['visible_goal_angle_degrees'] = np.degrees(X_copy['visible_goal_angle'])

        X_copy['visible_behind_angle'] = ((6.4*3)*X_copy['distance_to_goal_x']) / (X_copy['distance_to_goal_x']**2 + X_copy['distance_to_middle_y']**2-((3*6.4)/2)**2)
        X_copy['visible_behind_angle_degrees'] = np.degrees(X_copy['visible_behind_angle'])
        
        X_copy['distance_squared'] = X_copy['distance']**2
        X_copy['distance_log'] = np.log(X_copy['distance'])
        
        X_copy['ground_kick'] = X_copy['Description'].astype(str).apply(lambda x: 1 if 'Ground Kick' in x else 0)
        
        X_copy['set_shot'] = X_copy['description_1'].apply(lambda x: 1 if x is not None and isinstance(x, str) and ("Mark" in x or "Free" in x) else 0)
                
        return X_copy
    
    def filter_data(self, X, y=None):
        """
        Filters the dataset to include only rows where 'Shot_At_Goal' is not null.

        Args:
            X (DataFrame): Chain data.
            y (Series, optional): Shot result ['miss', 'behind', 'goal'].

        Returns:
            DataFrame or tuple: The filtered data, and optionally the filtered target variable.
        """
        mask = X['Shot_At_Goal'].notna()
        return X[mask] if y is None else (X[mask], y[mask])

    def fit(self, X, y=None):
        """
        Fits the preprocessor to the data by performing feature engineering and filtering.

        Args:
            X (DataFrame): Chain data.
            y (Series, optional): Shot result ['miss', 'behind', 'goal'].
        """
        X_copy = self.feature_engineering(X)
        X_copy = self.filter_data(X_copy)

        self.new_features = sorted(set(X_copy.columns) - set(X.columns))
                
    def transform(self, X, y=None):
        """
        Transforms the Chain data by applying feature engineering and filtering, and returns the new features.

        Args:
            X (DataFrame): Chain data.
            y (Series, optional): Shot result ['miss', 'behind', 'goal'].

        Returns:
            DataFrame or tuple: The transformed data, and optionally the transformed target variable.
        """
        X_transformed = self.feature_engineering(X)
        if y is None:
            X_transformed = self.filter_data(X_transformed)
            return X_transformed[self.new_features]
        else:
            X_transformed, y_transformed = self.filter_data(X_transformed, y)
            
            assert len(X_transformed) == len(y_transformed), "X and y must have the same length."
            
            return X_transformed[self.new_features], y_transformed

    def fit_transform(self, X, y=None):
        """
        Fits the preprocessor to the data and then transforms it.

        Args:
            X (DataFrame): Chain data.
            y (Series, optional): Shot result ['miss', 'behind', 'goal'].

        Returns:
            tuple: The transformed data and target variable.
        """
        self.fit(X, y)
        return self.transform(X, y)