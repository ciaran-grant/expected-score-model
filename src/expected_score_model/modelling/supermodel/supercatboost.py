from catboost import CatBoostClassifier
import joblib

class SuperCatBoostClassifier:
    def __init__(self, params):
        """ CatBoost Classifier model wrapper.

        Args:
            params (Dict): CatBoost model parameters
        """
        self.params = params
        self.cb_params = self._get_hyperparameters()
    
    def _get_hyperparameters(self):
        """ From given parameters dictionary, gets requried hyperparameters for CatBoost.

        Returns:
            Dict: CatBoost model hyperparameters
        """
        
        return {
            'depth': self.params['depth'],
            'min_child_samples': self.params['min_child_samples'],
            'eta': self.params['eta'],
            'reg_lambda': self.params['reg_lambda'],
            'colsample_bylevel': self.params['colsample_bylevel'],
        }
    
    def fit(self, X, y):
                
        cat_features = X.select_dtypes(include=['object']).columns.tolist()
                
        self.cb_clf = CatBoostClassifier(
                iterations=self.params['iterations'],
                depth = self.params['depth'],
                min_child_samples = self.params['min_child_samples'],
                eta = self.params['eta'],
                reg_lambda = self.params['reg_lambda'],
                colsample_bylevel = self.params['colsample_bylevel'],
                cat_features=cat_features,
                loss_function=self.params['error'], 
                eval_metric=self.params['error'],
                verbose = self.params['verbosity']
        )

        self.cb_clf.fit(X, y)
        
        self.classes_ = self.cb_clf.classes_
    
    def predict(self, X):
        
        return self.cb_clf.predict(X)
    
    def predict_proba(self, X):
        return self.cb_clf.predict_proba(X)
    
    def export_model(self, file_path):
        
        joblib.dump(self, file_path)