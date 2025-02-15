import xgboost as xgb
import joblib
from betacal import BetaCalibration

class SuperXGBClassifier:
    def __init__(self, params):
        """ XGBoost Regression model that requries training data, test data and parameters.

        Args:
            X_train (Dataframe): Training dataframe including modelling features
            y_train (Array): Training set response
            X_test (Dataframe): Test dataframe including modelling features
            y_test (Array): Test set response
            params (Dict): XGBoost model parameters
        """
        self.params = params        
        self.xgb_params = self._get_hyperparameters()
    
    def _get_hyperparameters(self):
        """ From given parameters dictionary, gets requried hyperparameters for XGBoost.

        Returns:
            Dict: XGBoost model hyperparameters
        """
        
        return {
            'max_depth': self.params['max_depth'],
            'min_child_weight': self.params['min_child_weight'],
            'eta': self.params['eta'],
            'gamma': self.params['gamma'],
            'lambda': self.params['lambda'],
            'alpha': self.params['alpha'],
            'subsample': self.params['subsample'],
            'colsample_bytree': self.params['colsample_bytree'],
        }
    
    def fit(self, X, y, X_test = None, y_test = None):
                
        self.xgb_clf = xgb.XGBClassifier(n_estimators=self.params['num_rounds'],
                                         objective = self.params['objective'],
                                         verbosity = self.params['verbosity'],
                                         early_stopping_rounds = self.params['early_stopping_rounds'],
                                         learning_rate = self.params['eta'],
                                         max_depth = self.params['max_depth'],
                                         min_child_weight = self.params['min_child_weight'],
                                         gamma = self.params['gamma'],
                                         subsample = self.params['subsample'],
                                         colsample_bytree = self.params['colsample_bytree'],
                                         reg_alpha = self.params['alpha'],
                                         reg_lambda = self.params['lambda'],
                                         monotone_constraints = self.params['monotone_constraints']
                                         )
        
        self.xgb_clf.fit(X = X, y = y, eval_set = [(X, y), (X_test, y_test)])
    
    def calibrate(self):
        
        cal_probas = self.predict_proba(self.X_cal)[:, 1]
        self.xgb_cal = BetaCalibration(parameters = "abm")
        self.xgb_cal.fit(cal_probas.reshape(-1, 1), self.y_cal)

    def predict(self, X):
        
        return self.xgb_clf.predict(X)
    
    def predict_proba(self, X, calibrate = False):
        if calibrate:
            return self.xgb_cal.predict(self.xgb_clf.predict_proba(X)[:, 1].reshape(-1, 1))
        else:
            return self.xgb_clf.predict_proba(X)
    
    def export_model(self, file_path):
        
        joblib.dump(self, file_path)