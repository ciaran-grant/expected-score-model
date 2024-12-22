from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    @abstractmethod
    def feature_engineering(self, X):
        pass
    
    @abstractmethod
    def filter_data(self, X):
        pass
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def transform(self, X):
        pass