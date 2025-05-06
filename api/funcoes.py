import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class minMaxQuantile(BaseEstimator, TransformerMixin):

    """ 
        Class to create a transformer that will return 
        the min and max quantile of a given dataset.
    """

    def __init__(self, min_quantile, max_quantile):
        """ 
            Class constructor that defines the min and
            max quantile values.
        """
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile
        self.min_values = None
        self.max_values = None
        self.feature_names = None
    
    def fit(self, X, y=None):
        """ 
            Calculate and store the min and max quantile values
            for each feature in the dataset or array X.
        """

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X = X.to_numpy()

        self.min_values = np.quantile(X, self.min_quantile, axis=0)
        self.max_values = np.quantile(X, self.max_quantile, axis=0)

        return self
    
    def transform(self, X, y=None):
        """ 
            Trasform the dataset or array X into a new dataset
            based on the min and max quantile values learned on the
            fit function.
            
        """
        if self.min_values is None or self.max_values is None:
            raise ValueError('Transformer not fitted yet.')
        
        is_dataframe = isinstance(X, pd.DataFrame)

        if is_dataframe:
            X = X.to_numpy()

        X_transformed = np.clip(X, self.min_values, self.max_values)

        if is_dataframe:
            return pd.DataFrame(X_transformed, columns=self.get_feature_names_out(self.feature_names))
        
        return X_transformed

    def get_feature_names_out(self, input_features):
        """ 
            Return the name of the features after the transformation.
        """
        return [f"{name}_clipped"for name in input_features]
    
class SimpleLog1pTransformer(BaseEstimator, TransformerMixin):
    """Aplica np.log1p (log(1+x)) a uma lista de colunas especificadas."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        
        return self

    def transform(self, X, y=None):
        """Aplica log1p às colunas especificadas."""

        X_transformed = X.copy()
        for col in self.columns:
            X_transformed[col] = np.where(
            X_transformed[col] > 0,
            np.log10(X_transformed[col]),
            np.log10(X_transformed[col] + 1) 
            )

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Retorna os nomes das features."""

        if input_features is None:
           
            if hasattr(self, 'columns'):
                 
                 return self.columns
            else:
                 raise ValueError("Input feature names are required.")
        return input_features