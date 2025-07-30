from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
from sklearn.impute import SimpleImputer

__all__ = ["DataFrameSimpleImputer"]


class DataFrameSimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="mean"):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.columns = None

    def fit(self, X, y=None):
        self.columns = X.columns
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_imputed = self.imputer.transform(X)
        return pd.DataFrame(X_imputed, columns=self.columns, index=X.index)
