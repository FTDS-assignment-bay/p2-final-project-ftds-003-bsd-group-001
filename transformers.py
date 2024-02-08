from sklearn.base import BaseEstimator, TransformerMixin

class UnitPriceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['unit_price'] = X['sales'] / X['quantity']
        return X
# Include other custom transformers as needed
