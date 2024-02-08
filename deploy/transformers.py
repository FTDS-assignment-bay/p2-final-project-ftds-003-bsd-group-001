from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import pandas as pd

class UnitPriceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['unit_price'] = X['sales'] / X['quantity']
        return X

class KMeansAndLabelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    def fit(self, X, y=None):
        # Fit the KMeans model on the 'unit_price', ensuring it's reshaped for a single feature
        self.kmeans.fit(X[['unit_price']])
        return self
    
    def transform(self, X):
        # Predict the cluster labels
        cluster_labels = self.kmeans.predict(X[['unit_price']])
        
        # Convert cluster labels to strings for concatenation
        # Create a new DataFrame column for 'distinct_cluster_label'
        # Here, we use the apply function with a lambda to concatenate the string representations safely
        X = X.copy()  # Avoid SettingWithCopyWarning
        X['cluster_labels_str'] = cluster_labels.astype(str)
        X['distinct_cluster_label'] = X.apply(lambda row: row['cluster_labels_str'] + "_" + str(row['sub_category']), axis=1)
        
        # Now that 'distinct_cluster_label' is created, 'cluster_labels_str' can be dropped
        X.drop(['cluster_labels_str'], axis=1, inplace=True)
        
        return X

class DynamicOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoder.fit(X[['distinct_cluster_label']])
        return self
    
    def transform(self, X):
        encoded_features = self.encoder.transform(X[['distinct_cluster_label']]).toarray()
        encoded_df = pd.DataFrame(encoded_features, columns=self.encoder.get_feature_names_out(['distinct_cluster_label']))
        X.reset_index(drop=True, inplace=True)
        result = pd.concat([X, encoded_df], axis=1)
        result.drop(['distinct_cluster_label', 'sub_category', 'unit_price'], axis=1, inplace=True)  # Drop original columns if not needed
        return result
