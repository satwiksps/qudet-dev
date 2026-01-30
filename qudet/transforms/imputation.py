"""
Quantum Imputation for handling missing data.

Real data has holes (NaNs). Classical imputation (mean/median) destroys 
structure. Quantum Imputation looks at the "nearest neighbors" in Hilbert 
space and borrows their values to fill the gaps.
"""

import numpy as np
import pandas as pd
from ..analytics.clustering import QuantumKMeans


class QuantumImputer:
    """
    Fills missing values (NaN) using Quantum Similarity.
    
    Logic:
    1. Identify rows with missing data.
    2. Use Quantum K-Means to find which cluster the row belongs to 
       (based on the columns that AREN'T missing).
    3. Fill missing values with the Centroid of that Quantum Cluster.
    
    This preserves the multi-variate structure of the data better than
    simple mean/median imputation.
    
    Attributes:
        n_clusters (int): Number of clusters for K-Means.
        clusterer: QuantumKMeans instance for clustering.
    """
    
    def __init__(self, n_clusters: int = 3):
        """
        Initialize QuantumImputer.
        
        Args:
            n_clusters: Number of clusters for K-Means. Default: 3
        """
        self.n_clusters = n_clusters
        self.clusterer = QuantumKMeans(n_clusters=n_clusters)

    def fit(self, X):
        """
        Fits the clusterer on COMPLETE data only.
        
        Args:
            X (pd.DataFrame or np.ndarray): Training data (should be mostly complete)
            
        Returns:
            self: Fitted QuantumImputer instance
        """
        if isinstance(X, pd.DataFrame):
            X_clean = X.dropna()
        else:
            X_clean = X[~np.isnan(X).any(axis=1)]
            
        self.clusterer.fit(X_clean)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Repair missing values in X using cluster centroids.
        
        Args:
            X (pd.DataFrame): Data with potential missing values
            
        Returns:
            pd.DataFrame: Data with NaNs filled from cluster centroids
        """
        X_repaired = X.copy()
        
        missing_rows = X_repaired.isnull().any(axis=1)
        
        if not missing_rows.any():
            return X_repaired
            
        for idx in X_repaired[missing_rows].index:
            row = X_repaired.loc[idx]
            
            temp_row = row.fillna(0)
            
            distances = [self.clusterer._quantum_distance(temp_row.values, c) 
                         for c in self.clusterer.centroids_]
            
            best_cluster_idx = np.argmin(distances)
            centroid = self.clusterer.centroids_[best_cluster_idx]
            
            for i, col in enumerate(X.columns):
                if pd.isna(row[col]):
                    X_repaired.at[idx, col] = centroid[i]
                    
        return X_repaired

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X (pd.DataFrame): Training data with missing values
            
        Returns:
            pd.DataFrame: Repaired data
        """
        return self.fit(X).transform(X)
