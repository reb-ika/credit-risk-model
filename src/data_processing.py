
"""
Task 4: Feature Engineering + Proxy Target Pipeline
Creates customer-level features AND adds is_high_risk proxy target
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class RFMFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for RFM feature engineering"""
    
    def __init__(self):
        self.snapshot_date = None
        
    def fit(self, X, y=None):
        self.snapshot_date = X['TransactionStartTime'].max()
        return self
    
    def transform(self, X):
        data = X.copy()
        
        # Aggregate customer-level features
        customer_features = data.groupby('CustomerId').agg(
            Total_Amount=('Amount', 'sum'),
            Avg_Amount=('Amount', 'mean'),
            Std_Amount=('Amount', 'std'),
            Max_Amount=('Amount', 'max'),
            Min_Amount=('Amount', 'min'),
            Transaction_Count=('TransactionId', 'count'),
            Unique_Products=('ProductId', 'nunique'),
            Unique_Categories=('ProductCategory', 'nunique'),
            Fraud_Rate=('FraudResult', 'mean'),
            Avg_PricingStrategy=('PricingStrategy', 'mean'),
            Last_Transaction=('TransactionStartTime', 'max')
        ).reset_index()
        
        # Recency (days since last transaction)
        customer_features['Recency_Days'] = (self.snapshot_date - customer_features['Last_Transaction']).dt.days
        
        # Time-based features
        customer_features['Avg_Transaction_Hour'] = data.groupby('CustomerId')['TransactionStartTime'].apply(
            lambda x: x.dt.hour.mean() if len(x) > 0 else 12
        ).values
        
        # Handle std deviation (fill NaN for single transactions)
        customer_features['Std_Amount'] = customer_features['Std_Amount'].fillna(0)
        
        # Drop the Last_Transaction column
        customer_features = customer_features.drop(columns=['Last_Transaction'])
        
        return customer_features

class ProxyTargetCreator(BaseEstimator, TransformerMixin):
    """Add proxy target variable using KMeans clustering"""
    
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.high_risk_cluster = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    
    def fit(self, X, y=None):
        # X is customer_features dataframe
        features_for_clustering = ['Recency_Days', 'Transaction_Count', 'Total_Amount']
        X_cluster = X[features_for_clustering].copy()
        
        # Scale and cluster
        X_scaled = self.scaler.fit_transform(X_cluster)
        self.kmeans.fit(X_scaled)
        
        # Identify high-risk cluster (highest recency + lowest frequency)
        cluster_summary = X.groupby(self.kmeans.labels_).agg({
            'Recency_Days': 'mean',
            'Transaction_Count': 'mean',
            'Total_Amount': 'mean'
        })
        
        cluster_summary['Risk_Score'] = cluster_summary['Recency_Days'] - cluster_summary['Transaction_Count']
        self.high_risk_cluster = cluster_summary['Risk_Score'].idxmax()
        
        print("\nCluster Summary:")
        print(cluster_summary.round(2))
        print(f"\nHigh-risk cluster identified: {self.high_risk_cluster}")
        
        return self
    
    def transform(self, X):
        data = X.copy()
        
        # Get cluster assignments
        features_for_clustering = ['Recency_Days', 'Transaction_Count', 'Total_Amount']
        X_cluster = data[features_for_clustering].copy()
        X_scaled = self.scaler.transform(X_cluster)
        data['Cluster'] = self.kmeans.predict(X_scaled)
        
        # Create proxy target
        data['is_high_risk'] = (data['Cluster'] == self.high_risk_cluster).astype(int)
        
        print(f"\nHigh-risk customers: {data['is_high_risk'].sum()} ({data['is_high_risk'].mean():.2%})")
        
        return data

def load_raw_data(filepath='data/raw/training.csv'):
    """Load raw transaction data"""
    df = pd.read_csv(filepath)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    print(f"Loaded {len(df):,} transactions")
    print(f"Unique customers: {df['CustomerId'].nunique():,}")
    return df

def create_complete_pipeline():
    """Create complete pipeline with feature engineering + proxy target"""
    return Pipeline([
        ('feature_engineer', RFMFeatureEngineer()),
        ('proxy_target', ProxyTargetCreator())
    ])

if __name__ == "__main__":
    # Load raw data
    df = load_raw_data()
    
    # Create and run pipeline
    pipeline = create_complete_pipeline()
    print("\n" + "="*50)
    print("Running Task 4: Feature Engineering + Proxy Target Pipeline...")
    print("="*50)
    
    processed_data = pipeline.fit_transform(df)
    
    # Save processed data with target
    import os
    os.makedirs('data/processed', exist_ok=True)
    processed_data.to_csv('data/processed/customer_features_with_target.csv', index=False)
    
    print("\n" + "="*50)
    print("✅ Task 4 Complete!")
    print(f"✅ Data saved to data/processed/customer_features_with_target.csv")
    print(f"✅ Shape: {processed_data.shape}")
    print(f"✅ Target distribution:")
    print(processed_data['is_high_risk'].value_counts())
