
"""
Task 3: Feature Engineering Pipeline ONLY
Creates customer-level features from raw transaction data
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

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

def load_raw_data(filepath='data/raw/training.csv'):
    """Load raw transaction data"""
    df = pd.read_csv(filepath)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    print(f"Loaded {len(df):,} transactions")
    print(f"Unique customers: {df['CustomerId'].nunique():,}")
    return df

def create_feature_pipeline():
    """Create feature engineering pipeline (Task 3 only)"""
    return Pipeline([
        ('feature_engineer', RFMFeatureEngineer())
    ])

if __name__ == "__main__":
    # Load raw data
    df = load_raw_data()
    
    # Create and run pipeline
    pipeline = create_feature_pipeline()
    print("\n" + "="*50)
    print("Running Task 3: Feature Engineering Pipeline...")
    print("="*50)
    
    customer_features = pipeline.fit_transform(df)
    
    # Save processed data (without target)
    import os
    os.makedirs('data/processed', exist_ok=True)
    customer_features.to_csv('data/processed/customer_features.csv', index=False)
    
    print("\n" + "="*50)
    print("✅ Task 3 Complete!")
    print(f"✅ Data saved to data/processed/customer_features.csv")
    print(f"✅ Shape: {customer_features.shape}")
    print(f"✅ Columns: {customer_features.columns.tolist()}")
