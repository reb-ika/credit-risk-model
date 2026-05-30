"""
Unit tests for data processing functions
"""
import pytest
import pandas as pd
import numpy as np
from src.data_processing import load_raw_data, RFMFeatureEngineer, ProxyTargetCreator

def test_load_raw_data():
    """Test that raw data loads correctly"""
    try:
        df = load_raw_data('data/raw/training.csv')
        assert df is not None
        assert 'TransactionId' in df.columns
        assert 'CustomerId' in df.columns
        assert 'Amount' in df.columns
        assert len(df) > 0
        print("✅ test_load_raw_data passed")
    except FileNotFoundError:
        pytest.skip("Raw data file not found - skipping test")

def test_rfm_feature_engineer():
    """Test RFM feature engineering creates expected columns"""
    # Create sample data
    sample_data = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'TransactionId': ['T1', 'T2', 'T3'],
        'Amount': [100, 200, 300],
        'TransactionStartTime': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'ProductId': ['P1', 'P2', 'P1'],
        'ProductCategory': ['Cat1', 'Cat2', 'Cat1'],
        'FraudResult': [0, 0, 1],
        'PricingStrategy': [1, 2, 1]
    })
    
    engineer = RFMFeatureEngineer()
    engineer.fit(sample_data)
    result = engineer.transform(sample_data)
    
    # Check expected columns
    expected_cols = ['CustomerId', 'Total_Amount', 'Avg_Amount', 'Transaction_Count', 
                     'Unique_Products', 'Recency_Days']
    for col in expected_cols:
        assert col in result.columns
    
    # Check number of customers
    assert len(result) == 2
    print("✅ test_rfm_feature_engineer passed")

def test_proxy_target_creator():
    """Test proxy target creation returns binary column"""
    # Create sample customer features
    sample_features = pd.DataFrame({
        'CustomerId': [1, 2, 3, 4, 5],
        'Recency_Days': [5, 10, 60, 70, 90],
        'Transaction_Count': [50, 30, 10, 5, 2],
        'Total_Amount': [100000, 50000, 10000, 5000, 1000],
        'Avg_Amount': [2000, 1667, 1000, 1000, 500],
        'Std_Amount': [500, 400, 300, 200, 100]
    })
    
    creator = ProxyTargetCreator(n_clusters=3, random_state=42)
    creator.fit(sample_features)
    result = creator.transform(sample_features)
    
    # Check target column exists
    assert 'is_high_risk' in result.columns
    assert 'Cluster' in result.columns
    
    # Check binary values
    assert set(result['is_high_risk'].unique()).issubset({0, 1})
    print("✅ test_proxy_target_creator passed")

def test_feature_columns_consistency():
    """Test that feature engineering produces consistent columns"""
    sample_data = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'TransactionId': ['T1', 'T2', 'T3'],
        'Amount': [100, 200, 300],
        'TransactionStartTime': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'ProductId': ['P1', 'P2', 'P1'],
        'ProductCategory': ['Cat1', 'Cat2', 'Cat1'],
        'FraudResult': [0, 0, 1],
        'PricingStrategy': [1, 2, 1]
    })
    
    engineer = RFMFeatureEngineer()
    engineer.fit(sample_data)
    result = engineer.transform(sample_data)
    
    # These columns should always be present
    required_cols = ['Total_Amount', 'Avg_Amount', 'Transaction_Count', 'Recency_Days']
    for col in required_cols:
        assert col in result.columns
    
    # No null values in these columns
    for col in required_cols:
        assert not result[col].isnull().any()
    
    print("✅ test_feature_columns_consistency passed")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
