
"""
Pydantic models for API request and response validation
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class CustomerFeatures(BaseModel):
    """Features required for credit risk prediction"""
    Total_Amount: float = Field(..., description="Total transaction amount")
    Avg_Amount: float = Field(..., description="Average transaction amount")
    Std_Amount: float = Field(..., description="Standard deviation of amounts")
    Max_Amount: float = Field(..., description="Maximum transaction amount")
    Min_Amount: float = Field(..., description="Minimum transaction amount")
    Transaction_Count: int = Field(..., description="Number of transactions")
    Unique_Products: int = Field(..., description="Number of unique products purchased")
    Unique_Categories: int = Field(..., description="Number of unique product categories")
    Fraud_Rate: float = Field(..., description="Proportion of flagged fraud transactions")
    Avg_PricingStrategy: float = Field(..., description="Average pricing strategy")
    Recency_Days: int = Field(..., description="Days since last transaction")
    Avg_Transaction_Hour: float = Field(..., description="Average hour of day for transactions")
    
    class Config:
        schema_extra = {
            "example": {
                "Total_Amount": 50000.0,
                "Avg_Amount": 2500.0,
                "Std_Amount": 1500.0,
                "Max_Amount": 10000.0,
                "Min_Amount": 100.0,
                "Transaction_Count": 20,
                "Unique_Products": 5,
                "Unique_Categories": 3,
                "Fraud_Rate": 0.0,
                "Avg_PricingStrategy": 2.5,
                "Recency_Days": 5,
                "Avg_Transaction_Hour": 14.5
            }
        }

class PredictionRequest(BaseModel):
    """Request body for prediction endpoint"""
    customers: List[CustomerFeatures] = Field(..., description="List of customers to score")
    
    class Config:
        schema_extra = {
            "example": {
                "customers": [
                    {
                        "Total_Amount": 50000.0,
                        "Avg_Amount": 2500.0,
                        "Std_Amount": 1500.0,
                        "Max_Amount": 10000.0,
                        "Min_Amount": 100.0,
                        "Transaction_Count": 20,
                        "Unique_Products": 5,
                        "Unique_Categories": 3,
                        "Fraud_Rate": 0.0,
                        "Avg_PricingStrategy": 2.5,
                        "Recency_Days": 5,
                        "Avg_Transaction_Hour": 14.5
                    }
                ]
            }
        }

class PredictionResponse(BaseModel):
    """Response body for prediction endpoint"""
    predictions: List[Dict[str, Any]] = Field(..., description="Risk predictions for each customer")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "risk_probability": 0.05,
                        "risk_class": "Low Risk",
                        "credit_score": 750
                    }
                ]
            }
        }

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
