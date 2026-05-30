
"""
FastAPI application for credit risk scoring
"""
import os
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import mlflow.sklearn
from pathlib import Path

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

from api.pydantic_models import PredictionRequest, PredictionResponse, HealthResponse

# Initialize FastAPI app
app = FastAPI(
    title="Bati Bank Credit Risk Scoring API",
    description="Predicts credit risk probability for eCommerce customers using RFM-based model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and feature columns
model = None
feature_columns = None
mlflow.set_tracking_uri("file:./mlruns")

def load_best_model():
    """Load the best model from MLflow registry"""
    global model, feature_columns
    
    try:
        # Get the latest version of CreditRiskModel from registry
        client = mlflow.tracking.MlflowClient()
        
        # Search for the best model run
        experiment = mlflow.get_experiment_by_name("Default")
        if experiment is None:
            raise ValueError("No experiment found")
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        # Find RandomForest run with highest ROC-AUC
        rf_runs = runs[runs['tags.mlflow.runName'] == 'RandomForest']
        if len(rf_runs) == 0:
            raise ValueError("No RandomForest run found")
        
        best_run = rf_runs.iloc[0]
        model_uri = f"runs:/{best_run.run_id}/RandomForest"
        
        # Load model
        model = mlflow.sklearn.load_model(model_uri)
        
        # Define feature columns (from training)
        feature_columns = [
            'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Max_Amount', 'Min_Amount',
            'Transaction_Count', 'Unique_Products', 'Unique_Categories', 'Fraud_Rate',
            'Avg_PricingStrategy', 'Recency_Days', 'Avg_Transaction_Hour'
        ]
        
        print(f"✅ Model loaded successfully from {model_uri}")
        print(f"✅ Feature columns: {feature_columns}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def calculate_credit_score(risk_probability):
    """Convert risk probability to credit score (300-850 range)"""
    # Higher risk probability = lower credit score
    # Map 0% risk -> 850, 100% risk -> 300
    credit_score = int(850 - (risk_probability * 550))
    return max(300, min(850, credit_score))

def get_risk_class(risk_probability):
    """Categorize risk probability into classes"""
    if risk_probability < 0.2:
        return "Low Risk"
    elif risk_probability < 0.4:
        return "Medium Risk"
    elif risk_probability < 0.6:
        return "High Risk"
    else:
        return "Very High Risk"

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    success = load_best_model()
    if not success:
        print("⚠️ Warning: Model not loaded. API will still run but predictions will fail.")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version="RandomForest_v1"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict credit risk for one or more customers
    
    Returns risk probability, risk class, and credit score for each customer
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        customers_data = [customer.dict() for customer in request.customers]
        df = pd.DataFrame(customers_data)
        
        # Ensure feature order matches training
        df = df[feature_columns]
        
        # Make predictions
        risk_probabilities = model.predict_proba(df)[:, 1]
        
        # Prepare response
        predictions = []
        for i, prob in enumerate(risk_probabilities):
            predictions.append({
                "risk_probability": round(float(prob), 4),
                "risk_class": get_risk_class(prob),
                "credit_score": calculate_credit_score(prob)
            })
        
        return PredictionResponse(predictions=predictions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/feature_importance")
async def get_feature_importance():
    """Return feature importance from the model"""
    if model is None or not hasattr(model, 'feature_importances_'):
        raise HTTPException(status_code=503, detail="Feature importance not available")
    
    importance_dict = dict(zip(feature_columns, model.feature_importances_.tolist()))
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return {"feature_importance": sorted_importance}
