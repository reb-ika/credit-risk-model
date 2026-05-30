
"""
Task 5: Model Training with MLflow Tracking
Trains multiple models, logs experiments, and registers best model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_processed_data():
    """Load the processed customer data with target variable from Task 4"""
    df = pd.read_csv('data/processed/customer_features_with_target.csv')
    print(f"✅ Loaded data shape: {df.shape}")
    print(f"✅ Columns: {df.columns.tolist()}")
    return df

def prepare_features_target(df):
    """Separate features and target, exclude identifier columns"""
    # Exclude CustomerId, Cluster (used only for target creation), and target
    exclude_cols = ['CustomerId', 'Cluster', 'is_high_risk']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['is_high_risk']
    
    print(f"\n📊 Features ({len(feature_cols)}): {feature_cols}")
    print(f"🎯 Target distribution:\n{y.value_counts()}")
    print(f"🎯 Target percentage high-risk: {y.mean():.2%}")
    
    return X, y, feature_cols

def train_and_log_models(X_train, X_test, y_train, y_test, feature_names):
    """Train multiple models and log to MLflow"""
    
    # Set MLflow tracking URI to local directory
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Define models and their hyperparameter grids
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', use_label_encoder=False),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        }
    }
    
    best_model = None
    best_auc = 0
    best_model_name = None
    best_params_dict = None
    
    for model_name, config in models.items():
        print(f"\n{'='*60}")
        print(f"🏋️ Training {model_name}...")
        print('='*60)
        
        # Start MLflow run
        with mlflow.start_run(run_name=model_name):
            # Grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Make predictions
            y_pred = best_estimator.predict(X_test)
            y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            
            # Log feature importance for tree-based models
            if hasattr(best_estimator, 'feature_importances_'):
                for i, feature in enumerate(feature_names):
                    mlflow.log_metric(f"importance_{feature}", float(best_estimator.feature_importances_[i]))
            
            # Log the model
            signature = infer_signature(X_train, y_pred_proba)
            mlflow.sklearn.log_model(best_estimator, model_name, signature=signature)
            
            # Print results
            print(f"\n✅ {model_name} Results:")
            print(f"   Best parameters: {best_params}")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1 Score: {metrics['f1_score']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
            
            # Track best model
            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                best_model = best_estimator
                best_model_name = model_name
                best_params_dict = best_params
    
    return best_model, best_model_name, best_auc, best_params_dict

def main():
    print("🚀 Starting Model Training Pipeline (Task 5)")
    print("="*60)
    
    # Load data
    df = load_processed_data()
    
    # Prepare features and target
    X, y, feature_names = prepare_features_target(df)
    
    # Train-test split (80-20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Training set size: {len(X_train)} ({len(X_train)/len(X):.1%})")
    print(f"   Test set size: {len(X_test)} ({len(X_test)/len(X):.1%})")
    
    # Train and log models
    best_model, best_name, best_auc, best_params = train_and_log_models(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_name}")
    print(f"🏆 Best ROC-AUC: {best_auc:.4f}")
    print(f"🏆 Best Parameters: {best_params}")
    print(f"{'='*60}")
    
    print("\n✨ Task 5 Complete!")
    print("📊 To view MLflow UI, run: mlflow ui")
    print("🌐 Then open http://localhost:5000 in your browser")
    
    return best_model, feature_names

if __name__ == "__main__":
    best_model, feature_names = main()
