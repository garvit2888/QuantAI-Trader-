import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("XGBoost not installed. Please `pip install xgboost`. Using RF as fallback.")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset_builder import prepare_dataset

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and print classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"[{model_name}] Accuracy: {acc:.2f} | Precision: {prec:.2f} | Recall: {rec:.2f} | F-1: {f1:.2f}")
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

def train_and_validate(X, y, model, n_splits=5):
    """
    Perform Walk-Forward Validation using TimeSeriesSplit.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    
    # We train incrementally without data leakage
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Fit on past
        model.fit(X_train, y_train)
        
        # Predict on future block
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        metrics_list.append(acc)
    
    avg_acc = np.mean(metrics_list)
    print(f"✅ Walk-Forward CV Avg Accuracy: {avg_acc:.2f} (Over {n_splits} splits)")
    
    # Finally, retrain on the complete dataset so the model is ready for live predictions
    model.fit(X, y)
    return model

def run_training_pipeline(ticker="RELIANCE.NS"):
    df = prepare_dataset(ticker, horizon=1)
    if df is None or len(df) < 30:
        raise ValueError("Dataset is too small for Machine Learning training. Please choose an older start date or ensure the company has enough public trading history (minimum ~60 days).")
        return None
        
    # Define features and target using only historical rows that HAVE a target
    train_df = df.dropna(subset=['Target_Return'])
    
    # Exclude non-feature columns
    exclude = ['Open', 'High', 'Low', 'Close', 'Target_Return', 'Target_Class', 'Risk_Level']
    features = [c for c in df.columns if c not in exclude]
    
    X_train = train_df[features].ffill().fillna(0) # Safety fill
    y_train = train_df['Target_Class']
    
    print("\n" + "="*50)
    print(f"🚀 TRAINING ENSEMBLE MODELS on {len(features)} Features")
    print("="*50)
    
    models = {
        "Baseline_LogReg": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    }
    
    if XGBClassifier is not None:
         models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
         
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        trained = train_and_validate(X_train, y_train, model, n_splits=5)
        trained_models[name] = trained
        
    # Feature Importance (Using RF or XGBoost)
    if "XGBoost" in trained_models:
        best_model = trained_models["XGBoost"]
        importances = best_model.feature_importances_
    else:
        best_model = trained_models["RandomForest"]
        importances = best_model.feature_importances_
        
    # Save the feature importance mapping
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    print("\n🌟 Top 10 Important Features:")
    print(feat_imp.head(10))
    
    return trained_models, df, feat_imp

if __name__ == "__main__":
    models, end_df, feat_imp = run_training_pipeline("RELIANCE.NS", "2020-01-01", "2024-01-01")
    
    # Save model using joblib (optional, Streamlit can retrain fast if data is small, but good practice)
    import joblib
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(models["RandomForest"], "models/rf_model.pkl")
    print("\n💾 Model saved to models/rf_model.pkl")
