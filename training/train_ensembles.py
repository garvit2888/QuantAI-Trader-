import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset_builder import prepare_dataset

def train_and_validate(X, y, model, n_splits=5):
    """
    Perform Walk-Forward Validation using TimeSeriesSplit.
    Returns:
        - model: The model trained on all data.
        - oos_predictions: A pd.Series of continuous probabilities (Class 1) made on independent test blocks.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    oos_preds_prob = []
    oos_indices = []
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Check if we have at least 2 classes to train
        if len(np.unique(y_train)) < 2:
            # Cannot train perfectly on 1 class. Dummy proba
            pred_prob = np.full(len(X_test), float(y_train.iloc[0]))
            pred_bin = np.full(len(X_test), y_train.iloc[0])
        else:
            try:
                # Some CV splits in CalibratedClassifier might still fail if internal folds lack 2 classes
                model.fit(X_train, y_train)
                pred_prob = model.predict_proba(X_test)[:, 1]
                pred_bin = model.predict(X_test)
            except ValueError:
                # Fallback to uncalibrated base estimator securely if Platt scaling fails due to size
                if isinstance(model, CalibratedClassifierCV):
                    fallback = model.estimator
                    fallback.fit(X_train, y_train)
                    pred_prob = fallback.predict_proba(X_test)[:, 1]
                    pred_bin = fallback.predict(X_test)
                else:
                    raise
            
        oos_preds_prob.extend(pred_prob)
        oos_indices.extend(X_test.index)
        
        if len(np.unique(y_test)) >= 2:
            acc = accuracy_score(y_test, pred_bin)
            metrics_list.append(acc)
    
    avg_acc = np.mean(metrics_list) if metrics_list else 0.5
    print(f"✅ Walk-Forward CV Avg Accuracy (Out-of-Sample): {avg_acc:.2f} (Over {len(metrics_list)} valid splits)")
    
    # Final model is trained on all data for future production predictions
    if len(np.unique(y)) >= 2:
        try:
            model.fit(X, y)
        except ValueError:
            if isinstance(model, CalibratedClassifierCV):
                model.estimator.fit(X, y)
                model = model.estimator
    else:
        print(f"⚠️ Warning: Dataset contains only one class. Model remains untrained for this ticker.")
    
    oos_series = pd.Series(oos_preds_prob, index=oos_indices).sort_index()
    oos_series = oos_series[~oos_series.index.duplicated(keep='last')]
    
    return model, oos_series

def run_training_pipeline(ticker="RELIANCE.NS"):
    df = prepare_dataset(ticker, horizon=1)
    if df is None or len(df) < 30:
        raise ValueError("Dataset is too small for Machine Learning training.")
        
    train_df = df.dropna(subset=['Target_Return']).copy()
    
    exclude = ['Open', 'High', 'Low', 'Close', 'Target_Return', 'Target_Class', 'Risk_Level']
    features = [c for c in df.columns if c not in exclude]
    
    X_train = train_df[features].ffill().fillna(0)
    y_train = train_df['Target_Class']

    # --- 1. Correlation Filtering (Collinearity Pruning) ---
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    orthogonal_features = [f for f in features if f not in to_drop]
    print(f"\n🧹 Collinearity Filter: Dropped {len(to_drop)} highly correlated/redundant features.")

    # --- 2. Advanced Feature Selection (Top 12) ---
    print("🌲 Running baseline Random Forest for elite feature extraction...")
    selector_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    # Check if we have two classes before fitting selector
    if len(np.unique(y_train)) >= 2:
        selector_rf.fit(X_train[orthogonal_features], y_train)
        selector_importances = pd.Series(selector_rf.feature_importances_, index=orthogonal_features).sort_values(ascending=False)
        top_n = min(12, len(orthogonal_features))
        features = selector_importances.head(top_n).index.tolist()
    else:
        # Fallback if impossible to calculate importance
        features = orthogonal_features[:12]
        
    print(f"🎯 Feature Pruning Complete. Selected the Top {len(features)} elite orthogonal predictors.")
    
    # Strictly truncate the training pipeline mathematically
    X_train = X_train[features]
    
    print("\n" + "="*50)
    print(f"🚀 TRAINING CALIBRATED ENSEMBLE on {len(features)} Orthogonal Features")
    print("="*50)
    
    # Wrap models in CalibratedClassifierCV (Platt Scaling)
    models = {
        "LogReg": CalibratedClassifierCV(
            estimator=LogisticRegression(max_iter=1000, class_weight='balanced'), 
            method='sigmoid', cv=3
        ),
        "RandomForest": CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=200, max_depth=7, min_samples_leaf=5,
                random_state=42, class_weight='balanced'
            ),
            method='sigmoid', cv=3
        )
    }
    
    if XGBClassifier is not None:
         models["XGBoost"] = CalibratedClassifierCV(
             estimator=XGBClassifier(
                 n_estimators=100, max_depth=4, learning_rate=0.05, 
                 subsample=0.8, colsample_bytree=0.8, random_state=42
             ),
             method='sigmoid', cv=3
         )
         
    trained_models = {}
    oos_signals = {}  # These will now contain probabilities
    
    for name, model in models.items():
        print(f"\nTraining {name} with Calibration...")
        trained, oos_pred = train_and_validate(X_train, y_train, model, n_splits=5)
        trained_models[name] = trained
        oos_signals[name] = oos_pred
        
    # Voter Logic using continuous probabilities
    all_oos = pd.DataFrame(oos_signals)
    avg_oos_prob = all_oos.mean(axis=1)
    # Threshold for backtesting needs to be binary (1 or 0)
    oos_signals["Ensemble"] = (avg_oos_prob >= 0.50).astype(int)
    
    # Feature Importance (Extract from uncalibrated base estimator of RF)
    best_model = trained_models["RandomForest"]
    if isinstance(best_model, CalibratedClassifierCV):
        # The base estimator might be an unfitted object if calibration failed and we caught it,
        # but in most cases CalibratedClassifierCV provides .estimator or .calibrated_classifiers_
        if hasattr(best_model, 'calibrated_classifiers_') and len(best_model.calibrated_classifiers_) > 0:
            base_rf = best_model.calibrated_classifiers_[0].estimator
        else:
            base_rf = best_model.estimator
    else:
        base_rf = best_model
        
    if hasattr(base_rf, 'feature_importances_'):
        importances = base_rf.feature_importances_
        feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    else:
        feat_imp = pd.Series([1.0/len(features)]*len(features), index=features)
    
    print("\n🌟 Top 10 Important Features:")
    print(feat_imp.head(10))
    
    return trained_models, df, feat_imp, oos_signals

if __name__ == "__main__":
    res = run_training_pipeline("RELIANCE.NS")
    if res:
        models, df, feat_imp, oos = res
        print("✅ Pipeline test successful.")
