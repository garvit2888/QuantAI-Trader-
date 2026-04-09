import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset_builder import prepare_dataset
from engine.regime_detector import MarketRegimeDetector

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

def run_training_pipeline(ticker="RELIANCE.NS", lookback_years=None):
    df = prepare_dataset(ticker, horizon=1, lookback_years=lookback_years)
    if df is None or len(df) < 30:
        raise ValueError("Dataset is too small for Machine Learning training.")
        
    train_df = df.dropna(subset=['Target_Return']).copy()
    
    exclude = ['Open', 'High', 'Low', 'Close', 'Target_Return', 'Target_Class', 'Risk_Level']
    features = [c for c in df.columns if c not in exclude]
    
    X_train = train_df[features].ffill().fillna(0)
    y_train = train_df['Target_Class']

    # --- 1. Spectral Denoising (PCA Integration) ---
    print(f"\n🧪 Applying Spectral Denoising (PCA) to {len(features)} features...")
    
    # PCA requires scaled data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # We choose components that explain 95% of the variance to filter out noise tail
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_feature_names = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
    X_train_final = pd.DataFrame(X_pca, columns=pca_feature_names, index=X_train.index)
    
    print(f"✅ PCA Denoising Complete: Condensed {len(features)} features into {X_pca.shape[1]} Principal Factors.")
    print(f"📊 Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    print("\n" + "="*50)
    print(f"🚀 TRAINING CALIBRATED ENSEMBLE with ElasticNet Regularization")
    print("="*50)
    
    # Wrap models in CalibratedClassifierCV (Platt Scaling)
    # REGULARIZATION UPGRADES:
    # 1. LogReg -> ElasticNet (L1 + L2)
    # 2. XGBoost -> reg_alpha + reg_lambda
    # 3. RF -> Reduced max_features for better pruning
    models = {
        "LogReg": CalibratedClassifierCV(
            estimator=LogisticRegression(
                penalty='elasticnet', 
                solver='saga', 
                l1_ratio=0.5, 
                max_iter=2000, 
                class_weight='balanced'
            ), 
            method='sigmoid', cv=3
        ),
        "RandomForest": CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=200, max_depth=7, min_samples_leaf=5,
                max_features='sqrt',
                random_state=42, class_weight='balanced'
            ),
            method='sigmoid', cv=3
        )
    }
    
    if XGBClassifier is not None:
         models["XGBoost"] = CalibratedClassifierCV(
             estimator=XGBClassifier(
                 n_estimators=100, max_depth=4, learning_rate=0.05, 
                 subsample=0.8, colsample_bytree=0.8, 
                 reg_alpha=0.1, reg_lambda=1.0, # L1 and L2 Regularization
                 random_state=42
             ),
             method='sigmoid', cv=3
         )
         
    trained_models = {}
    oos_signals = {}  # These will now contain probabilities
    
    for name, model in models.items():
        print(f"\nTraining {name} with Calibration...")
        trained, oos_pred = train_and_validate(X_train_final, y_train, model, n_splits=5)
        trained_models[name] = trained
        oos_signals[name] = oos_pred
        
    # Voter Logic using continuous probabilities
    all_oos = pd.DataFrame(oos_signals)
    avg_oos_prob = all_oos.mean(axis=1)
    # Raising threshold to 0.59 to filter out random noise and weak predictions 
    oos_signals["Ensemble"] = (avg_oos_prob >= 0.59).astype(int)
    
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
        # 1. Get importance of Principal Components
        pc_importances = base_rf.feature_importances_
        
        # 2. Map PC importance back to original features using PCA loadings
        # PCA.components_ is (n_components, n_features)
        # We take the absolute loadings to see which original features contribute most to important PCs
        loadings = np.abs(pca.components_)
        
        # Weighted sum: multiply each PC's importance by its feature loadings
        # Transpose loadings to get (n_features, n_components) then dot with pc_importances (n_components,)
        feat_scores = np.dot(loadings.T, pc_importances)
        
        feat_imp = pd.Series(feat_scores, index=features).sort_values(ascending=False)
    else:
        feat_imp = pd.Series([1.0/len(features)]*len(features), index=features)
    
    # --- 4. Market Regime Intelligence ---
    print("\n🔍 Identifying Structural Market Regimes...")
    regime_detector = MarketRegimeDetector(n_regimes=4).fit(df)
    current_regime, _ = regime_detector.predict(df)
    print(f"📊 Current Sector Regime Detected: {current_regime}")
    
    return trained_models, df, feat_imp, oos_signals, scaler, pca, features, regime_detector

if __name__ == "__main__":
    res = run_training_pipeline("RELIANCE.NS")
    if res:
        models, df, feat_imp, oos, scaler, pca, trained_features, regime_detector = res
        print("✅ Pipeline test successful.")
