import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from training.dataset_builder import prepare_dataset
from training.train_ensembles import run_training_pipeline

res = run_training_pipeline('GC=F')
models, df, feat_imp, oos_signals = res

print('\n--- Diagnostic ---')
print("LogReg prob stats:")
print(oos_signals['LogReg'].describe())
print("RF prob stats:")
print(oos_signals['RandomForest'].describe())
print("XGB prob stats:")
print(oos_signals['XGBoost'].describe())
print("Ensemble Output Stats:")
print(oos_signals['Ensemble'].describe())

