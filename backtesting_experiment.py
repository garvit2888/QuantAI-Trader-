import sys
import os
import pandas as pd
import numpy as np

sys.path.append('.')
from backtesting.backtester import run_backtest
from training.dataset_builder import prepare_dataset

df = prepare_dataset('GC=F')
# Load models separately to avoid the infinite loop issue we keep hitting with run_training_pipeline
print(f"Data loaded: {len(df)} rows")
