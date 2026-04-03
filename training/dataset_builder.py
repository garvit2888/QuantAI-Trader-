import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

def create_targets(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Create classification and regression targets.
    
    Args:
        df: DataFrame with technical indicators and price data.
        horizon: Number of days ahead to predict.
        
    Returns:
        DataFrame with added target columns.
    """
    data = df.copy()
    
    # Target 1: Regression target (Expected Return)
    # The return from today's Close to the Close `horizon` days ahead.
    data['Target_Return'] = data['Close'].shift(-horizon) / data['Close'] - 1.0
    
    # Target 2: Classification target (Probability of UP)
    # 1 if Target_Return > 0 else 0
    data['Target_Class'] = (data['Target_Return'] > 0).astype(int)
    
    # Target 3: Risk Level (Historical Volatility over 20 days)
    data['Risk_Level'] = data['Close'].pct_change().rolling(window=20).std()
    
    # Do NOT drop rows where Target_Return is NaN (that is 'Today' which we want to predict).
    # Only drop rows where the indicators are NaN (the beginning of the time series).
    features = [c for c in data.columns if c not in ['Target_Return', 'Target_Class']]
    initial_len = len(data)
    data = data.dropna(subset=features)
    print(f"🎯 Target generation complete. Dropped {initial_len - len(data)} leading NaN rows.")
    return data

def get_train_test_splits(X, y, n_splits=5):
    """
    Returns TimeSeriesSplit indices to prevent walk-forward lookahead bias.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X, y))

from typing import Optional

def prepare_dataset(ticker: str, horizon: int = 1, lookback_years: Optional[float] = None):
    """
    Combines Data Loader, Features, and Targets.
    
    NOTE on Sentiment: Google News RSS only provides recent news (last few days/weeks). 
    We will append it, but historical training data will likely have 0 sentiment.
    The sentiment heavily influences recent/live predictions.
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.data_loader import fetch_stock_data
    from features.technical_indicators import add_technical_indicators
    from data.news_loader import fetch_google_news
    from sentiment.sentiment_analyzer import compute_daily_sentiment
    
    print("\n" + "="*50)
    print(f"🏗️  BUILDING DATASET FOR {ticker}")
    print("="*50)
    
    # 1. Load Data
    df = fetch_stock_data(ticker)
    if df.empty:
        raise ValueError("Failed to fetch market data.")
        
    if lookback_years is not None:
        try:
            start_date = pd.Timestamp.now() - pd.DateOffset(days=int(365 * lookback_years))
            # Handle timezone neutrality safely
            if df.index.tz is not None:
                start_date = start_date.tz_localize(df.index.tzinfo)
            df = df[df.index >= start_date]
            print(f"✂️ Sliced dataset to last {lookback_years} years (from {start_date.date()})")
        except Exception as e:
            print(f"⚠️ Could not slice date range: {e}")
            
    # 2. Add Tech Indicators
    df = add_technical_indicators(df)
    
    # 3. Add Sentiment
    # Cap to 10 to massively speed up FinBERT CPU inference time
    news = fetch_google_news(ticker, max_results=10)
    sentiment_df = compute_daily_sentiment(news)
    
    # Merge Sentiment
    df = df.reset_index()
    if not sentiment_df.empty:
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.date
        
        # Left join to preserve all market days
        df = pd.merge(df, sentiment_df, on='Date', how='left')
        
    if 'avg_sentiment' not in df.columns:
        df['avg_sentiment'] = 0.0
        df['news_count'] = 0.0
    else:
        # Forward fill recent sentiment (e.g. weekend news to Monday, or yesterday to today)
        df['avg_sentiment'] = df['avg_sentiment'].ffill(limit=5).fillna(0.0)
        df['news_count'] = df['news_count'].ffill(limit=5).fillna(0.0)
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
        
    # 4. Create Targets
    df = create_targets(df, horizon=horizon)
    
    print(f"\n✅ FINAL DATASET: {df.shape[0]} rows, {df.shape[1]} columns ready for ML.")
    return df

if __name__ == "__main__":
    df = prepare_dataset("RELIANCE.NS", "2020-01-01", "2024-01-01")
    if df is not None:
        print(df[['Close', 'RSI_14', 'Target_Return', 'Target_Class']].tail())
