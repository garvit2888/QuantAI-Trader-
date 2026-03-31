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
    
    # Drop rows where target is NaN (at the end of the dataset and some start indicators)
    initial_len = len(data)
    data = data.dropna()
    print(f"🎯 Target generation complete. Dropped {initial_len - len(data)} NaN rows.")
    return data

def get_train_test_splits(X, y, n_splits=5):
    """
    Returns TimeSeriesSplit indices to prevent walk-forward lookahead bias.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X, y))

def prepare_dataset(ticker: str, start_date: str, end_date: str, horizon: int = 1):
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
    df = fetch_stock_data(ticker, start_date, end_date)
    if df.empty:
        raise ValueError("Failed to fetch market data.")
        
    # 2. Add Tech Indicators
    df = add_technical_indicators(df)
    
    # 3. Add Sentiment
    news = fetch_google_news(ticker, max_results=20)
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
        df['avg_sentiment'] = df['avg_sentiment'].fillna(0.0)
        df['news_count'] = df['news_count'].fillna(0.0)
        
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
