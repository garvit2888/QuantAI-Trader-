import yfinance as yf
import pandas as pd
from typing import Optional

def fetch_stock_data(ticker: str, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch all historical OHLCV data from Yahoo Finance.
    
    Args:
        ticker (str): The stock symbol (e.g., 'AAPL' or 'RELIANCE.NS')
        interval (str): Data interval ('1d' for daily, '1h' for hourly, etc.)
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing Open, High, Low, Close, Volume data.
    """
    print(f"📥 Fetching market data for {ticker} (Interval: {interval})...")
    try:
        data = yf.download(ticker, period="max", interval=interval, progress=False)
        
        if data.empty:
            raise ValueError(f"No data fetched for {ticker}. Check the ticker symbol and date range.")
            
        # yf > 0.2.0 sometimes returns MultiIndex columns. Flatten if necessary.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]
            data.columns = [c.replace(f"_{ticker}", "") for c in data.columns] # Remove ticker suffix
            
        # Standardize column names (Capitalized)
        rename_map = {
            col: col.replace("Price_", "").replace("Adj Close", "Close")  # Simplified handling
            for col in data.columns
        }
        data = data.rename(columns=rename_map)
        
        # Ensure we just keep standard OHLCV
        cols_to_keep = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in data.columns]
        data = data[cols_to_keep]

        data.index.name = "Date"
        print(f"✅ Successfully fetched {len(data)} rows of market data.")
        return data

    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test the fetcher
    df = fetch_stock_data("RELIANCE.NS", start_date="2020-01-01", end_date="2023-12-31")
    print(df.head())
