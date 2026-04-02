import pandas as pd
import numpy as np
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add moving averages, RSI, MACD, and Bollinger Bands using the 'ta' library.
    Requires a DataFrame with 'High', 'Low', 'Close', 'Volume' columns.
    
    Args:
        df: Pandas DataFrame containing price data.
        
    Returns:
        pd.DataFrame: A new DataFrame with added technical indicators.
    """
    print("📈 Calculating technical indicators...")
    # Copy to avoid view warnings
    data = df.copy()
    
    # Ensure columns exist
    required = ["High", "Low", "Close", "Volume"]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing required column for TA: {col}")

    # 1. Trend Indicators
    # Simple Moving Averages
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    
    # Exponential Moving Averages
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    
    # MACD
    macd = ta.trend.MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()
    
    # Average Directional Movement Index (ADX) - Trend Strength
    try:
        data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=14)
    except Exception:
        pass # ADX sometimes fails on flat data

    # 2. Momentum Indicators
    # Relative Strength Index
    data['RSI_14'] = ta.momentum.rsi(data['Close'], window=14)
    
    # Stochastic Oscillator
    data['Stoch_k'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
    data['Stoch_d'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)

    # 3. Volatility Indicators
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    data['BB_Width'] = bb.bollinger_wband()
    data['BB_Percent'] = bb.bollinger_pband()

    # Average True Range (ATR)
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)

    # 4. Volume Indicators
    # Volume Profile / VWAP
    data['VWAP'] = ta.volume.volume_weighted_average_price(
        high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=14
    )
    
    # Chaikin Money Flow
    data['CMF'] = ta.volume.chaikin_money_flow(
        high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=20
    )

    # 5. Custom Price Action Features
    # Daily Return
    data['Daily_Return'] = data['Close'].pct_change()
    
    # Log Return (better for ML)
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Distance from Moving Averages
    data['Dist_from_SMA20'] = (data['Close'] - data['SMA_20']) / data['SMA_20']
    
    # NEW: Momentum-Delta Features (Change in indicators)
    data['RSI_Change'] = data['RSI_14'] - data['RSI_14'].shift(1)
    data['Volume_Momentum'] = data['Volume'].pct_change(periods=5)
    data['MACD_Slope'] = data['MACD_Diff'].pct_change(periods=3)
    
    # 6. Market Regime Proxies
    # Structural Trend Filter
    data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
    data['Market_Trend_Regime'] = data['Close'] / (data['SMA_200'] + 1e-9)
    
    # Volatility Clustering Proxy
    rolling_atr_50 = data['ATR'].rolling(window=50).mean()
    data['Volatility_Regime'] = data['ATR'] / (rolling_atr_50 + 1e-9)
    
    # 7. Final Clean-up (Preventing 'Infinity' or 'Large Value' errors in ML)
    # Replace inf with NaN and then fill with 0
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.fillna(0)
    
    # Clip extreme outliers that can crash LogisticRegression (e.g., > 1000% moves in indicators)
    # We clip to a very large but finite range
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].clip(lower=-1e6, upper=1e6)
    
    initial_len = len(data)
    data = data.dropna()
    print(f"✅ Technical features generated. Dropped {initial_len - len(data)} rows.")
    
    return data

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.data_loader import fetch_stock_data
    
    df = fetch_stock_data("RELIANCE.NS", "2023-01-01", "2024-01-01")
    if not df.empty:
        df_ta = add_technical_indicators(df)
        print(f"Columns after TA: {len(df_ta.columns)}")
        print(df_ta[['Close', 'RSI_14', 'MACD', 'BB_High', 'BB_Low']].tail())
