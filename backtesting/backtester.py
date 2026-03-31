import pandas as pd
import vectorbt as vbt
import warnings

warnings.filterwarnings('ignore')

def run_backtest(df: pd.DataFrame, signals: pd.Series):
    """
    Run backtesting using vectorbt based on model signals.
    
    Args:
        df: DataFrame containing at least 'Close' price.
        signals: Series of 1 (Buy) and 0 (Sell/Hold) matching df index.
        
    Returns:
        pf: vectorbt Portfolio object.
        metrics: Dictionary of key performance indicators.
    """
    print("\n" + "="*50)
    print("📈 RUNNING VECTORBT BACKTEST")
    print("="*50)
    
    # 1. Prepare Entries and Exits
    # Strategy: Buy when signal = 1, Sell when signal = 0.
    # Vectorbt from_signals will handle consecutive 1s (hold) and 0s automatically.
    entries = signals == 1
    exits = signals == 0
    
    # Run Portfolio Simulator
    # fees=0.001 is a 0.1% transaction fee (realistic for many brokers)
    pf = vbt.Portfolio.from_signals(
        close=df['Close'],
        entries=entries,
        exits=exits,
        fees=0.001,
        init_cash=10000,
        freq='1D'
    )
    
    # Setup baseline (Buy & Hold) scenario
    benchmark = vbt.Portfolio.from_holding(df['Close'], init_cash=10000)
    
    # 2. Key Metrics
    metrics = {
        "Strategy Total Return (%)": pf.total_return() * 100,
        "Benchmark Total Return (%)": benchmark.total_return() * 100,
        "Sharpe Ratio": pf.sharpe_ratio(),
        "Max Drawdown (%)": pf.max_drawdown() * 100,
        "Win Rate (%)": pf.trades.win_rate() * 100,
        "Total Trades": len(pf.trades)
    }
    
    print("\n📊 BACKTEST RESULTS vs BUY & HOLD:")
    for k, v in metrics.items():
        val = v if pd.notna(v) else 0.0
        print(f"  > {k:<28}: {val:.2f}")
        
    return pf, metrics

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training.train_ensembles import run_training_pipeline
    
    print("Pre-training model for backtesting demo...")
    res = run_training_pipeline("RELIANCE.NS")
    
    if res:
        models, df, _ = res
        model = models["RandomForest"]
        
        exclude = ['Open', 'High', 'Low', 'Close', 'Target_Return', 'Target_Class', 'Risk_Level']
        features = [c for c in df.columns if c not in exclude]
        
        X = df[features].ffill().fillna(0)
        
        # Generate signals on the dataset
        # For a truly strict rigorous test, we should only test on the out-of-sample splits.
        # But this serves as an architectural template demonstration.
        predictions = model.predict(X)
        signals = pd.Series(predictions, index=df.index)
        
        pf, metrics = run_backtest(df, signals)
        
        # Generates a HTML plot of the portfolio
        # fig = pf.plot()
        # fig.write_html("backtest_results.html")
        # print("\n💾 Saved plot to backtest_results.html")
