import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# Ensure backend modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset_builder import prepare_dataset
from training.train_ensembles import run_training_pipeline
from backtesting.backtester import run_backtest

st.set_page_config(page_title="Quant AI Dashboard", layout="wide")

st.title("AI Quantitative Trading Intelligence System")
st.markdown("Predictive analysis using Machine Learning, Technical Indicators, and FinBERT News Sentiment. Assesses probabilities of daily market movements.")

# Sidebar for inputs
st.sidebar.header("System Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "RELIANCE.NS")
# Let's set the default start date 3 years ago to ensure we have enough data for ML convergence
default_start = pd.to_datetime("today") - pd.DateOffset(years=3)
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
run_btn = st.sidebar.button("Run Intelligence Engine")

st.sidebar.markdown("---")
st.sidebar.info("""
**How this works:**
1. Downloader gets market prices.
2. Google News fetches headlines.
3. FinBERT scores news sentiment.
4. Technical indicators (TA) created.
5. Emsemble ML Models train on history.
6. VectorBT backtests simulated returns.
""")

if run_btn:
    with st.spinner(f"Fetching data, parsing news, computing TA, and training ML models for {ticker}..."):
        try:
            res = run_training_pipeline(ticker, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))
        except Exception as e:
            st.error(f"Execution Error: {e}")
            st.warning(f"Troubleshooting Tips:\n1. If this is a recent IPO, data is only available from its public listing date.\n2. Ensure the ticker symbol exactly matches Yahoo Finance (e.g., Lenskart is not publicly listed yet. For Indian stocks, you must use the '.NS' suffix like 'RELIANCE.NS').")
            res = None

    if res is not None:
        models, df, feat_imp = res
        model = models["RandomForest"]

        # Current Price & Data
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = ((latest_price - prev_price) / prev_price) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Close Price", f"[{ticker}] {latest_price:.2f}", f"{change:.2f}%")
        
        # Next Day Prediction Processing
        exclude = ['Open', 'High', 'Low', 'Close', 'Target_Return', 'Target_Class', 'Risk_Level']
        features = [c for c in df.columns if c not in exclude]
        
        X = df[features].ffill().fillna(0)
        
        # Predict the latest row (today's close predicting tomorrow's movement)
        latest_features = X.iloc[-1:]
        pred = model.predict(latest_features)[0]
        prob = model.predict_proba(latest_features)[0]
        
        # Recent Sentiment
        recent_sentiment = df['avg_sentiment'].iloc[-1]
        
        # Signal Generation logic with Risk/Confidence Thresholds
        if pred == 1 and prob[1] > 0.55:
            signal = "BUY"
            conf = prob[1] * 100
        elif pred == 1 and prob[1] <= 0.55:
            signal = "HOLD (Weak Buy)"
            conf = prob[1] * 100
        elif pred == 0 and prob[0] > 0.55:
            signal = "SELL"
            conf = prob[0] * 100
        else:
            signal = "HOLD (Weak Sell)"
            conf = prob[0] * 100

        col2.metric("AI Signal (Next Interval Target)", signal)
        col3.metric("Prediction Confidence", f"{conf:.1f}%")

        # Visualizations
        st.write("---")
        st.subheader(f"Price Action Profile ({ticker})")
        
        fig = go.Figure()
        # Candlestick or Line
        fig.add_trace(go.Scatter(x=df.index[-200:], y=df['Close'].iloc[-200:], mode='lines', name='Close Price', line=dict(color='royalblue')))
        if 'BB_High' in df.columns:
             fig.add_trace(go.Scatter(x=df.index[-200:], y=df['BB_High'].iloc[-200:], mode='lines', name='BB High', line=dict(color='gray', dash='dash')))
             fig.add_trace(go.Scatter(x=df.index[-200:], y=df['BB_Low'].iloc[-200:], mode='lines', name='BB Low', line=dict(color='gray', dash='dash')))
        
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        colA, colB = st.columns(2)
        
        with colA:
            st.subheader("Model Feature Importance")
            st.bar_chart(feat_imp.head(10))
            
        with colB:
            st.subheader("Market Context")
            st.write(f"**Latest Average Sentiment:** {recent_sentiment:.2f} (Scale: -1 to +1)")
            st.write(f"**Current RSI (14):** {df['RSI_14'].iloc[-1]:.2f}")
            macd_sgn = "Bullish" if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "Bearish"
            st.write(f"**Current MACD Trend:** {macd_sgn}")
            volatility = df['Risk_Level'].iloc[-1] * 100
            st.write(f"**Systematic Volatility (20-day):** {volatility:.2f}%")
        
        # Backtest Report
        st.write("---")
        st.subheader("Historical VectorBT Simulation")
        st.markdown("This backtest simulates using the trained AI logic systematically over the dataset.")
        
        # Generate signals on the whole dataset
        signals = pd.Series(model.predict(X), index=df.index)
        pf, metrics = run_backtest(df, signals)
        
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        bcol1.metric("Strategy Yield", f"{metrics['Strategy Total Return (%)']:.2f}%")
        bcol2.metric("Buy & Hold Yield", f"{metrics['Benchmark Total Return (%)']:.2f}%")
        bcol3.metric("Win Rate", f"{metrics['Win Rate (%)']:.2f}%")
        bcol4.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")

        st.success("Analysis Complete. Pipeline execution successful.")
