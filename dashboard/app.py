import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
import warnings

# QuantAI Dashboard - Alpha Booster Upgrade Sync
warnings.filterwarnings('ignore')

# Ensure backend modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset_builder import prepare_dataset
from training.train_ensembles import run_training_pipeline
from backtesting.backtester import run_backtest
from data.news_loader import fetch_google_news

def generate_ai_commentary(ticker, signal, conf, sentiment, feat_imp, metrics, df):
    """
    Generate a human-readable trading summary based on all system outputs.
    """
    top_feat = feat_imp.index[0]
    top_val = df[top_feat].iloc[-1]
    win_rate = metrics['Win Rate (%)']
    sharpe = metrics['Sharpe Ratio']
    
    # 1. Consensus & The "Edge"
    edge = f"The Calibrated Ensemble is **{conf:.1f}% probabilistically confident** in a **{signal}** direction. "
    if conf >= 75:
        edge += "This represents an exceptionally strong calibrated edge, indicating a high statistical probability of the trade succeeding based on historical mapping."
    elif conf >= 65:
        edge += "The underlying models have calculated a solid statistical edge, fully supporting the trade direction."
    else:
        edge += "The probability calibration shows internal variance, suggesting a more cautious or volatile entry."

    # 2. The Catalyst (Sentiment)
    if sentiment > 0.5:
        catalyst = f"**Strong Bullish Sentiment ({sentiment:.2f}):** The latest financial news headlines are overwhelmingly positive, acting as a powerful fundamental tailwind for this trade."
    elif sentiment < -0.5:
        catalyst = f"**Strong Bearish Sentiment ({sentiment:.2f}):** Negative news headlines are creating significant selling pressure, which aligns with the AI's downward target."
    else:
        catalyst = f"**Neutral Sentiment ({sentiment:.2f}):** The news feed is currently balanced. This trade is being driven primarily by technical price action rather than news catalysts."

    # 3. Technical Reasoning
    reasoning = f"The primary mathematical driver today is **{top_feat}** (Value: {top_val:.4f}). "
    if "RSI" in top_feat:
        reasoning += "The AI is currently prioritizing overall momentum and overbought/oversold levels to time this entry."
    elif "BB" in top_feat or "SMA" in top_feat:
        reasoning += "The model is focused on mean-reversion, specifically how far the price has deviated from its historical average."
    elif "Return" in top_feat:
        reasoning += "The engine is 'chasing' the immediate short-term trend, betting on a continuation of today's price velocity."
    else:
        reasoning += "This indicator is providing the highest statistical signal for predicting tomorrow's direction based on historical patterns."

    # 4. The "Reality Check" (Risk)
    risk = f"**Risk Assessment:** Historically, this system has a **{win_rate:.1f}% Win Rate** on {ticker}. "
    if sharpe < 0.5:
        risk += f"The Sharpe Ratio of {sharpe:.2f} is relatively low, meaning that while the system is profitable, the price action for {ticker} can be extremely 'choppy' and volatile. **Strict stop-losses are mandatory.**"
    elif sharpe > 1.0:
        risk += f"The Sharpe Ratio of {sharpe:.2f} is excellent, indicating this has historically been a very smooth and reliable trend for the AI to capture."
    else:
        risk += f"The Sharpe Ratio of {sharpe:.2f} indicates moderate risk-adjusted returns."

    return {
        "edge": edge,
        "catalyst": catalyst,
        "reasoning": reasoning,
        "risk": risk
    }

st.set_page_config(page_title="Quant AI Dashboard", layout="wide")

st.title("AI Quantitative Trading Intelligence System")
st.markdown("Predictive analysis using Machine Learning, Technical Indicators, and FinBERT News Sentiment. Assesses probabilities of daily market movements.")

# Sidebar for inputs
st.sidebar.header("System Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "RELIANCE.NS")
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
            # Auto-fetches the absolute maximum historical data available for best prediction accuracy
            res = run_training_pipeline(ticker)
        except Exception as e:
            st.error(f"Execution Error: {e}")
            st.warning(f"Troubleshooting Tips:\n1. If this is a recent IPO, data is only available from its public listing date.\n2. Ensure the ticker symbol exactly matches Yahoo Finance (e.g., Lenskart is not publicly listed yet. For Indian stocks, you must use the '.NS' suffix like 'RELIANCE.NS').")
            res = None

    if res is not None:
        models, df, feat_imp, oos_signals = res
        # Use Ensemble OOS for backtesting
        if "Ensemble" in oos_signals:
            oos_preds = oos_signals["Ensemble"]
        else:
            # Fallback if Ensemble signal is missing
            oos_preds = oos_signals[next(iter(oos_signals))]

        # Align DF and Run Backtest early for metrics availability
        oos_df = df.loc[oos_preds.index]
        pf, metrics = run_backtest(oos_df, oos_preds)

        # Current Price & Data
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = ((latest_price - prev_price) / prev_price) * 100

        # --- 1. Top Level Metrics & Profile ---
        st.subheader(f"📊 Trading Profile: {ticker}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Asset Price", f"{latest_price:.2f}", f"{change:.2f}%")
        
        # The exact elite orthogonal features in the EXACT sequence the models were trained on
        m_sample = next(iter(models.values()))
        trained_features = list(m_sample.feature_names_in_)
        
        X = df[trained_features].ffill().fillna(0)
        
        # Recent Sentiment
        recent_sentiment = df['avg_sentiment'].iloc[-1]
        
        # Multi-Model Continuous Probabilities via Calibration
        latest_features = X.iloc[-1:]
        probas = []
        for name, m in models.items():
            try:
                prob = m.predict_proba(latest_features)[0][1]
            except Exception:
                # Fallback if a specific model failed to train class probabilities
                prob = m.predict(latest_features)[0]
            probas.append(prob)
        
        # Exact calibrated statistical mean probability
        avg_prob = sum(probas) / len(probas)
        
        # Signal Generation (Based on Statistical Edges)
        if avg_prob >= 0.65:
            signal = "BUY (Strong)"
            conf = avg_prob * 100
        elif avg_prob > 0.50:
            signal = "HOLD (Weak Buy)"
            conf = avg_prob * 100
        elif avg_prob <= 0.35:
            signal = "SELL (Strong)"
            conf = (1.0 - avg_prob) * 100
        else:
            signal = "HOLD (Weak Sell)"
            conf = (1.0 - avg_prob) * 100

        col2.metric("AI Calibrated Signal", signal)
        col3.metric("True Probability", f"{conf:.1f}%")

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

        # --- AI Intelligence Report (Virtual Analyst) ---
        st.write("---")
        with st.container():
            st.subheader("🧠 AI Intelligence Report & Executive Summary")
            report = generate_ai_commentary(ticker, signal, conf, recent_sentiment, feat_imp, metrics, df)
            
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"### The Strategy Edge\n{report['edge']}")
                st.info(f"### The Market Catalyst\n{report['catalyst']}")
            with c2:
                st.success(f"### Mathematical Reasoning\n{report['reasoning']}")
                st.warning(f"### Transparency & Risk\n{report['risk']}")

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
            
        # --- Data Transparency & Explainability ---
        st.write("---")
        st.subheader("Data Transparency & AI Decision Log")
        st.markdown("Building trust requires transparency. Below is the exact live data the intelligence engine processed to calculate the confidence score.")
        
        tcol1, tcol2, tcol3 = st.columns(3)
        
        with tcol1:
            st.markdown("#### 1. Why Did The AI Choose This?")
            st.markdown("These are the Top 5 mathematical drivers that influenced the algorithm, along with their values today:")
            top_5_feats = feat_imp.head(5).index.tolist()
            for feat in top_5_feats:
                current_val = df[feat].iloc[-1]
                st.write(f"- **{feat}**: `{current_val:.4f}`")
                
        with tcol2:
            st.markdown("#### 2. Raw Market Inputs")
            st.markdown("The most recent OHLCV market history used for the technical indicators:")
            st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5), use_container_width=True)
            
        with tcol3:
            st.markdown("#### 3. Live News Feed")
            st.markdown("The exact headlines fetched and scored by the FinBERT NLP engine:")
            recent_news = fetch_google_news(ticker, max_results=5)
            if not recent_news.empty:
                for idx, row in recent_news.iterrows():
                    pub_str = str(row['published_at'])[:10]
                    st.markdown(f"- **{pub_str}**: [{row['title']}]({row['link']})")
            else:
                st.write("*No recent news found for this ticker.*")
        
        # Backtest Report (USING OUT-OF-SAMPLE DATA)
        st.write("---")
        st.subheader("Historical VectorBT Simulation (Out-Of-Sample)")
        st.markdown("This backtest uses **Walk-Forward Validation**. It only shows performance on data the AI had *never seen before* making each trade. This is a realistic representation of real-world performance.")
        
        # Align DF with OOS signals
        # (Moved up for metrics availability)
        
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        bcol1.metric("Strategy Yield", f"{metrics['Strategy Total Return (%)']:.2f}%")
        bcol2.metric("Buy & Hold Yield", f"{metrics['Benchmark Total Return (%)']:.2f}%")
        bcol3.metric("Win Rate", f"{metrics['Win Rate (%)']:.2f}%")
        bcol4.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")

        st.success("Analysis Complete. Pipeline execution successful.")
