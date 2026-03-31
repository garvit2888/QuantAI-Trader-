# 📈 QuantAI Trader

An intelligent, fully open-source Stock Market Prediction Platform built from scratch using Machine Learning, Time-Series Modeling, Technical Analysis, and Natural Language Processing (News Sentiment). 

QuantAI Trader automatically downloads historical stock data, computes technical indicators, analyzes the real-time news sentiment over Google News using HuggingFace's **FinBERT** transformer, trains ensemble algorithms, and provides actionable **BUY / HOLD / SELL** probabilities for the next trading interval.

## 🔥 Key Features
* **Data Ingestion:** Fetches free historical market data through `yfinance`. Extracts live financial news headlines via Google News RSS.
* **Feature Engineering:** Implements 20+ features, including RSI, MACD, Bollinger Bands, Moving Averages (EMA, SMA), and Volume Weighted Average Price (VWAP) using the `ta` library.
* **Sentiment Analysis:** Utilizes `FinBERT` (a financial-domain NLP model) to score the current market news from Google RSS and maps numerical metrics directly into the trading datasets.
* **Machine Learning Forecaster:** Fully trains decision models like `RandomForest`, `LogisticRegression` to make day-trading or long-term baseline probability predictions. Prevent data leakage using strict **TimeSeriesSplit Walk-Forward Validation**.
* **Deep Sequence Engine (LSTM):** Includes custom PyTorch LSTM blocks internally structuring historical sequence arrays.
* **Backtesting Engine:** Fast vectorized backtests using `vectorbt` comparing the AI's predictions to benchmark 'Buy & Hold' returns (providing Sharpe Ratios, Win Rates, and Max Drawdowns).
* **Interactive Dashboard:** Complete end-to-end user frontend built on **Streamlit** with Plotly graphical analysis.

---

## 🛠 Project Structure

```bash
📦 QuantAI Trader
├── 📂 data/             # yfinance OHLCV fetchers & News RSS web scrapers
├── 📂 features/         # Algorithm computing MACD, RSI, Bollingers
├── 📂 sentiment/        # Transformers Pipeline & FinBERT inference
├── 📂 training/         # ML dataset assemblers & Walk-Forward Validation algorithms
├── 📂 models/           # Exported serializations (joblib/pth)
├── 📂 backtesting/      # Python VectorBT quantitative simulation models
└── 📂 dashboard/        # Streamlit Main App & UI configuration
```

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/garvit2888/QuantAI-Trader.git
   cd QuantAI-Trader
   ```

2. **Set up the Virtual Environment & Dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Launch the Intelligence Engine**
   ```bash
   streamlit run dashboard/app.py
   ```

*(On the very first run, FinBERT's transformer models will automatically download from HuggingFace to your local cache. It will be much faster in the future.)*

## 📝 Usage Tips
- **For US Stocks:** Enter standard tickers (e.g., `AAPL`, `MSFT`, `TSLA`).
- **For Indian/NSE Stocks:** Use the `.NS` suffix exactly as they exist on Yahoo Finance (e.g., `PAYTM.NS`, `RELIANCE.NS`, `ZOMATO.NS`).
- Due to the rolling nature of the predictive algorithms (e.g. 20-day averages), if testing on a very recently public IPO, ensure there is at least **~30 days of public trading history**, or the system cannot converge the math formulas!
