import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import datetime
import pytz
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

st.set_page_config(page_title="QuantAI Terminal", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS INJECTION ---
# Institutional Dark Theme & Glassmorphism
st.markdown("""
<style>
    /* Global Background and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    
    html, body {
        font-family: 'Inter', sans-serif !important;
        background-color: #050505 !important; 
    }
    
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
        color: #E5E5E5;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 99999 !important;
    }
    
    /* Clean Top Bar elements rather than hiding the whole header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Ensure the sidebar toggle itself is extremely visible */
    [data-testid="collapsedControl"] {
        z-index: 999999 !important;
        background-color: rgba(11, 11, 11, 0.5) !important;
        border-radius: 4px;
    }
    [data-testid="collapsedControl"] svg {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    
    /* Custom Card Glassmorphism */
    .quant-card {
        background: rgba(11, 11, 11, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid #1A1A1A;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .quant-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.1);
        border-color: #2A2A2A;
    }
    
    /* Signal Specifics */
    .signal-BUY { color: #22C55E; text-shadow: 0 0 10px rgba(34, 197, 94, 0.3); }
    .signal-SELL { color: #EF4444; text-shadow: 0 0 10px rgba(239, 68, 68, 0.3); }
    .signal-HOLD { color: #3B82F6; text-shadow: 0 0 10px rgba(59, 130, 246, 0.3); }
    
    /* Custom Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
        letter-spacing: -0.5px;
        color: #FFFFFF;
    }
    
    hr {
        border-color: #1A1A1A;
        margin: 2rem 0;
    }
    
    /* Hide Streamlit default metric styling to use our own */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }
    
    /* Responsive Grid Layouts */
    .unified-metrics-card {
        display: flex;
        flex-direction: column;
        gap: 20px;
        padding: 20px;
        background: rgba(11, 11, 11, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid #1A1A1A;
        border-radius: 8px;
    }
    
    .metrics-row {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: center;
        width: 100%;
    }
    
    .metric-segment {
        flex: 1;
        text-align: center;
        border-right: 1px solid #1A1A1A;
        padding: 0 10px;
    }
    .metric-segment:last-child {
        border-right: none;
    }
    .metric-segment.signal-area {
        flex: 1.5;
        text-align: left;
        padding-right: 15px;
        border-left: 4px solid #3B82F6;
        padding-left: 15px;
    }
    
    /* Mobile Media Queries */
    @media (max-width: 768px) {
        .unified-metrics-card {
            flex-direction: column;
            align-items: flex-start;
            padding: 15px;
            border-left: none;
            border-top: 4px solid #3B82F6;
        }
        .metric-segment {
            flex: none;
            width: 100%;
            text-align: left;
            border-right: none;
            border-bottom: 1px solid #1A1A1A;
            padding: 10px 0;
        }
        .metric-segment:last-child {
            border-bottom: none;
        }
        .metric-segment.signal-area {
            padding-right: 0;
        }
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.4rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# --- PARTICLE BACKGROUND INJECTION ---
components.html("""
<script>
    const parentWindow = window.parent;
    const parentDoc = parentWindow.document;
    
    if (!parentDoc.getElementById('quant-starfield')) {
        const canvas = parentDoc.createElement('canvas');
        canvas.id = 'quant-starfield';
        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100vw';
        canvas.style.height = '100vh';
        canvas.style.pointerEvents = 'none';
        canvas.style.zIndex = '0'; 
        parentDoc.body.insertBefore(canvas, parentDoc.body.firstChild);
        
        const ctx = canvas.getContext('2d');
        let width, height;
        let particles = [];
        
        function resize() {
            width = canvas.width = parentWindow.innerWidth;
            height = canvas.height = parentWindow.innerHeight;
        }
        
        parentWindow.addEventListener('resize', resize);
        resize();
        
        let mouse = { x: null, y: null, radius: 150 };
        
        parentWindow.addEventListener('mousemove', function(event) {
            mouse.x = event.clientX;
            mouse.y = event.clientY;
        });
        
        parentWindow.addEventListener('mouseout', function() {
            mouse.x = null;
            mouse.y = null;
        });
        
        class Particle {
            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.size = Math.random() * 2 + 0.5;
                this.speedX = Math.random() * 0.6 - 0.3;
                this.speedY = Math.random() * 0.6 - 0.3;
                this.color = 'rgba(59, 130, 246, 0.4)';
            }
            update() {
                this.x += this.speedX;
                this.y += this.speedY;
                
                if (this.size > 0.2) this.size -= 0.01;
                
                if (this.x < 0 || this.x > width) this.speedX = -this.speedX;
                if (this.y < 0 || this.y > height) this.speedY = -this.speedY;
                
                // Cursor interaction
                if (mouse.x != null) {
                    let dx = mouse.x - this.x;
                    let dy = mouse.y - this.y;
                    let distance = Math.sqrt(dx*dx + dy*dy);
                    if (distance < mouse.radius) {
                        const forceDirectionX = dx / distance;
                        const forceDirectionY = dy / distance;
                        const force = (mouse.radius - distance) / mouse.radius;
                        const directionX = forceDirectionX * force * 5;
                        const directionY = forceDirectionY * force * 5;
                        this.x -= directionX;
                        this.y -= directionY;
                    }
                }
            }
            draw() {
                ctx.fillStyle = this.color;
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        
        function init() {
            particles = [];
            let numParticles = (width * height) / 5500;
            for(let i=0; i<numParticles; i++) {
                particles.push(new Particle());
            }
        }
        
        function animate() {
            ctx.clearRect(0, 0, width, height);
            for(let i=0; i<particles.length; i++) {
                particles[i].update();
                particles[i].draw();
                
                for(let j=i; j<particles.length; j++) {
                    let dx = particles[i].x - particles[j].x;
                    let dy = particles[i].y - particles[j].y;
                    let distance = Math.sqrt(dx*dx + dy*dy);
                    
                    if (distance < 130) {
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(59, 130, 246, ${0.25 - distance/520})`;
                        ctx.lineWidth = 0.8;
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                        ctx.closePath();
                    }
                }
            }
            requestAnimationFrame(animate);
        }
        
        init();
        animate();
    }
</script>
""", height=0, width=0)

# --- HELPER UI COMPONENTS ---
def render_kpi_card(title, value, subtitle="", highlight=False):
    style = "border-color: #3B82F6;" if highlight else ""
    val_color = "#FFFFFF"
    
    if "%" in str(value) and not "-" in str(value) and not "nan" in str(value).lower() and title != "Win Rate":
        if float(str(value).replace("%", "")) > 0: val_color = "#22C55E"
    if "-" in str(value): val_color = "#EF4444"
        
    st.markdown(f"""
    <div class="quant-card" style="{style}">
        <p style="margin:0; font-size: 0.85rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px;">{title}</p>
        <h2 style="margin: 5px 0; font-size: 1.8rem; color: {val_color}; font-family: 'IBM Plex Sans'; font-weight: 500;">{value}</h2>
        <p style="margin:0; font-size: 0.75rem; color: #6B7280;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def generate_ai_commentary(ticker, signal, conf, sentiment, feat_imp, metrics, df, regime_info):
    """
    Generate a human-readable trading summary based on all system outputs.
    Formatted strictly for professional UI.
    """
    top_feat = feat_imp.index[0]
    top_val = df[top_feat].iloc[-1]
    
    reg_name, _ = regime_info
    
    # 1. Consensus & The "Edge"
    edge = f"Calibrated Ensemble indicates {conf:.1f}% probability in a {reg_name} regime. "
    if conf >= 75:
        edge += "Statistical advantage threshold met. Historical mapping validates high-conviction probability."
    elif conf >= 65:
        edge += "Acceptable statistical edge detected, supporting intended trajectory."
    else:
        edge += "Internal variance detected. Caution advised due to probability dilution."

    # 2. The Catalyst (Sentiment)
    if sentiment > 0.5:
        catalyst = f"Positive drift detected ({sentiment:.2f} relative magnitude). Macro fundamental tailwinds present."
    elif sentiment < -0.5:
        catalyst = f"Negative drift detected ({sentiment:.2f} relative magnitude). Macro headwinds align with downside targets."
    else:
        catalyst = f"Neutral drift ({sentiment:.2f} magnitude). Signal relies entirely on mathematical momentum."

    # 3. Technical Reasoning
    reasoning = f"Primary predictive vector identified as {top_feat} (Current State: {top_val:.4f}). "
    if "RSI" in top_feat:
        reasoning += "Oscillator exhaustion factored into primary threshold logic."
    elif "BB" in top_feat or "SMA" in top_feat:
        reasoning += "Mean-reversion deviation exceeds normal distributional bounds."
    elif "Return" in top_feat:
        reasoning += "Velocity-based momentum continuation projected."
    else:
        reasoning += "Vector aligns with historically profitable structural setups."

    # 4. Market Guardrail (The Veto)
    mkt_ret = df['Index_Return'].iloc[-1]
    total_trades = metrics.get('Total Trades', 0)
    
    if total_trades < 15:
        veto = f"⚠️ STATISTICAL VETO: Only {int(total_trades)} historical trades found. This signal is rejected due to insufficient proof of edge."
    elif mkt_ret < -0.015:
        veto = f"⚠️ MARKET CRASH VETO: {reg_name} is selling off aggressively ({mkt_ret*100:.1f}%). All BUY signals are high-risk. Capital preservation is priority."
    elif mkt_ret < -0.005:
        veto = f"⚠️ MARKET WEAKNESS: Benchmarks are down. Sector correlation might drag this asset lower regardless of conviction."
    else:
        veto = f"✅ MARKET SYNC: Global benchmarks are stable. This signal is mathematically unencumbered by macro noise."

    return {
        "edge": edge,
        "catalyst": catalyst,
        "reasoning": reasoning,
        "veto": veto
    }

def get_market_status():
    utcnow = datetime.datetime.now(pytz.utc)
    day = utcnow.weekday()
    
    # India Market Hours (NSE: 9:15 AM to 3:30 PM IST)
    ist_tz = pytz.timezone('Asia/Kolkata')
    ist_now = utcnow.astimezone(ist_tz)
    ind_open = (ist_now.hour > 9 or (ist_now.hour == 9 and ist_now.minute >= 15)) and \
               (ist_now.hour < 15 or (ist_now.hour == 15 and ist_now.minute < 30))
    ind_status = "OPEN" if ind_open and day < 5 else "CLOSED"
    ind_color = "#22C55E" if ind_status == "OPEN" else "#EF4444"
    
    # US Market Hours (NYSE: 9:30 AM to 4:00 PM EST)
    est_tz = pytz.timezone('US/Eastern')
    est_now = utcnow.astimezone(est_tz)
    us_open = (est_now.hour > 9 or (est_now.hour == 9 and est_now.minute >= 30)) and \
              (est_now.hour < 16)
    us_status = "OPEN" if us_open and day < 5 else "CLOSED"
    us_color = "#22C55E" if us_status == "OPEN" else "#EF4444"
    
    return ind_status, ind_color, us_status, us_color

ind_st, ind_col, us_st, us_col = get_market_status()

# --- MAIN APP ---
col_head1, col_head2 = st.columns([4, 1])
with col_head1:
    st.markdown("<h1 style='color: #E5E5E5; margin-bottom: 0px;'>QUANT<span style='color: #3B82F6;'>AI</span> TRADER</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #9CA3AF; font-size: 0.9rem; margin-top: -5px;'>INSTITUTIONAL PREDICTIVE TERMINAL</p>", unsafe_allow_html=True)
with col_head2:
    st.markdown(f"""
    <div style="text-align: right; color: #9CA3AF; font-size: 0.75rem; font-family: monospace; margin-top: 5px;">
        US MARKET: <span style="color: {us_col}; font-weight: bold;">{us_st}</span><br>
        IND MARKET: <span style="color: {ind_col}; font-weight: bold;">{ind_st}</span><br>
        SYSTEM: <span style="color: #22C55E; font-weight: bold;">CALIBRATED</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin-top: 0px;'>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<h3 style='color: #E5E5E5; margin-top: 0px;'>TERMINAL CONTROLS</h3>", unsafe_allow_html=True)
ticker = st.sidebar.text_input("ASSET TICKER", "RELIANCE.NS")

period_map = {
    "Auto-Optimize (Best Sharpe)": "auto",
    "Max Available Data": None,
    "Last 5 Years": 5.0,
    "Last 3 Years": 3.0,
    "Last 2 Years": 2.0,
    "Last 1 Year": 1.0,
    "Last 6 Months": 0.5,
    "Last 3 Months": 0.25
}
selected_period = st.sidebar.selectbox("TRAINING HORIZON", list(period_map.keys()), index=0)
lookback = period_map[selected_period]

run_btn = st.sidebar.button("EXECUTE PIPELINE")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.8rem; color: #6B7280; font-family: monospace;">
    <strong>PIPELINE TOPOLOGY:</strong><br><br>
    [1] Data Ingestion<br>
    [2] NLP Sentiment Parsing<br>
    [3] Mathematical Feature Gen<br>
    [4] Ensemble Training<br>
    [5] Probability Calibration<br>
    [6] VectorBT Simulation
</div>
""", unsafe_allow_html=True)

if run_btn:
    with st.spinner(f"INITIALIZING PIPELINE FOR {ticker}... [AUTO-OPT: {lookback == 'auto'}]"):
        best_sharpe = -float('inf')
        best_res = None
        best_metrics = None
        best_oos_preds = None
        selected_lookback_str = ""

        if lookback == "auto":
            periods_to_test = [
                ("Last 3 Years", 3.0), 
                ("Last 2 Years", 2.0), 
                ("Last 1 Year", 1.0), 
                ("Last 6 Months", 0.5)
            ]
        else:
            periods_to_test = [(selected_period, lookback)]

        for p_name, p_val in periods_to_test:
            try:
                res = run_training_pipeline(ticker, lookback_years=p_val)
                if res is None:
                    continue
                
                models_temp, df_temp, feat_imp_temp, oos_signals_temp, scaler_temp, pca_temp, trained_features_temp, regime_detector_temp = res
                
                if "Ensemble" in oos_signals_temp:
                    oos_preds_temp = oos_signals_temp["Ensemble"]
                else:
                    oos_preds_temp = oos_signals_temp[next(iter(oos_signals_temp))]
                    
                oos_df_temp = df_temp.loc[oos_preds_temp.index]
                pf_temp, metrics_temp = run_backtest(oos_df_temp, oos_preds_temp)
                
                sharpe = metrics_temp.get('Sharpe Ratio', -float('inf'))
                win_rate = metrics_temp.get('Win Rate (%)', 0.0)
                trades = metrics_temp.get('Total Trades', 0)
                
                if pd.isna(sharpe) or np.isinf(sharpe) or trades == 0:
                    sharpe = -100.0
                if pd.isna(win_rate) or np.isinf(win_rate):
                    win_rate = 0.0
                    
                # STATISTICAL GUARDRAIL: Penalize low sample size
                if trades < 15:
                    sharpe = sharpe * 0.1 # Heavily de-weight unreliable results
                
                if (sharpe > best_sharpe) or (sharpe == best_sharpe and win_rate > best_metrics.get('Win Rate (%)', 0.0)):
                    best_sharpe = sharpe
                    best_res = res
                    best_metrics = metrics_temp
                    best_oos_preds = oos_preds_temp
                    best_sharpe_raw = metrics_temp.get('Sharpe Ratio', 0.0)
                    selected_lookback_str = p_name
                    
            except Exception as e:
                continue

        if best_res is None:
            st.error("SYSTEM HALTED: Execution failed. Verify ticker liquidity and data availability.")
            res = None
        else:
            res = best_res

    if res is not None:
        models, df, feat_imp, oos_signals, scaler, pca, trained_features_raw, regime_detector = res
        
        # Identify benchmark for UI
        benchmark_ticker = "^NSEI" if ticker.upper().endswith(".NS") else "^GSPC"
        benchmark_name = "Nifty 50" if benchmark_ticker == "^NSEI" else "S&P 500"
        
        oos_preds = best_oos_preds
        metrics = best_metrics
        
        # Determine signals
        # Override historical close with true live spot price if available
        try:
            import yfinance as yf
            live_ticker = yf.Ticker(ticker)
            t_info = live_ticker.info
            latest_price = t_info.get('regularMarketPrice', live_ticker.fast_info['lastPrice'])
            prev_price = t_info.get('previousClose', live_ticker.fast_info['previousClose'])
            # Fallbacks just in case
            if pd.isna(latest_price): latest_price = df['Close'].iloc[-1]
            if pd.isna(prev_price): prev_price = df['Close'].iloc[-2]
        except Exception:
            latest_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            
        change = ((latest_price - prev_price) / prev_price) * 100
        
        X = df[trained_features_raw].ffill().fillna(0)
        recent_sentiment = df['avg_sentiment'].iloc[-1]
        
        latest_features_raw = X.iloc[-1:]
        # Apply Spectral Denoising (Scaling + PCA) to live features
        latest_features_scaled = scaler.transform(latest_features_raw)
        latest_features = pca.transform(latest_features_scaled)
        
        probas = []
        for name, m in models.items():
            try:
                # Sklearn objects expect 2D array, which pca.transform already returns
                prob = m.predict_proba(latest_features)[0][1]
            except:
                prob = m.predict(latest_features)[0]
            probas.append(prob)
        
        # --- SIGNAL VETO LOGIC ---
        avg_prob = sum(probas) / len(probas)
        total_trades = metrics.get('Total Trades', 0)
        
        if total_trades < 15:
            signal_txt = "HOLD"
            signal_cls = "signal-HOLD"
            signal_sub = "INSUFFICIENT DATA"
            conf = avg_prob * 100
        elif avg_prob >= 0.59:
            signal_txt = "BUY"
            signal_cls = "signal-BUY"
            signal_sub = "STRONG CONVICTION"
            conf = avg_prob * 100
        elif avg_prob > 0.50:
            signal_txt = "HOLD/BUY"
            signal_cls = "signal-HOLD"
            signal_sub = "WEAK CONVICTION"
            conf = avg_prob * 100
        elif avg_prob <= 0.41:
            signal_txt = "SELL"
            signal_cls = "signal-SELL"
            signal_sub = "STRONG CONVICTION"
            conf = (1.0 - avg_prob) * 100
        else:
            signal_txt = "HOLD/SELL"
            signal_cls = "signal-HOLD"
            signal_sub = "WEAK CONVICTION"
            conf = (1.0 - avg_prob) * 100

        # Regime Detection
        reg_name, reg_color = regime_detector.predict(df)

        # Optimization Note
        if lookback == "auto":
            st.markdown(f"""
            <div style="background: rgba(34, 197, 94, 0.1); border-left: 3px solid #22C55E; padding: 10px; margin-bottom: 20px; font-size: 0.85rem; color: #A7F3D0;">
                OVERRIDE: Auto-Optimizer selected <strong>{selected_lookback_str}</strong> horizon. Risk-Adjusted Return (Sharpe: {best_sharpe_raw:.2f}).
            </div>
            """, unsafe_allow_html=True)

        # --- GRID LAYOUT ---
        # SECTION A & E: Signal and Metrics (Unified Card)
        display_win = metrics.get('Win Rate (%)', 0.0)
        display_win_str = "0.00%" if pd.isna(display_win) else f"{display_win:.2f}%"
        s_yield = metrics.get('Strategy Total Return (%)', 0.0)
        s_ratio = metrics.get('Sharpe Ratio', 0.0)
        
        # Color logic
        c_price = "#22C55E" if change > 0 else "#EF4444"
        c_yield = "#22C55E" if s_yield > 0 else "#EF4444" if s_yield < 0 else "#FFFFFF"
        c_ratio = "#22C55E" if s_ratio > 0 else "#EF4444" if s_ratio < 0 else "#FFFFFF"
        
        metrics_html = f"""
        <div class="unified-metrics-card">
            <div class="metrics-row" style="border-bottom: 1px solid #1A1A1A; padding-bottom: 20px;">
                <div class="metric-segment signal-area">
                    <p style="margin:0; font-size: 0.75rem; color: #9CA3AF; letter-spacing: 1px; text-transform: uppercase;">Algorithmic Signal</p>
                    <h1 class="{signal_cls}" style="margin: 5px 0; font-size: 2.2rem; font-weight: 600; line-height: 1;">{signal_txt}</h1>
                    <p style="margin:0; font-size: 0.8rem; color: #D1D5DB;">{signal_sub} • {conf:.1f}% PROBABILITY</p>
                </div>
                <div class="metric-segment">
                    <p style="margin:0; font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase;">Asset Price</p>
                    <h2 style="margin: 5px 0; font-size: 1.6rem; color: {c_price}; font-family: 'IBM Plex Sans';">{latest_price:.2f}</h2>
                    <p style="margin:0; font-size: 0.65rem; color: #6B7280;">{change:+.2f}% Daily</p>
                </div>
                <div class="metric-segment">
                    <p style="margin:0; font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase;">Win Rate</p>
                    <h2 style="margin: 5px 0; font-size: 1.6rem; color: #FFFFFF; font-family: 'IBM Plex Sans';">{display_win_str}</h2>
                    <p style="margin:0; font-size: 0.65rem; color: #6B7280;">{int(metrics.get('Total Trades', 0))} Trades</p>
                </div>
                <div class="metric-segment" style="border-right: none;">
                    <p style="margin:0; font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase;">Strategy Yield</p>
                    <h2 style="margin: 5px 0; font-size: 1.6rem; color: {c_yield}; font-family: 'IBM Plex Sans';">{s_yield:.2f}%</h2>
                    <p style="margin:0; font-size: 0.65rem; color: #6B7280;">B&H: {metrics.get('Benchmark Total Return (%)', 0.0):.2f}%</p>
                </div>
            </div>
            
            <div class="metrics-row">
                <div class="metric-segment" style="text-align: left; flex: 1.5;">
                    <p style="margin:0; font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase;">Sharpe Ratio</p>
                    <h2 style="margin: 5px 0; font-size: 1.6rem; color: {c_ratio}; font-family: 'IBM Plex Sans';">{s_ratio:.2f}</h2>
                    <p style="margin:0; font-size: 0.65rem; color: #6B7280;">{"⚠️ LOW SAMPLE" if metrics.get('Total Trades', 0) < 15 else "Risk-Adjusted"}</p>
                </div>
                <div class="metric-segment" style="flex: 1.5;">
                    <p style="margin:0; font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase;">{benchmark_name} Health</p>
                    <h2 style="margin: 5px 0; font-size: 1.4rem; color: #FFFFFF; font-family: 'IBM Plex Sans'; font-weight: 600;">
                        {df['Index_Return'].iloc[-1]*100:+.2f}%
                    </h2>
                    <p style="margin:0; font-size: 0.65rem; color: #6B7280;">Market Guardrail</p>
                </div>
                <div class="metric-segment" style="flex: 1.5; border-right: none; text-align: right;">
                    <p style="margin:0; font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase;">Market Regime</p>
                    <h2 style="margin: 5px 0; font-size: 1.4rem; color: {reg_color}; font-family: 'IBM Plex Sans'; font-weight: 600;">{reg_name}</h2>
                    <p style="margin:0; font-size: 0.65rem; color: #6B7280;">Factor Classification</p>
                </div>
            </div>
        </div>
        """
        st.markdown(metrics_html.replace('\n', ' '), unsafe_allow_html=True)

        # SECTION B: Multi-Chart Plotly Override
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #E5E5E5; margin-bottom: 10px;'>PRICE ACTION TERMINAL</h4>", unsafe_allow_html=True)
        
        plot_df = df.iloc[-200:]
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=plot_df.index,
            open=plot_df['Open'], high=plot_df['High'],
            low=plot_df['Low'], close=plot_df['Close'],
            name='Price Structure',
            increasing_line_color='#22C55E', decreasing_line_color='#EF4444'
        ))
        
        if 'SMA_20' in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_20'], line=dict(color='#3B82F6', width=1), name='SMA 20'))
        if 'BB_High' in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_High'], line=dict(color='#4B5563', dash='dot', width=1), name='BB High'))
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Low'], line=dict(color='#4B5563', dash='dot', width=1), name='BB Low'))

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            height=450,
            xaxis=dict(showgrid=False, zeroline=False, rangeslider=dict(visible=False)),
            yaxis=dict(showgrid=True, gridcolor='#1A1A1A', zeroline=False)
        )
        st.plotly_chart(fig, use_container_width=True)

        # SECTION C & D: Intelligence & Regime
        col_c, col_d = st.columns([2, 2])
        
        with col_c:
            st.markdown("<h4 style='color: #E5E5E5;'>MODEL FEATURE IMPORTANCE</h4>", unsafe_allow_html=True)
            top_feats = feat_imp.head(10).sort_values(ascending=True)
            
            fig_bar = go.Figure(go.Bar(
                x=top_feats.values,
                y=top_feats.index,
                orientation='h',
                marker=dict(color='#3B82F6', line=dict(color='#0B0B0B', width=1))
            ))
            fig_bar.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                height=250,
                xaxis=dict(showgrid=True, gridcolor='#1A1A1A', zeroline=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_d:
            st.markdown("<h4 style='color: #E5E5E5;'>STRUCTURAL INTELLIGENCE</h4>", unsafe_allow_html=True)
            report = generate_ai_commentary(ticker, signal_txt, conf, recent_sentiment, feat_imp, metrics, df, (reg_name, reg_color))
            
            volatility = df['Risk_Level'].iloc[-1] * 100
            
            st.markdown(f"""
            <div class="quant-card" style="font-size: 0.85rem; color: #D1D5DB; line-height: 1.6;">
                <strong style="color: #EF4444;">[00] MARKET GUARDRAIL:</strong> {report['veto']}<br>
                <strong style="color: #3B82F6;">[01] SYSTEMATIC VOLATILITY:</strong> {volatility:.2f}%<br>
                <strong style="color: #3B82F6;">[02] STATISTICAL EDGE:</strong> {report['edge']}<br>
                <strong style="color: #3B82F6;">[03] NLP DRIFT ANALYSIS:</strong> {report['catalyst']}<br>
                <strong style="color: #3B82F6;">[04] STRUCTURAL VECTOR:</strong> {report['reasoning']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h4 style='color: #E5E5E5;'>DATA TRANSPARENCY & VALIDATION LOG</h4>", unsafe_allow_html=True)
        
        col_t1, col_t2, col_t3 = st.columns(3)
        
        with col_t1:
            st.markdown("<span style='color:#9CA3AF; font-size:0.8rem; letter-spacing:1px; text-transform:uppercase;'>[01] Structural Vectors</span>", unsafe_allow_html=True)
            top_5_feats = feat_imp.head(5).index.tolist()
            text_html = "<div class='quant-card' style='font-family: monospace; font-size: 0.8rem; color:#E5E5E5;'>"
            for feat in top_5_feats:
                current_val = df[feat].iloc[-1]
                text_html += f"<div style='border-left: 2px solid #3B82F6; padding-left: 8px; margin-bottom: 5px;'><strong>{feat}</strong>: {current_val:.4f}</div>"
            text_html += "</div>"
            st.markdown(text_html, unsafe_allow_html=True)
            
        with col_t2:
            st.markdown("<span style='color:#9CA3AF; font-size:0.8rem; letter-spacing:1px; text-transform:uppercase;'>[02] Raw Execution Inputs</span>", unsafe_allow_html=True)
            st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5), use_container_width=True)
            
        with col_t3:
            st.markdown("<span style='color:#9CA3AF; font-size:0.8rem; letter-spacing:1px; text-transform:uppercase;'>[03] Live NLP Feed</span>", unsafe_allow_html=True)
            recent_news = fetch_google_news(ticker, max_results=5)
            news_html = "<div class='quant-card' style='font-family: Inter; font-size: 0.75rem; color:#D1D5DB;'>"
            if not recent_news.empty:
                for idx, row in recent_news.iterrows():
                    pub_str = str(row['published_at'])[:10]
                    news_html += f"<div style='margin-bottom:8px;'><span style='color:#3B82F6'>{pub_str}</span> <a href='{row['link']}' style='color:#E5E5E5; text-decoration:none;'>{row['title']}</a></div>"
            else:
                news_html += "NO RECENT DATA"
            news_html += "</div>"
            st.markdown(news_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; color: #4B5563; font-size: 0.7rem; font-family: monospace;">
            QUANT-AI PREDICTIVE TERMINAL V2.0 • ALGORITHMIC EXECUTION MODULE<br>
            STRICTLY NO FINANCIAL ADVICE
        </div>
        """, unsafe_allow_html=True)
