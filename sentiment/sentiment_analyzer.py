from transformers import pipeline
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

_ANALYZER = None

def get_sentiment_analyzer():
    """
    Load FinBERT pipeline for financial sentiment analysis lazily.
    Uses ProsusAI/finbert.
    """
    global _ANALYZER
    if _ANALYZER is None:
        print("⏳ Loading FinBERT Sentiment Analyzer... (This may take a moment)")
        # Using specific financial BERT
        _ANALYZER = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        print("✅ FinBERT loaded.")
    return _ANALYZER

def compute_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe of news from news_loader, calculate the daily sentiment.
    
    Args:
        news_df: DataFrame containing 'title' and 'published_at' columns.
        
    Returns:
        DataFrame aggregated by Date with 'avg_sentiment' and 'news_count'.
    """
    if news_df.empty or 'title' not in news_df.columns:
        print("⚠️ No news data to analyze.")
        # Return empty DF with expected columns
        return pd.DataFrame(columns=['Date', 'avg_sentiment', 'news_count'])
        
    print(f"🧠 Analyzing sentiment for {len(news_df)} articles...")
    analyzer = get_sentiment_analyzer()
    
    # We will score the titles
    titles = news_df['title'].astype(str).tolist()
    
    # Batch process
    try:
        results = analyzer(titles)
    except Exception as e:
        print(f"❌ Error in sentiment analysis: {e}")
        return pd.DataFrame(columns=['Date', 'avg_sentiment', 'news_count'])
        
    news_df['sentiment_label'] = [res['label'] for res in results]
    news_df['sentiment_score'] = [res['score'] for res in results]
    
    # Convert text label to numerical score
    # Positive = +1 * score, Negative = -1 * score, Neutral = 0
    def map_sentiment(row):
        lbl = row['sentiment_label']
        score = row['sentiment_score']
        if lbl == 'positive': return score
        elif lbl == 'negative': return -score
        else: return 0.0
        
    news_df['numeric_sentiment'] = news_df.apply(map_sentiment, axis=1)
    
    # Ensure datetime format for grouping
    news_df['Date'] = pd.to_datetime(news_df['published_at']).dt.date
    
    # Aggregate by day
    # We take the mean sentiment of all news on that day
    daily_sentiment = news_df.groupby('Date', as_index=False).agg(
        avg_sentiment=('numeric_sentiment', 'mean'),
        news_count=('title', 'count')
    )
    
    # Convert 'Date' to datetime for mapping later
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
    
    print("✅ Sentiment aggregation complete.")
    return daily_sentiment

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.news_loader import fetch_google_news
    
    news = fetch_google_news("Reliance Industries", max_results=10)
    if not news.empty:
        daily_sent = compute_daily_sentiment(news)
        print("\nDaily Sentiment Summary:")
        print(daily_sent)
