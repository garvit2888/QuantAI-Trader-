import urllib.parse
import feedparser
import pandas as pd
from datetime import datetime

def fetch_google_news(query: str, max_results: int = 50) -> pd.DataFrame:
    """
    Fetch news from Google News RSS feed for a given query (e.g., 'AAPL stock' or 'Reliance Industries').
    
    Args:
        query (str): The search phrase to look for in news.
        max_results (int): Maximum number of articles to return.
        
    Returns:
        pd.DataFrame: A DataFrame containing 'title', 'link', 'published_at', and 'source'.
    """
    print(f"📰 Fetching news for query: '{query}'...")
    encoded_query = urllib.parse.quote_plus(f"{query} stock financial news")
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    feed = feedparser.parse(rss_url)
    articles = []
    
    for entry in feed.entries[:max_results]:
        # Parse published date
        try:
            pub_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z")
        except ValueError:
            pub_date = pd.NaT
            
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published_at": pub_date,
            "source": entry.source.title if hasattr(entry, 'source') else "Unknown"
        })
        
    df = pd.DataFrame(articles)
    if not df.empty:
        df = df.dropna(subset=['published_at'])
        df = df.sort_values("published_at", ascending=False).reset_index(drop=True)
        print(f"✅ Successfully fetched {len(df)} news articles.")
    else:
        print(f"⚠️ No news found for query: {query}.")
        
    return df

if __name__ == "__main__":
    # Test news fetcher
    news_df = fetch_google_news("Reliance Industries", max_results=5)
    for _, row in news_df.iterrows():
        print(f"[{row['published_at']}] {row['title']}")
