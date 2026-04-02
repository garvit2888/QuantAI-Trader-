import urllib.parse
import feedparser
import pandas as pd
from datetime import datetime
import re

# Maps common ticker formats to human-readable search terms for Google News.
_TICKER_TO_NAME = {
    # Commodities (Yahoo Finance futures format)
    "GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil", "NG=F": "Natural Gas",
    "HG=F": "Copper", "ZC=F": "Corn", "ZW=F": "Wheat", "ZS=F": "Soybeans",
    "PL=F": "Platinum", "PA=F": "Palladium", "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
    # US Indices
    "^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "Nasdaq", "^RUT": "Russell 2000",
    # Common US Stocks
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google Alphabet",
    "AMZN": "Amazon", "TSLA": "Tesla", "META": "Meta Facebook",
    "NVDA": "Nvidia", "NFLX": "Netflix", "AMD": "AMD",
    # Indian Stocks
    "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys", "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank",
    "WIPRO.NS": "Wipro", "SBIN.NS": "State Bank of India",
    "TATAMOTORS.NS": "Tata Motors", "BAJFINANCE.NS": "Bajaj Finance",
    "ADANIENT.NS": "Adani Enterprises",
}

def resolve_ticker_to_name(ticker: str) -> str:
    """Convert a ticker symbol to a human-readable name for news search."""
    # Direct lookup first
    if ticker in _TICKER_TO_NAME:
        return _TICKER_TO_NAME[ticker]
    # Strip common suffixes like .NS, .BO, .L etc. for a cleaner fallback
    clean = re.sub(r'\.(NS|BO|L|TO|AX|PA|DE|HK)$', '', ticker, flags=re.IGNORECASE)
    clean = re.sub(r'[=\^\-].*', '', clean)  # strip =F, ^, -USD suffixes
    return clean

def fetch_google_news(query: str, max_results: int = 50) -> pd.DataFrame:
    """
    Fetch news from Google News RSS feed for a given query (e.g., 'AAPL stock' or 'Reliance Industries').
    
    Args:
        query (str): The search phrase to look for in news.
        max_results (int): Maximum number of articles to return.
        
    Returns:
        pd.DataFrame: A DataFrame containing 'title', 'link', 'published_at', and 'source'.
    """
    # Resolve ticker to human-readable name for better Google News results
    search_term = resolve_ticker_to_name(query)
    print(f"📰 Fetching news for: '{query}' → searching as '{search_term}'...")
    encoded_query = urllib.parse.quote_plus(f"{search_term} stock financial news")
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
