import os
import sys
import json
import time
import datetime
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import schedule
from requests.exceptions import RequestException, Timeout
from ratelimit import limits, sleep_and_retry

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config.config import FINNHUB_API_KEY, NEWS_API_KEY, TWITTER_API_KEY, TWITTER_API_SECRET
from config.config import SENTIMENT_CACHE_DIR
from src.sentiment.sentiment_analyzer import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeSentimentFeed:
    def __init__(self, cache_dir: Optional[str] = None, max_workers: int = 4):
        self.cache_dir = cache_dir or SENTIMENT_CACHE_DIR
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.sentiment_analyzer = SentimentAnalyzer()
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.feed_queue = queue.Queue()
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        self.finnhub_api_key = FINNHUB_API_KEY
        self.news_api_key = NEWS_API_KEY
        self.twitter_api_key = TWITTER_API_KEY
        self.twitter_api_secret = TWITTER_API_SECRET
        
        self.scheduled_jobs = []
    
    def start(self):
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Real-time sentiment feed started")
    
    def stop(self):
        self.stop_event.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=False)
        
        for job in self.scheduled_jobs:
            schedule.cancel_job(job)
        
        logger.info("Real-time sentiment feed stopped")
    
    def schedule_data_fetching(self, symbols: List[str], interval_minutes: int = 60):
        for symbol in symbols:
            job = schedule.every(interval_minutes).minutes.do(
                self._fetch_all_sources_for_symbol, symbol
            )
            self.scheduled_jobs.append(job)
            
        threading.Thread(target=self._run_scheduler, daemon=True).start()
        
        logger.info(f"Scheduled data fetching for {len(symbols)} symbols every {interval_minutes} minutes")
    
    def _run_scheduler(self):
        while not self.stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)
    
    def _process_queue(self):
        while not self.stop_event.is_set():
            try:
                item = self.feed_queue.get(timeout=1.0)
                if item is None:
                    continue
                    
                source, data = item
                self._process_sentiment_data(source, data)
                self.feed_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing queue item: {str(e)}")
    
    def _process_sentiment_data(self, source: str, data: Dict):
        try:
            symbol = data.get('symbol', 'MARKET')
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            
            cache_file = os.path.join(
                self.cache_dir, 
                f"{symbol}_{source}_{timestamp}.json"
            )
            
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            logger.debug(f"Saved {source} data for {symbol} to {cache_file}")
        except Exception as e:
            logger.error(f"Error processing sentiment data: {str(e)}")
    
    def fetch_data_for_symbols(self, symbols: List[str]):
        for symbol in symbols:
            self._fetch_all_sources_for_symbol(symbol)
    
    def _fetch_all_sources_for_symbol(self, symbol: str):
        try:
            self.executor.submit(self._fetch_finnhub_news, symbol)
            self.executor.submit(self._fetch_news_api, symbol)
            self.executor.submit(self._fetch_twitter_data, symbol)
            
            logger.info(f"Fetched data from all sources for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
    
    @sleep_and_retry
    @limits(calls=60, period=60)
    def _fetch_finnhub_news(self, symbol: str):
        if not self.finnhub_api_key:
            logger.warning("Finnhub API key not set, skipping finnhub news")
            return
        
        try:
            today = datetime.datetime.now()
            week_ago = today - datetime.timedelta(days=7)
            
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': week_ago.strftime('%Y-%m-%d'),
                'to': today.strftime('%Y-%m-%d'),
                'token': self.finnhub_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            news_data = response.json()
            
            if isinstance(news_data, list) and len(news_data) > 0:
                articles = []
                
                for article in news_data[:20]:
                    articles.append({
                        'title': article.get('headline', ''),
                        'summary': article.get('summary', ''),
                        'source': article.get('source', 'Finnhub'),
                        'url': article.get('url', ''),
                        'published_at': article.get('datetime', 0)
                    })
                
                # Process sentiment
                texts = [f"{a['title']}. {a['summary']}" for a in articles]
                sentiment_results = self.sentiment_analyzer.analyze_batch(texts)
                
                for i, sentiment in enumerate(sentiment_results):
                    articles[i]['sentiment'] = sentiment
                
                data = {
                    'symbol': symbol,
                    'source': 'finnhub',
                    'timestamp': datetime.datetime.now().timestamp(),
                    'articles': articles
                }
                
                self.feed_queue.put(('finnhub', data))
                
                logger.info(f"Fetched {len(articles)} news articles from Finnhub for {symbol}")
            else:
                logger.info(f"No news found from Finnhub for {symbol}")
                
        except RequestException as e:
            logger.error(f"Error fetching Finnhub news for {symbol}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing Finnhub news for {symbol}: {str(e)}")
    
    @sleep_and_retry
    @limits(calls=100, period=86400)
    def _fetch_news_api(self, symbol: str):
        if not self.news_api_key:
            logger.warning("News API key not set, skipping news API")
            return
        
        try:
            today = datetime.datetime.now()
            week_ago = today - datetime.timedelta(days=7)
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{symbol} stock",
                'from': week_ago.strftime('%Y-%m-%d'),
                'to': today.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            news_data = response.json()
            
            if news_data.get('status') == 'ok' and news_data.get('articles'):
                articles = []
                
                for article in news_data['articles'][:20]:  # Process up to 20 most recent news
                    articles.append({
                        'title': article.get('title', ''),
                        'summary': article.get('description', ''),
                        'source': article.get('source', {}).get('name', 'NewsAPI'),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', '')
                    })
                
                # Process sentiment
                texts = [f"{a['title']}. {a['summary']}" for a in articles]
                sentiment_results = self.sentiment_analyzer.analyze_batch(texts)
                
                for i, sentiment in enumerate(sentiment_results):
                    articles[i]['sentiment'] = sentiment
                
                data = {
                    'symbol': symbol,
                    'source': 'newsapi',
                    'timestamp': datetime.datetime.now().timestamp(),
                    'articles': articles
                }
                
                # Add to processing queue
                self.feed_queue.put(('newsapi', data))
                
                logger.info(f"Fetched {len(articles)} news articles from NewsAPI for {symbol}")
            else:
                logger.info(f"No news found from NewsAPI for {symbol}")
                
        except RequestException as e:
            logger.error(f"Error fetching NewsAPI data for {symbol}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing NewsAPI data for {symbol}: {str(e)}")
    
    def _fetch_twitter_data(self, symbol: str):
        if not self.twitter_api_key or not self.twitter_api_secret:
            logger.warning("Twitter API keys not set, skipping twitter data")
            return
        
        try:
            # Note: Full Twitter API integration would require OAuth2 flow and elevated access
            # This is a simplified version that would need to be expanded with actual Twitter API v2
            
            # Simulate tweet data for demonstration purposes
            tweets = self._get_simulated_tweets(symbol, 20)
            
            # Process sentiment
            texts = [tweet['text'] for tweet in tweets]
            sentiment_results = self.sentiment_analyzer.analyze_batch(texts)
            
            for i, sentiment in enumerate(sentiment_results):
                tweets[i]['sentiment'] = sentiment
            
            data = {
                'symbol': symbol,
                'source': 'twitter',
                'timestamp': datetime.datetime.now().timestamp(),
                'tweets': tweets
            }
            
            # Add to processing queue
            self.feed_queue.put(('twitter', data))
            
            logger.info(f"Processed {len(tweets)} tweets for {symbol}")
                
        except Exception as e:
            logger.error(f"Error processing Twitter data for {symbol}: {str(e)}")
    
    def _get_simulated_tweets(self, symbol: str, count: int = 20) -> List[Dict]:
        """Generate simulated tweet data for demonstration purposes"""
        tweets = []
        
        base_phrases = [
            f"Just bought some ${symbol} stock! Feeling optimistic about their growth potential.",
            f"${symbol} earnings report looks strong. Q3 numbers exceeded expectations.",
            f"Not sure about ${symbol}, their latest product launch seems underwhelming.",
            f"Analyst downgrade for ${symbol} today. Might be time to sell.",
            f"${symbol} is my top pick for 2025. Their new CEO is making great changes.",
            f"Market volatility affecting ${symbol} today. Holding for long term though.",
            f"${symbol} dividend announcement just came out. Solid increase!",
            f"Concerns about ${symbol}'s debt levels after their recent acquisition.",
            f"Technical analysis shows ${symbol} breaking through resistance levels.",
            f"${symbol} making strategic moves in the AI space. Very bullish!",
        ]
        
        now = datetime.datetime.now()
        
        for i in range(count):
            hours_ago = np.random.randint(0, 24 * 7)  # Random time within past week
            timestamp = now - datetime.timedelta(hours=hours_ago)
            
            base_text = np.random.choice(base_phrases)
            
            # Add some random variation to make tweets different
            variation = np.random.choice([
                "I think ", 
                "IMHO ", 
                "My analysis: ", 
                "Hot take: ", 
                "Just saying: ",
                ""
            ])
            
            suffix = np.random.choice([
                " #investing", 
                " #stocks", 
                " #finance", 
                " #trading", 
                " #markets",
                ""
            ])
            
            tweet = {
                'id': f"sim_{i}_{int(timestamp.timestamp())}",
                'text': f"{variation}{base_text}{suffix}",
                'created_at': timestamp.isoformat(),
                'user': {
                    'username': f"trader_{np.random.randint(1000, 9999)}",
                    'followers_count': np.random.randint(100, 10000)
                },
                'retweet_count': np.random.randint(0, 50),
                'like_count': np.random.randint(0, 100)
            }
            
            tweets.append(tweet)
        
        return tweets
    
    def aggregate_sentiment_data(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Aggregate sentiment data from all sources for a given symbol over a time period"""
        try:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(days=days)
            cutoff_timestamp = cutoff_time.timestamp()
            
            # Find all relevant cached files
            all_files = []
            for filename in os.listdir(self.cache_dir):
                if filename.startswith(f"{symbol}_") and filename.endswith(".json"):
                    all_files.append(os.path.join(self.cache_dir, filename))
            
            # Read and process each file
            sentiment_data = []
            
            for file_path in all_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    file_timestamp = data.get('timestamp', 0)
                    source = data.get('source', 'unknown')
                    
                    if file_timestamp < cutoff_timestamp:
                        continue
                    
                    if source == 'twitter' and 'tweets' in data:
                        for tweet in data['tweets']:
                            if 'sentiment' in tweet:
                                sentiment_data.append({
                                    'symbol': symbol,
                                    'source': 'twitter',
                                    'date': datetime.datetime.fromisoformat(tweet['created_at']).date(),
                                    'text': tweet['text'],
                                    'compound': tweet['sentiment'].get('compound', 0),
                                    'positive': tweet['sentiment'].get('positive', 0),
                                    'negative': tweet['sentiment'].get('negative', 0),
                                    'neutral': tweet['sentiment'].get('neutral', 0),
                                    'followers': tweet['user'].get('followers_count', 0),
                                    'engagement': tweet.get('retweet_count', 0) + tweet.get('like_count', 0)
                                })
                    elif (source == 'finnhub' or source == 'newsapi') and 'articles' in data:
                        for article in data['articles']:
                            if 'sentiment' in article:
                                if source == 'finnhub' and 'published_at' in article:
                                    date = datetime.datetime.fromtimestamp(article['published_at']).date()
                                else:
                                    date = datetime.datetime.fromisoformat(article.get('published_at', 
                                                                                        datetime.datetime.now().isoformat())).date()
                                
                                sentiment_data.append({
                                    'symbol': symbol,
                                    'source': source,
                                    'date': date,
                                    'text': f"{article.get('title', '')}. {article.get('summary', '')}",
                                    'compound': article['sentiment'].get('compound', 0),
                                    'positive': article['sentiment'].get('positive', 0),
                                    'negative': article['sentiment'].get('negative', 0),
                                    'neutral': article['sentiment'].get('neutral', 0),
                                    'source_name': article.get('source', ''),
                                    'url': article.get('url', '')
                                })
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    continue
            
            if not sentiment_data:
                logger.info(f"No sentiment data found for {symbol} in the past {days} days")
                return pd.DataFrame()
            
            df = pd.DataFrame(sentiment_data)
            
            # Add weighting based on source importance
            df['weight'] = 1.0
            
            # Give more weight to tweets with more followers
            if 'followers' in df.columns:
                follower_mask = df['source'] == 'twitter'
                if follower_mask.any():
                    max_followers = df.loc[follower_mask, 'followers'].max()
                    if max_followers > 0:
                        df.loc[follower_mask, 'weight'] = 0.5 + 0.5 * (df.loc[follower_mask, 'followers'] / max_followers)
            
            # Give more weight to more recent data
            df['date'] = pd.to_datetime(df['date'])
            latest_date = df['date'].max()
            df['days_old'] = (latest_date - df['date']).dt.days
            max_days = df['days_old'].max()
            
            if max_days > 0:
                df['recency_weight'] = 1 - (df['days_old'] / (max_days + 1))
                df['weight'] = df['weight'] * df['recency_weight']
            
            # Calculate weighted sentiment
            df['weighted_compound'] = df['compound'] * df['weight']
            
            # Aggregate by date
            daily_sentiment = df.groupby(['symbol', 'date']).agg({
                'weighted_compound': 'mean',
                'compound': 'mean',
                'positive': 'mean',
                'negative': 'mean',
                'neutral': 'mean',
                'weight': 'mean',
                'text': 'count'
            }).reset_index()
            
            daily_sentiment = daily_sentiment.rename(columns={'text': 'mention_count'})
            
            return daily_sentiment
            
        except Exception as e:
            logger.error(f"Error aggregating sentiment data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """Get the latest sentiment data for a given symbol"""
        sentiment_df = self.aggregate_sentiment_data(symbol, days)
        
        if sentiment_df.empty:
            return {
                'symbol': symbol,
                'latest_compound': 0,
                'latest_positive': 0,
                'latest_negative': 0,
                'latest_neutral': 0,
                'avg_compound': 0,
                'data_points': 0,
                'available': False
            }
        
        latest = sentiment_df.sort_values('date', ascending=False).iloc[0]
        
        return {
            'symbol': symbol,
            'latest_compound': latest['compound'],
            'latest_positive': latest['positive'],
            'latest_negative': latest['negative'],
            'latest_neutral': latest['neutral'],
            'latest_date': latest['date'],
            'avg_compound': sentiment_df['compound'].mean(),
            'data_points': sentiment_df['mention_count'].sum(),
            'available': True
        }
