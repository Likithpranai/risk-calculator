"""
Sentiment analyzer module for processing news and social media data.
Uses NLP techniques to extract market sentiment for risk prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import time
from datetime import datetime, timedelta
import re
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor

# Import configuration
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config.config import SENTIMENT_BATCH_SIZE, USE_PARALLEL, MAX_THREADS

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyzes market sentiment from news and social media feeds.
    Provides sentiment scores that can be integrated into risk models.
    """
    
    def __init__(self, 
                model_type: str = 'transformer',
                use_gpu: bool = False,
                batch_size: int = SENTIMENT_BATCH_SIZE):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            model_type: Type of model to use ('transformer', 'vader', 'textblob')
            use_gpu: Whether to use GPU acceleration if available
            batch_size: Batch size for processing
        """
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load the sentiment analysis model based on model_type."""
        start_time = time.time()
        
        try:
            if self.model_type == 'transformer':
                # Use Hugging Face transformers for sentiment analysis
                try:
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
                    
                    # Use FinBERT for financial sentiment analysis if available
                    model_name = "ProsusAI/finbert"
                    
                    # Only load tokenizer initially (lazy loading)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    
                    # Defer model loading until first use to save memory
                    logger.info(f"Initialized transformer tokenizer ({model_name})")
                    
                except ImportError:
                    logger.warning("Transformers library not available. Falling back to VADER")
                    self.model_type = 'vader'
                    
            if self.model_type == 'vader':
                # Use VADER (Valence Aware Dictionary and sEntiment Reasoner)
                try:
                    from nltk.sentiment.vader import SentimentIntensityAnalyzer
                    import nltk
                    
                    # Download VADER lexicon if not already downloaded
                    try:
                        nltk.data.find('sentiment/vader_lexicon.zip')
                    except LookupError:
                        nltk.download('vader_lexicon')
                        
                    self.model = SentimentIntensityAnalyzer()
                    logger.info("Initialized VADER sentiment analyzer")
                    
                except ImportError:
                    logger.warning("NLTK VADER not available. Falling back to TextBlob")
                    self.model_type = 'textblob'
                    
            if self.model_type == 'textblob':
                # Use TextBlob (simple but effective)
                try:
                    from textblob import TextBlob
                    # TextBlob is initialized on demand
                    logger.info("Will use TextBlob for sentiment analysis")
                except ImportError:
                    logger.error("No sentiment analysis libraries available")
                    
            logger.info(f"Sentiment model initialized in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment model: {e}")
            # Fall back to a simple dictionary-based approach if all else fails
            self.model_type = 'dictionary'
            
    def _ensure_model_loaded(self):
        """Ensure the model is loaded (for lazy loading)."""
        if self.model_type == 'transformer' and self.model is None and self.tokenizer is not None:
            try:
                from transformers import AutoModelForSequenceClassification, pipeline
                
                logger.info("Loading transformer model (first use)...")
                start_time = time.time()
                
                model_name = "ProsusAI/finbert"
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Set up device
                device = -1  # CPU
                if self.use_gpu:
                    import torch
                    if torch.cuda.is_available():
                        device = 0  # First GPU
                        
                self.model = pipeline(
                    "sentiment-analysis", 
                    model=model, 
                    tokenizer=self.tokenizer,
                    device=device
                )
                
                logger.info(f"Transformer model loaded in {time.time() - start_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error loading transformer model: {e}")
                self.model_type = 'vader'
                self._load_model()
                
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Clean the text
        clean_text = self._preprocess_text(text)
        
        if not clean_text:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
            
        try:
            if self.model_type == 'transformer':
                # Use Hugging Face transformer
                result = self.model(clean_text[:512])[0]  # Limit text length
                label = result['label'].lower()
                score = result['score']
                
                # Convert to common format
                sentiment = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
                sentiment[label] = score
                
                # Calculate compound score (-1 to 1)
                if label == 'positive':
                    compound = score
                elif label == 'negative':
                    compound = -score
                else:
                    compound = 0.0
                    
                sentiment['compound'] = compound
                
            elif self.model_type == 'vader':
                # Use VADER
                scores = self.model.polarity_scores(clean_text)
                sentiment = {
                    'positive': scores['pos'],
                    'negative': scores['neg'],
                    'neutral': scores['neu'],
                    'compound': scores['compound']
                }
                
            elif self.model_type == 'textblob':
                # Use TextBlob
                from textblob import TextBlob
                blob = TextBlob(clean_text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Convert to common format
                if polarity > 0:
                    positive = polarity
                    negative = 0.0
                else:
                    positive = 0.0
                    negative = abs(polarity)
                    
                neutral = 1.0 - subjectivity
                
                sentiment = {
                    'positive': positive,
                    'negative': negative,
                    'neutral': neutral,
                    'compound': polarity
                }
                
            else:
                # Dictionary-based approach
                # Simple keyword-based approach as fallback
                pos_words = ['up', 'rise', 'gain', 'positive', 'bull', 'bullish', 'growth', 'profit']
                neg_words = ['down', 'fall', 'drop', 'negative', 'bear', 'bearish', 'loss', 'decline']
                
                tokens = clean_text.lower().split()
                pos_count = sum(word in tokens for word in pos_words)
                neg_count = sum(word in tokens for word in neg_words)
                total = max(pos_count + neg_count, 1)  # Avoid division by zero
                
                sentiment = {
                    'positive': pos_count / total if total > 0 else 0.0,
                    'negative': neg_count / total if total > 0 else 0.0,
                    'neutral': 1.0 - (pos_count + neg_count) / len(tokens) if tokens else 1.0,
                    'compound': (pos_count - neg_count) / total if total > 0 else 0.0
                }
                
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
            
    def analyze_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts in an efficient way.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment scores
        """
        start_time = time.time()
        
        if not texts:
            return []
            
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Process in batches for better performance
        results = []
        
        if self.model_type == 'transformer' and USE_PARALLEL:
            # Process texts in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                clean_batch = [self._preprocess_text(text)[:512] for text in batch]  # Limit text length
                
                try:
                    batch_results = self.model(clean_batch)
                    
                    for result in batch_results:
                        label = result['label'].lower()
                        score = result['score']
                        
                        sentiment = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
                        sentiment[label] = score
                        
                        # Calculate compound score
                        if label == 'positive':
                            compound = score
                        elif label == 'negative':
                            compound = -score
                        else:
                            compound = 0.0
                            
                        sentiment['compound'] = compound
                        results.append(sentiment)
                        
                except Exception as e:
                    logger.error(f"Error in batch sentiment analysis: {e}")
                    # Add default values for the entire batch
                    for _ in batch:
                        results.append({'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0})
        else:
            # Process individually for other model types or if parallelism is disabled
            if USE_PARALLEL and len(texts) > 10:
                # Use thread pool for parallel processing
                with ThreadPoolExecutor(max_workers=min(MAX_THREADS, len(texts))) as executor:
                    results = list(executor.map(self.analyze_text, texts))
            else:
                # Process sequentially
                results = [self.analyze_text(text) for text in texts]
                
        logger.info(f"Analyzed {len(texts)} texts in {time.time() - start_time:.2f} seconds")
        return results
        
    def analyze_news_feed(self, news_df: pd.DataFrame, 
                        content_col: str = 'content',
                        date_col: str = 'date',
                        symbol_col: Optional[str] = 'symbol') -> pd.DataFrame:
        """
        Analyze sentiment from a news feed dataframe.
        
        Args:
            news_df: DataFrame with news articles
            content_col: Column name with article content
            date_col: Column name with article date
            symbol_col: Optional column name with ticker symbol
            
        Returns:
            DataFrame with sentiment scores
        """
        if content_col not in news_df.columns:
            logger.error(f"Content column {content_col} not found in news dataframe")
            return pd.DataFrame()
            
        # Analyze sentiment for each news item
        contents = news_df[content_col].tolist()
        sentiments = self.analyze_texts(contents)
        
        # Create results dataframe
        results = pd.DataFrame(sentiments)
        
        # Add date and symbol if available
        if date_col in news_df.columns:
            results[date_col] = news_df[date_col]
        if symbol_col in news_df.columns:
            results[symbol_col] = news_df[symbol_col]
            
        return results
        
    def analyze_social_media(self, posts_df: pd.DataFrame,
                           content_col: str = 'text',
                           date_col: str = 'created_at',
                           symbol_col: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze sentiment from social media posts.
        
        Args:
            posts_df: DataFrame with social media posts
            content_col: Column name with post content
            date_col: Column name with post date
            symbol_col: Optional column name with ticker symbol
            
        Returns:
            DataFrame with sentiment scores
        """
        return self.analyze_news_feed(
            posts_df, 
            content_col=content_col, 
            date_col=date_col, 
            symbol_col=symbol_col
        )
        
    def aggregate_sentiment(self, sentiment_df: pd.DataFrame,
                          date_col: str = 'date',
                          symbol_col: Optional[str] = 'symbol',
                          window_days: int = 1) -> pd.DataFrame:
        """
        Aggregate sentiment scores over a time window.
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            date_col: Column name with date
            symbol_col: Optional column name with ticker symbol
            window_days: Window size in days
            
        Returns:
            DataFrame with aggregated sentiment scores
        """
        if date_col not in sentiment_df.columns:
            logger.error(f"Date column {date_col} not found in sentiment dataframe")
            return pd.DataFrame()
            
        # Ensure date column is datetime
        sentiment_df[date_col] = pd.to_datetime(sentiment_df[date_col])
        
        # Group by date (and symbol if available)
        if symbol_col in sentiment_df.columns:
            # Group by symbol and date
            sentiment_df['date_trunc'] = sentiment_df[date_col].dt.floor('D')
            grouped = sentiment_df.groupby(['date_trunc', symbol_col])
        else:
            # Group by date only
            sentiment_df['date_trunc'] = sentiment_df[date_col].dt.floor('D')
            grouped = sentiment_df.groupby('date_trunc')
            
        # Aggregate sentiment scores
        aggregated = grouped.agg({
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean',
            'compound': 'mean',
            date_col: 'count'  # Count of items per day/symbol
        }).rename(columns={date_col: 'count'})
        
        # Calculate rolling averages if window_days > 1
        if window_days > 1:
            if symbol_col in sentiment_df.columns:
                # Rolling average for each symbol
                aggregated = aggregated.groupby(symbol_col).rolling(window_days).mean()
            else:
                # Rolling average for all data
                aggregated = aggregated.rolling(window_days).mean()
                
        return aggregated.reset_index()
        
    def sentiment_impact_score(self, sentiment_df: pd.DataFrame,
                             market_df: pd.DataFrame,
                             symbol_col: str = 'Symbol',
                             date_col: str = 'Date',
                             return_col: str = 'DailyReturn') -> pd.DataFrame:
        """
        Calculate the impact of sentiment on market returns.
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            market_df: DataFrame with market returns
            symbol_col: Column name with ticker symbol
            date_col: Column name with date
            return_col: Column name with return values
            
        Returns:
            DataFrame with sentiment impact scores
        """
        # Ensure date columns are datetime
        sentiment_df[date_col] = pd.to_datetime(sentiment_df[date_col])
        market_df[date_col] = pd.to_datetime(market_df[date_col])
        
        # Merge sentiment and market data
        if symbol_col in sentiment_df.columns:
            merged = pd.merge(
                sentiment_df,
                market_df[[symbol_col, date_col, return_col]],
                on=[symbol_col, date_col],
                how='inner'
            )
        else:
            # If no symbol column in sentiment, assume it applies to all symbols
            merged = pd.merge(
                sentiment_df,
                market_df[[date_col, return_col]],
                on=[date_col],
                how='inner'
            )
            
        if len(merged) == 0:
            logger.warning("No matching data after merging sentiment and market data")
            return pd.DataFrame()
            
        # Calculate correlation between sentiment and returns
        corr = merged.corr()[['compound', 'positive', 'negative']][return_col].to_dict()
        
        # Calculate impact scores
        impact_scores = {
            'compound_impact': corr.get('compound', 0) * merged['compound'].std() / merged[return_col].std(),
            'positive_impact': corr.get('positive', 0) * merged['positive'].std() / merged[return_col].std(),
            'negative_impact': corr.get('negative', 0) * merged['negative'].std() / merged[return_col].std()
        }
        
        # Add impact score to original sentiment data
        result = sentiment_df.copy()
        for score_name, score_value in impact_scores.items():
            result[score_name] = score_value
            
        return result
        
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Replace URLs with a placeholder
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        
        # Replace user mentions (for social media) with a placeholder
        text = re.sub(r'@\w+', '[USER]', text)
        
        # Replace stock tickers with a placeholder
        text = re.sub(r'\$[A-Za-z]+', '[TICKER]', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
