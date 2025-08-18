
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


sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config.config import SENTIMENT_BATCH_SIZE, USE_PARALLEL, MAX_THREADS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:

    
    def __init__(self, 
                model_type: str = 'transformer',
                use_gpu: bool = False,
                batch_size: int = SENTIMENT_BATCH_SIZE):

        self.model_type = model_type
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        start_time = time.time()
        
        try:
            if self.model_type == 'transformer':

                try:
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
                    

                    model_name = "ProsusAI/finbert"
                    

                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    

                    logger.info(f"Initialized transformer tokenizer ({model_name})")
                    
                except ImportError:
                    logger.warning("Transformers library not available. Falling back to VADER")
                    self.model_type = 'vader'
                    
            if self.model_type == 'vader':

                try:
                    from nltk.sentiment.vader import SentimentIntensityAnalyzer
                    import nltk
                    

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

                try:
                    from textblob import TextBlob

                    logger.info("Will use TextBlob for sentiment analysis")
                except ImportError:
                    logger.error("No sentiment analysis libraries available")
                    
            logger.info(f"Sentiment model initialized in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment model: {e}")

            self.model_type = 'dictionary'
            
    def _ensure_model_loaded(self):
        if self.model_type == 'transformer' and self.model is None and self.tokenizer is not None:
            try:
                from transformers import AutoModelForSequenceClassification, pipeline
                
                logger.info("Loading transformer model (first use)...")
                start_time = time.time()
                
                model_name = "ProsusAI/finbert"
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                

                device = -1
                if self.use_gpu:
                    import torch
                    if torch.cuda.is_available():
                        device = 0
                        
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
        # Ensure model is loaded
        self._ensure_model_loaded()
        

        clean_text = self._preprocess_text(text)
        
        if not clean_text:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
            
        try:
            if self.model_type == 'transformer':

                result = self.model(clean_text[:512])[0]
                label = result['label'].lower()
                score = result['score']
                
                # Convert to common format
                sentiment = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
                sentiment[label] = score
                

                if label == 'positive':
                    compound = score
                elif label == 'negative':
                    compound = -score
                else:
                    compound = 0.0
                    
                sentiment['compound'] = compound
                
            elif self.model_type == 'vader':

                scores = self.model.polarity_scores(clean_text)
                sentiment = {
                    'positive': scores['pos'],
                    'negative': scores['neg'],
                    'neutral': scores['neu'],
                    'compound': scores['compound']
                }
                
            elif self.model_type == 'textblob':

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

                pos_words = ['up', 'rise', 'gain', 'positive', 'bull', 'bullish', 'growth', 'profit']
                neg_words = ['down', 'fall', 'drop', 'negative', 'bear', 'bearish', 'loss', 'decline']
                
                tokens = clean_text.lower().split()
                pos_count = sum(word in tokens for word in pos_words)
                neg_count = sum(word in tokens for word in neg_words)
                total = max(pos_count + neg_count, 1)
                
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
        if content_col not in news_df.columns:
            logger.error(f"Content column {content_col} not found in news dataframe")
            return pd.DataFrame()
            

        contents = news_df[content_col].tolist()
        sentiments = self.analyze_texts(contents)
        

        results = pd.DataFrame(sentiments)
        

        if date_col in news_df.columns:
            results[date_col] = news_df[date_col]
        if symbol_col in news_df.columns:
            results[symbol_col] = news_df[symbol_col]
            
        return results
        
    def analyze_social_media(self, posts_df: pd.DataFrame,
                           content_col: str = 'text',
                           date_col: str = 'created_at',
                           symbol_col: Optional[str] = None) -> pd.DataFrame:
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

        sentiment_df[date_col] = pd.to_datetime(sentiment_df[date_col])
        market_df[date_col] = pd.to_datetime(market_df[date_col])
        

        if symbol_col in sentiment_df.columns:
            merged = pd.merge(
                sentiment_df,
                market_df[[symbol_col, date_col, return_col]],
                on=[symbol_col, date_col],
                how='inner'
            )
        else:

            merged = pd.merge(
                sentiment_df,
                market_df[[date_col, return_col]],
                on=[date_col],
                how='inner'
            )
            
        if len(merged) == 0:
            logger.warning("No matching data after merging sentiment and market data")
            return pd.DataFrame()
            

        corr = merged.corr()[['compound', 'positive', 'negative']][return_col].to_dict()
        

        impact_scores = {
            'compound_impact': corr.get('compound', 0) * merged['compound'].std() / merged[return_col].std(),
            'positive_impact': corr.get('positive', 0) * merged['positive'].std() / merged[return_col].std(),
            'negative_impact': corr.get('negative', 0) * merged['negative'].std() / merged[return_col].std()
        }
        

        result = sentiment_df.copy()
        for score_name, score_value in impact_scores.items():
            result[score_name] = score_value
            
        return result
        
    def _preprocess_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
            

        text = text.lower()
        

        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        

        text = re.sub(r'@\w+', '[USER]', text)
        

        text = re.sub(r'\$[A-Za-z]+', '[TICKER]', text)
        

        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
