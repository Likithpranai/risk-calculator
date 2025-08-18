"""
Data loader module for the AI-Driven Trade Risk Assessment System.
Optimized for low latency with efficient data loading techniques.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from concurrent.futures import ThreadPoolExecutor
import time

# Import configuration
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config.config import DATA_DIR, CHUNK_SIZE, USE_PARALLEL, MAX_THREADS

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader class that handles loading and preprocessing of market and trade data
    with optimizations for low latency.
    """
    
    def __init__(self, cache_data: bool = True):
        """
        Initialize the DataLoader.
        
        Args:
            cache_data: Whether to cache data in memory for faster access
        """
        self.cache_data = cache_data
        self.data_cache = {}
        self.last_load_time = {}
        
    def load_market_data(self, 
                        symbols: List[str], 
                        start_date: str, 
                        end_date: str, 
                        source: str = 'yahoo') -> pd.DataFrame:
        """
        Load historical market data for given symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            source: Data source ('yahoo', 'csv', etc.)
            
        Returns:
            DataFrame with market data
        """
        cache_key = f"market_{'-'.join(symbols)}_{start_date}_{end_date}_{source}"
        
        # Check if data is in cache
        if self.cache_data and cache_key in self.data_cache:
            logger.info(f"Using cached market data for {symbols}")
            return self.data_cache[cache_key]
        
        start_time = time.time()
        logger.info(f"Loading market data for {symbols} from {start_date} to {end_date}")
        
        if source == 'yahoo':
            try:
                import yfinance as yf
                
                # Use parallel processing for multiple symbols
                if USE_PARALLEL and len(symbols) > 1:
                    with ThreadPoolExecutor(max_workers=min(len(symbols), MAX_THREADS)) as executor:
                        data_pieces = list(executor.map(
                            lambda symbol: yf.download(symbol, start=start_date, end=end_date, progress=False),
                            symbols
                        ))
                    
                    # Combine data
                    market_data = pd.concat(data_pieces, keys=symbols, names=['Symbol', 'Date'])
                    market_data = market_data.reset_index()
                else:
                    market_data = yf.download(symbols, start=start_date, end=end_date, progress=False)
                    if len(symbols) == 1:
                        market_data['Symbol'] = symbols[0]
                    else:
                        market_data = market_data.reset_index()
                        
            except Exception as e:
                logger.error(f"Error loading market data from Yahoo: {e}")
                return pd.DataFrame()
                
        elif source == 'csv':
            try:
                file_path = os.path.join(DATA_DIR, 'market_data.csv')
                market_data = pd.read_csv(file_path)
                
                # Filter by date and symbols
                market_data = market_data[
                    (market_data['Date'] >= start_date) & 
                    (market_data['Date'] <= end_date) &
                    (market_data['Symbol'].isin(symbols))
                ]
                
            except Exception as e:
                logger.error(f"Error loading market data from CSV: {e}")
                return pd.DataFrame()
        else:
            logger.error(f"Unsupported data source: {source}")
            return pd.DataFrame()
            
        # Process data in chunks for memory efficiency if large
        if len(market_data) > CHUNK_SIZE:
            logger.info(f"Processing large market dataset in chunks")
            chunks = [market_data.iloc[i:i+CHUNK_SIZE] for i in range(0, len(market_data), CHUNK_SIZE)]
            processed_chunks = []
            
            for chunk in chunks:
                processed_chunk = self._preprocess_market_data(chunk)
                processed_chunks.append(processed_chunk)
                
            market_data = pd.concat(processed_chunks)
        else:
            market_data = self._preprocess_market_data(market_data)
            
        # Cache the data
        if self.cache_data:
            self.data_cache[cache_key] = market_data
            self.last_load_time[cache_key] = time.time()
            
        logger.info(f"Market data loaded in {time.time() - start_time:.2f} seconds")
        return market_data
        
    def load_trade_data(self, 
                       file_path: Optional[str] = None, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load trade data from CSV or database.
        
        Args:
            file_path: Path to the trade data file (if None, use default)
            start_date: Filter trades after this date
            end_date: Filter trades before this date
            
        Returns:
            DataFrame with trade data
        """
        cache_key = f"trade_{file_path}_{start_date}_{end_date}"
        
        # Check if data is in cache
        if self.cache_data and cache_key in self.data_cache:
            logger.info("Using cached trade data")
            return self.data_cache[cache_key]
            
        start_time = time.time()
        logger.info("Loading trade data")
        
        if file_path is None:
            file_path = os.path.join(DATA_DIR, 'sample/trades.csv')
            
        try:
            trades = pd.read_csv(file_path)
            
            # Apply date filters if provided
            if start_date:
                trades = trades[trades['TradeDate'] >= start_date]
            if end_date:
                trades = trades[trades['TradeDate'] <= end_date]
                
            # Process data in chunks for memory efficiency if large
            if len(trades) > CHUNK_SIZE:
                logger.info(f"Processing large trade dataset in chunks")
                chunks = [trades.iloc[i:i+CHUNK_SIZE] for i in range(0, len(trades), CHUNK_SIZE)]
                processed_chunks = []
                
                for chunk in chunks:
                    processed_chunk = self._preprocess_trade_data(chunk)
                    processed_chunks.append(processed_chunk)
                    
                trades = pd.concat(processed_chunks)
            else:
                trades = self._preprocess_trade_data(trades)
                
            # Cache the data
            if self.cache_data:
                self.data_cache[cache_key] = trades
                self.last_load_time[cache_key] = time.time()
                
            logger.info(f"Trade data loaded in {time.time() - start_time:.2f} seconds")
            return trades
            
        except Exception as e:
            logger.error(f"Error loading trade data: {e}")
            return pd.DataFrame()
            
    def _preprocess_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess market data for analysis.
        
        Args:
            data: Raw market data
            
        Returns:
            Preprocessed market data
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Convert date to datetime if it's not already
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
            
        # Calculate daily returns if 'Close' column exists
        if 'Close' in df.columns:
            # Group by symbol if multiple symbols
            if 'Symbol' in df.columns:
                df['DailyReturn'] = df.groupby('Symbol')['Close'].pct_change()
                df['LogReturn'] = np.log(df['Close'] / df.groupby('Symbol')['Close'].shift(1))
            else:
                df['DailyReturn'] = df['Close'].pct_change()
                df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
                
        # Forward fill any missing values
        df = df.ffill()
        
        return df
        
    def _preprocess_trade_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess trade data for analysis.
        
        Args:
            data: Raw trade data
            
        Returns:
            Preprocessed trade data
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Convert date columns to datetime
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
        # Calculate trade age if trade date exists
        if 'TradeDate' in df.columns:
            latest_date = df['TradeDate'].max()
            df['TradeAge'] = (latest_date - df['TradeDate']).dt.days
            
        return df
        
    def clear_cache(self, older_than_seconds: Optional[int] = None):
        """
        Clear the data cache, optionally only items older than specified time.
        
        Args:
            older_than_seconds: Only clear items older than this many seconds
        """
        if older_than_seconds is None:
            self.data_cache = {}
            self.last_load_time = {}
            logger.info("Cache cleared")
        else:
            current_time = time.time()
            keys_to_remove = [
                key for key, timestamp in self.last_load_time.items() 
                if current_time - timestamp > older_than_seconds
            ]
            
            for key in keys_to_remove:
                del self.data_cache[key]
                del self.last_load_time[key]
                
            logger.info(f"Cleared {len(keys_to_remove)} items from cache older than {older_than_seconds} seconds")
