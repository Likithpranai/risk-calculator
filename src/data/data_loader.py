
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from concurrent.futures import ThreadPoolExecutor
import time


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config.config import DATA_DIR, CHUNK_SIZE, USE_PARALLEL, MAX_THREADS
from src.data_providers.market_data_provider import MarketDataProvider


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    
    
    def __init__(self, cache_data: bool = True, polygon_api_key: Optional[str] = None):
        self.cache_data = cache_data
        self.data_cache = {}
        self.last_load_time = {}
        self.polygon_api_key = polygon_api_key or os.environ.get('POLYGON_API_KEY', 'HGaV6oKbB4mbuVEzhfhaKJPeEi8blmjn')
        self.market_data_provider = MarketDataProvider(polygon_api_key=self.polygon_api_key)
        
    def load_market_data(self, 
                        symbols: List[str], 
                        start_date: str, 
                        end_date: str, 
                        source: str = 'polygon') -> pd.DataFrame:

        cache_key = f"market_{'-'.join(symbols)}_{start_date}_{end_date}_{source}"
        

        if self.cache_data and cache_key in self.data_cache:
            logger.info(f"Using cached market data for {symbols}")
            return self.data_cache[cache_key]
        
        start_time = time.time()
        logger.info(f"Loading market data for {symbols} from {start_date} to {end_date}")
        
        if source == 'polygon':
            try:
                # Use the Polygon data provider
                market_data = self.market_data_provider.get_historical_prices(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    provider='polygon'
                )
                
                if market_data.empty:
                    logger.warning(f"No data returned from Polygon API for {symbols}. Falling back to alternate source.")
                    source = 'yahoo'
                else:
                    if 'Price' in market_data.columns and 'Close' not in market_data.columns:
                        market_data['Close'] = market_data['Price']
                        
            except Exception as e:
                logger.error(f"Error loading market data from Polygon: {e}")
                logger.info("Falling back to alternate data source")
                source = 'yahoo'
        
        if source == 'yahoo':
            try:
                import yfinance as yf
                logger.info("Using Yahoo Finance as data source")
                
                if USE_PARALLEL and len(symbols) > 1:
                    with ThreadPoolExecutor(max_workers=min(len(symbols), MAX_THREADS)) as executor:
                        data_pieces = list(executor.map(
                            lambda symbol: yf.download(symbol, start=start_date, end=end_date, progress=False),
                            symbols
                        ))
                    
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
                
                market_data = market_data[
                    (market_data['Date'] >= start_date) & 
                    (market_data['Date'] <= end_date) &
                    (market_data['Symbol'].isin(symbols))
                ]
                
            except Exception as e:
                logger.error(f"Error loading market data from CSV: {e}")
                return pd.DataFrame()
        elif source != 'polygon':
            logger.error(f"Unsupported data source: {source}")
            return pd.DataFrame()
            

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
            
        if self.cache_data:
            self.data_cache[cache_key] = market_data
            self.last_load_time[cache_key] = time.time()
            
        logger.info(f"Market data loaded in {time.time() - start_time:.2f} seconds")
        return market_data
        
    def load_trade_data(self, 
                       file_path: Optional[str] = None, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
            
        cache_key = f"trade_{file_path}_{start_date}_{end_date}"
        

        if self.cache_data and cache_key in self.data_cache:
            logger.info("Using cached trade data")
            return self.data_cache[cache_key]
            
        start_time = time.time()
        logger.info("Loading trade data")
        
        if file_path is None:
            file_path = os.path.join(DATA_DIR, 'sample/trades.csv')
            
        try:
            trades = pd.read_csv(file_path)
            
            if start_date:
                trades = trades[trades['TradeDate'] >= start_date]
            if end_date:
                trades = trades[trades['TradeDate'] <= end_date]
                
    
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
                
            if self.cache_data:
                self.data_cache[cache_key] = trades
                self.last_load_time[cache_key] = time.time()
                
            logger.info(f"Trade data loaded in {time.time() - start_time:.2f} seconds")
            return trades
            
        except Exception as e:
            logger.error(f"Error loading trade data: {e}")
            return pd.DataFrame()
            
    def _preprocess_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        

        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
            

        if 'Close' in df.columns:

            if 'Symbol' in df.columns:
                df['DailyReturn'] = df.groupby('Symbol')['Close'].pct_change()
                df['LogReturn'] = np.log(df['Close'] / df.groupby('Symbol')['Close'].shift(1))
            else:
                df['DailyReturn'] = df['Close'].pct_change()
                df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
                

        df = df.ffill()
        
        return df
        
    def _preprocess_trade_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        

        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
                

        if 'TradeDate' in df.columns:
            latest_date = df['TradeDate'].max()
            df['TradeAge'] = (latest_date - df['TradeDate']).dt.days
            
        return df
        
    def clear_cache(self, older_than_seconds: Optional[int] = None):


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
