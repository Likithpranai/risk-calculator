import os
import logging
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta

# Use absolute imports
from src.data_providers.market_data_provider import MarketDataProvider
from src.config.api_config import APIConfig

logger = logging.getLogger(__name__)

class RealMarketData:
    def __init__(self, use_provider: Optional[str] = None):
        # Set environment variables for API keys
        os.environ['POLYGON_API_KEY'] = 'HGaV6oKbB4mbuVEzhfhaKJPeEi8blmjn'
        os.environ['ALPHA_VANTAGE_API_KEY'] = 'A95HQUB5AWCRUQ1J'
        
        # Default provider from config, can be overridden
        self.provider = use_provider or APIConfig.DEFAULT_PROVIDER
        self.data_provider = MarketDataProvider()
        logger.info(f"Initialized market data provider with {self.provider} as primary source")
        
    def get_historical_market_data(self, 
                                  symbols: List[str], 
                                  lookback_days: int = 365) -> pd.DataFrame:
        """
        Get historical market data for the given symbols
        
        Args:
            symbols: List of ticker symbols
            lookback_days: Number of days of history to retrieve
            
        Returns:
            DataFrame with historical price data
        """
        logger.info(f"Fetching {lookback_days} days of historical data for {len(symbols)} symbols")
        
        try:
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            data = self.data_provider.get_historical_prices(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                provider=self.provider
            )
            
            if data.empty:
                logger.error(f"Failed to retrieve market data from {self.provider}")
                return pd.DataFrame()
            
            logger.info(f"Successfully retrieved {len(data)} records for {len(symbols)} symbols")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving historical market data: {e}")
            return pd.DataFrame()
    
    def get_formatted_data_for_risk_calculator(self,
                                              symbols: List[str],
                                              lookback_days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Get and format market data for risk calculator
        
        Args:
            symbols: List of ticker symbols
            lookback_days: Number of days of history
            
        Returns:
            Dictionary with 'prices' and 'returns' DataFrames
        """
        # Get raw data
        raw_data = self.get_historical_market_data(symbols, lookback_days)
        
        if raw_data.empty:
            logger.error("No data available for risk calculation")
            return {
                'prices': pd.DataFrame(),
                'returns': pd.DataFrame()
            }
        
        # Convert data to wide format (dates x symbols)
        price_wide = raw_data.pivot(index='Date', columns='Symbol', values='Price')
        price_wide = price_wide.sort_index()  # Sort by date
        
        # Calculate daily returns
        returns_wide = price_wide.pct_change().dropna()
        
        logger.info(f"Prepared market data with {len(price_wide)} days and {len(symbols)} symbols")
        
        return {
            'prices': price_wide,
            'returns': returns_wide
        }
    
    def get_asset_fundamentals(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get fundamental data for assets
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dictionary mapping symbols to fundamental data
        """
        fundamentals = {}
        
        for symbol in symbols:
            try:
                fund_data = self.data_provider.get_company_fundamentals(
                    symbol=symbol,
                    provider=self.provider
                )
                
                if fund_data:
                    fundamentals[symbol] = fund_data
                
            except Exception as e:
                logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        
        logger.info(f"Retrieved fundamental data for {len(fundamentals)} out of {len(symbols)} symbols")
        return fundamentals
