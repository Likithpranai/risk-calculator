import os
import pandas as pd
import logging
from typing import List, Dict, Optional, Union

from .market_data_provider import MarketDataProvider
from ..config.api_config import APIConfig

logger = logging.getLogger(__name__)

class DataIntegration:
    def __init__(self):
        alpha_vantage_key = APIConfig.get_alpha_vantage_key()
        self.market_data = MarketDataProvider(alpha_vantage_api_key=alpha_vantage_key)
    
    def get_market_data_for_risk_calculator(self, 
                                           symbols: List[str], 
                                           lookback_days: int = 365,
                                           provider: str = 'yahoo') -> pd.DataFrame:
        """
        Fetch and format market data for use in the risk calculator
        
        Args:
            symbols: List of ticker symbols
            lookback_days: Number of days of historical data to fetch
            provider: Data provider ('yahoo' or 'alphavantage')
            
        Returns:
            DataFrame formatted for use in risk calculator
        """
        logger.info(f"Fetching {lookback_days} days of market data for {len(symbols)} symbols")
        
        # Get historical price data
        price_data = self.market_data.get_historical_prices(
            symbols=symbols,
            period=f"{lookback_days}d" if provider == 'yahoo' else None,
            provider=provider
        )
        
        if price_data.empty:
            logger.error("Failed to retrieve market data")
            return pd.DataFrame()
        
        # Format data for risk calculator
        formatted_data = self._format_data_for_risk_calculator(price_data)
        logger.info(f"Prepared market data with shape {formatted_data.shape}")
        
        return formatted_data
    
    def _format_data_for_risk_calculator(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Format price data for use in risk calculator
        
        The risk calculator expects data in the format:
        - Index: Date
        - Columns: Symbols
        - Values: Prices
        """
        if price_data.empty:
            return pd.DataFrame()
            
        # Convert to wide format (Date x Symbol)
        pivot_data = price_data.pivot(index='Date', columns='Symbol', values='Price')
        
        # Sort by date
        pivot_data = pivot_data.sort_index()
        
        # Calculate returns (for use in simulation)
        returns_data = pivot_data.pct_change().dropna()
        
        # Add returns to the dataset
        result = {
            'prices': pivot_data,
            'returns': returns_data
        }
        
        return result
