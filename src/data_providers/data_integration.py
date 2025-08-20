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
        logger.info(f"Fetching {lookback_days} days of market data for {len(symbols)} symbols")
        price_data = self.market_data.get_historical_prices(
            symbols=symbols,
            period=f"{lookback_days}d" if provider == 'yahoo' else None,
            provider=provider
        )
        
        if price_data.empty:
            logger.error("Failed to retrieve market data")
            return pd.DataFrame()
        formatted_data = self._format_data_for_risk_calculator(price_data)
        logger.info(f"Prepared market data with shape {formatted_data.shape}")
        
        return formatted_data
    
    def _format_data_for_risk_calculator(self, price_data: pd.DataFrame) -> pd.DataFrame:
        if price_data.empty:
            return pd.DataFrame()
        pivot_data = price_data.pivot(index='Date', columns='Symbol', values='Price')
        pivot_data = pivot_data.sort_index()
        returns_data = pivot_data.pct_change().dropna()
        result = {
            'prices': pivot_data,
            'returns': returns_data
        }
        
        return result
