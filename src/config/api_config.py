import os
from typing import Dict, Optional

class APIConfig:
    @staticmethod
    def get_alpha_vantage_key() -> Optional[str]:
        """Get the Alpha Vantage API key from environment variable"""
        return os.environ.get('ALPHA_VANTAGE_API_KEY')
        
    @staticmethod
    def get_api_config() -> Dict[str, Dict]:
        """Get all API configurations"""
        return {
            'alpha_vantage': {
                'api_key': APIConfig.get_alpha_vantage_key(),
                'base_url': 'https://www.alphavantage.co/query',
                'daily_limit': 500  # Free tier limit
            },
            'yahoo_finance': {
                # Yahoo Finance doesn't require an API key for basic usage
                'daily_limit': None  # No explicit limit, but be respectful
            }
        }
