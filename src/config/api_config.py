import os
from typing import Dict, Optional

class APIConfig:
    DEFAULT_PROVIDER = 'polygon' 
    @staticmethod
    def get_alpha_vantage_key() -> Optional[str]:
        return os.environ.get('ALPHA_VANTAGE_API_KEY')
        
    @staticmethod
    def get_polygon_key() -> Optional[str]:
        return os.environ.get('POLYGON_API_KEY')
        
    @staticmethod
    def get_api_config() -> Dict[str, Dict]:
        return {
            'polygon': {
                'api_key': APIConfig.get_polygon_key(),
                'base_url': 'https://api.polygon.io',
                'daily_limit': 5000  
            },
            'alpha_vantage': {
                'api_key': APIConfig.get_alpha_vantage_key(),
                'base_url': 'https://www.alphavantage.co/query',
                'daily_limit': 500 
            },
            'yahoo_finance': {
                'daily_limit': None  
            }
        }
