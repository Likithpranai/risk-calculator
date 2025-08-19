import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

class MarketDataProvider:
    """Provides market data from various sources (Yahoo Finance, Alpha Vantage, Polygon)"""
    
    def __init__(self, alpha_vantage_api_key: Optional[str] = None, polygon_api_key: Optional[str] = None):
        """Initialize the market data provider
        
        Args:
            alpha_vantage_api_key: API key for Alpha Vantage (optional)
        """
        self.alpha_vantage_api_key = alpha_vantage_api_key or os.environ.get('ALPHA_VANTAGE_API_KEY')
        self.polygon_api_key = polygon_api_key or os.environ.get('POLYGON_API_KEY')
        
    def get_historical_prices(self, 
                             symbols: List[str], 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             period: str = '1y',
                             provider: str = 'yahoo') -> pd.DataFrame:
        """Get historical price data for a list of symbols
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD format), defaults to 1 year ago
            end_date: End date (YYYY-MM-DD format), defaults to today
            period: Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
                   Only used if start_date and end_date are None
            provider: 'yahoo' or 'alphavantage'
            
        Returns:
            DataFrame with historical price data
        """
        if provider == 'yahoo':
            return self._get_yahoo_historical_prices(symbols, start_date, end_date, period)
        elif provider == 'alphavantage':
            return self._get_alphavantage_historical_prices(symbols, start_date, end_date)
        elif provider == 'polygon':
            return self._get_polygon_historical_prices(symbols, start_date, end_date)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _get_yahoo_historical_prices(self,
                                    symbols: List[str],
                                    start_date: Optional[str] = None,
                                    end_date: Optional[str] = None,
                                    period: str = '1y') -> pd.DataFrame:
        """Get historical price data from Yahoo Finance"""
        try:
            if start_date and end_date:
                data = yf.download(symbols, start=start_date, end=end_date, progress=False)
            else:
                data = yf.download(symbols, period=period, progress=False)
            
            # Handle case where only one symbol is requested
            if len(symbols) == 1:
                data.columns = pd.MultiIndex.from_product([data.columns, symbols])
                
            # Reformat the data to a more convenient format
            prices_dict = {}
            for symbol in symbols:
                try:
                    symbol_data = data['Adj Close'][symbol].reset_index()
                    symbol_data.columns = ['Date', 'Price']
                    symbol_data['Symbol'] = symbol
                    prices_dict[symbol] = symbol_data
                except KeyError:
                    logger.warning(f"No data found for symbol: {symbol}")
            
            if not prices_dict:
                return pd.DataFrame()
                
            # Combine all symbols into a single DataFrame
            result = pd.concat(prices_dict.values(), ignore_index=True)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance: {e}")
            return pd.DataFrame()
            
    def _get_alphavantage_historical_prices(self,
                                           symbols: List[str],
                                           start_date: Optional[str] = None,
                                           end_date: Optional[str] = None) -> pd.DataFrame:
        """Get historical price data from Alpha Vantage"""
        if not self.alpha_vantage_api_key:
            logger.error("Alpha Vantage API key is not set")
            return pd.DataFrame()
            
        try:
            all_data = []
            
            for symbol in symbols:
                url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={self.alpha_vantage_api_key}&outputsize=full'
                response = requests.get(url)
                data = response.json()
                
                if 'Error Message' in data:
                    logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
                    continue
                    
                if 'Time Series (Daily)' not in data:
                    logger.error(f"No data found for {symbol} in Alpha Vantage response")
                    continue
                    
                time_series = data['Time Series (Daily)']
                
                symbol_data = []
                for date, values in time_series.items():
                    # Filter by date range if provided
                    if (start_date and date < start_date) or (end_date and date > end_date):
                        continue
                        
                    symbol_data.append({
                        'Date': date,
                        'Price': float(values['5. adjusted close']),
                        'Symbol': symbol
                    })
                
                all_data.extend(symbol_data)
            
            if not all_data:
                return pd.DataFrame()
                
            result = pd.DataFrame(all_data)
            result['Date'] = pd.to_datetime(result['Date'])
            return result
            
        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage: {e}")
            return pd.DataFrame()
            
    def _get_polygon_historical_prices(self,
                                      symbols: List[str],
                                      start_date: Optional[str] = None,
                                      end_date: Optional[str] = None) -> pd.DataFrame:
        """Get historical price data from Polygon.io"""
        if not self.polygon_api_key:
            logger.error("Polygon API key is not set")
            return pd.DataFrame()
            
        try:
            all_data = []
            
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                
            # Format dates for Polygon API (YYYY-MM-DD)
            start_date_formatted = start_date
            end_date_formatted = end_date
            
            for symbol in symbols:
                url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date_formatted}/{end_date_formatted}?apiKey={self.polygon_api_key}'
                response = requests.get(url)
                data = response.json()
                
                if 'results' not in data:
                    logger.error(f"Polygon API error for {symbol}: {data.get('error', 'Unknown error')}")
                    continue
                    
                symbol_data = []
                for result in data['results']:
                    # Convert timestamp to date
                    date = datetime.fromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d')
                    
                    symbol_data.append({
                        'Date': date,
                        'Price': result['c'],  # Closing price
                        'Symbol': symbol
                    })
                
                all_data.extend(symbol_data)
            
            if not all_data:
                return pd.DataFrame()
                
            result = pd.DataFrame(all_data)
            result['Date'] = pd.to_datetime(result['Date'])
            return result
            
        except Exception as e:
            logger.error(f"Error fetching data from Polygon: {e}")
            return pd.DataFrame()
    
    def get_company_fundamentals(self, symbol: str, provider: str = 'yahoo') -> Dict:
        """Get company fundamental data
        
        Args:
            symbol: Ticker symbol
            provider: 'yahoo' or 'alphavantage'
            
        Returns:
            Dictionary with company fundamental data
        """
        if provider == 'yahoo':
            return self._get_yahoo_fundamentals(symbol)
        elif provider == 'alphavantage':
            return self._get_alphavantage_fundamentals(symbol)
        elif provider == 'polygon':
            return self._get_polygon_fundamentals(symbol)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _get_yahoo_fundamentals(self, symbol: str) -> Dict:
        """Get company fundamentals from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key financial metrics
            fundamentals = {
                'name': info.get('shortName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0)
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals from Yahoo Finance for {symbol}: {e}")
            return {}
    
    def _get_alphavantage_fundamentals(self, symbol: str) -> Dict:
        """Get company fundamentals from Alpha Vantage"""
        if not self.alpha_vantage_api_key:
            logger.error("Alpha Vantage API key is not set")
            return {}
            
        try:
            url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={self.alpha_vantage_api_key}'
            response = requests.get(url)
            data = response.json()
            
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
                return {}
                
            fundamentals = {
                'name': data.get('Name', 'N/A'),
                'sector': data.get('Sector', 'N/A'),
                'industry': data.get('Industry', 'N/A'),
                'market_cap': float(data.get('MarketCapitalization', 0)),
                'pe_ratio': float(data.get('PERatio', 0)) if data.get('PERatio') else None,
                'dividend_yield': float(data.get('DividendYield', 0)) * 100,
                'beta': float(data.get('Beta', 0)),
                '52_week_high': float(data.get('52WeekHigh', 0)),
                '52_week_low': float(data.get('52WeekLow', 0))
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals from Alpha Vantage for {symbol}: {e}")
            return {}
            
    def _get_polygon_fundamentals(self, symbol: str) -> Dict:
        """Get company fundamentals from Polygon.io"""
        if not self.polygon_api_key:
            logger.error("Polygon API key is not set")
            return {}
            
        try:
            # Get ticker details
            url = f'https://api.polygon.io/v3/reference/tickers/{symbol}?apiKey={self.polygon_api_key}'
            response = requests.get(url)
            data = response.json()
            
            if 'results' not in data:
                logger.error(f"Polygon API error for {symbol}: {data.get('error', 'Unknown error')}")
                return {}
                
            ticker_info = data['results']
            
            # Get financials (if available)
            financials = {}
            try:
                fin_url = f'https://api.polygon.io/v2/reference/financials/{symbol}?apiKey={self.polygon_api_key}'
                fin_response = requests.get(fin_url)
                fin_data = fin_response.json()
                if 'results' in fin_data and fin_data['results']:
                    financials = fin_data['results'][0]
            except Exception as e:
                logger.warning(f"Could not retrieve financials for {symbol}: {e}")
            
            # Extract fundamentals
            fundamentals = {
                'name': ticker_info.get('name', 'N/A'),
                'sector': ticker_info.get('sic_description', 'N/A'),
                'industry': ticker_info.get('standard_industrial_classification', 'N/A'),
                'market_cap': ticker_info.get('market_cap', 0),
                'pe_ratio': None,  # Not directly available from ticker endpoint
                'dividend_yield': None,  # Need separate dividend API call
                'beta': None,  # Not directly available
                '52_week_high': None,  # Need separate endpoint
                '52_week_low': None   # Need separate endpoint
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals from Polygon for {symbol}: {e}")
            return {}
