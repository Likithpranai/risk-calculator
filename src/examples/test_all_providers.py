import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_providers.market_data_provider import MarketDataProvider

def main():
    """Test all market data providers with the provided API keys"""
    
    # Set API keys as environment variables
    os.environ['ALPHA_VANTAGE_API_KEY'] = 'A95HQUB5AWCRUQ1J'
    os.environ['POLYGON_API_KEY'] = 'HGaV6oKbB4mbuVEzhfhaKJPeEi8blmjn'
    
    # Create market data provider
    data_provider = MarketDataProvider()
    
    # Define symbols to fetch
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    single_symbol = 'AAPL'  # For testing providers with rate limits
    
    print(f"Testing market data providers with {single_symbol} and {symbols}")
    
    # 1. Test Yahoo Finance
    print("\n===== TESTING YAHOO FINANCE =====")
    yahoo_data = data_provider.get_historical_prices(
        symbols=[single_symbol],
        period='1mo',  # Use shorter period for test
        provider='yahoo'
    )
    
    if not yahoo_data.empty:
        print(f"Successfully retrieved {len(yahoo_data)} records from Yahoo Finance")
        print("\nSample of Yahoo Finance data:")
        print(yahoo_data.head())
    else:
        print("No data retrieved from Yahoo Finance")
    
    # 2. Test Alpha Vantage
    print("\n===== TESTING ALPHA VANTAGE =====")
    alpha_data = data_provider.get_historical_prices(
        symbols=[single_symbol],
        provider='alphavantage'
    )
    
    if not alpha_data.empty:
        print(f"Successfully retrieved {len(alpha_data)} records from Alpha Vantage")
        print("\nSample of Alpha Vantage data:")
        print(alpha_data.head())
    else:
        print("No data retrieved from Alpha Vantage")
    
    # 3. Test Polygon
    print("\n===== TESTING POLYGON =====")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    polygon_data = data_provider.get_historical_prices(
        symbols=[single_symbol],
        start_date=start_date,
        end_date=end_date,
        provider='polygon'
    )
    
    if not polygon_data.empty:
        print(f"Successfully retrieved {len(polygon_data)} records from Polygon")
        print("\nSample of Polygon data:")
        print(polygon_data.head())
    else:
        print("No data retrieved from Polygon")
    
    # 4. Test fundamentals from all providers
    print("\n===== TESTING FUNDAMENTALS =====")
    
    print("\nYahoo Finance Fundamentals:")
    yahoo_fund = data_provider.get_company_fundamentals(single_symbol, provider='yahoo')
    if yahoo_fund:
        for key, value in yahoo_fund.items():
            print(f"  {key}: {value}")
    else:
        print("  No data retrieved")
        
    print("\nAlpha Vantage Fundamentals:")
    alpha_fund = data_provider.get_company_fundamentals(single_symbol, provider='alphavantage')
    if alpha_fund:
        for key, value in alpha_fund.items():
            print(f"  {key}: {value}")
    else:
        print("  No data retrieved")
        
    print("\nPolygon Fundamentals:")
    polygon_fund = data_provider.get_company_fundamentals(single_symbol, provider='polygon')
    if polygon_fund:
        for key, value in polygon_fund.items():
            print(f"  {key}: {value}")
    else:
        print("  No data retrieved")

if __name__ == "__main__":
    main()
