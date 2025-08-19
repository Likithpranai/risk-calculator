import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_providers.market_data_provider import MarketDataProvider

def main():
    """Example of using MarketDataProvider to fetch real market data"""
    
    # Get Alpha Vantage API key from environment variable or use None for Yahoo only
    alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    
    # Create market data provider
    data_provider = MarketDataProvider(alpha_vantage_api_key=alpha_vantage_key)
    
    # Define symbols to fetch
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    print(f"Fetching historical price data for {symbols}")
    
    # Get 1 year of historical data from Yahoo Finance
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Date range: {start_date} to {end_date}")
    print("Using Yahoo Finance API...")
    
    yahoo_data = data_provider.get_historical_prices(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        provider='yahoo'
    )
    
    if not yahoo_data.empty:
        print(f"Successfully retrieved {len(yahoo_data)} records from Yahoo Finance")
        print("\nSample of the data:")
        print(yahoo_data.head())
        
        # Calculate daily returns
        for symbol in symbols:
            symbol_data = yahoo_data[yahoo_data['Symbol'] == symbol].sort_values('Date')
            if len(symbol_data) > 1:
                print(f"\nDaily returns statistics for {symbol}:")
                returns = symbol_data['Price'].pct_change().dropna()
                print(f"Mean daily return: {returns.mean():.4f}")
                print(f"Standard deviation: {returns.std():.4f}")
                print(f"Min daily return: {returns.min():.4f}")
                print(f"Max daily return: {returns.max():.4f}")
    else:
        print("No data retrieved from Yahoo Finance")
    
    # If Alpha Vantage key is available, also try fetching from there
    if alpha_vantage_key:
        print("\nUsing Alpha Vantage API...")
        
        # Just fetch one symbol as Alpha Vantage has strict rate limits
        alpha_symbol = symbols[0]
        alpha_data = data_provider.get_historical_prices(
            symbols=[alpha_symbol],
            start_date=start_date,
            end_date=end_date,
            provider='alphavantage'
        )
        
        if not alpha_data.empty:
            print(f"Successfully retrieved {len(alpha_data)} records from Alpha Vantage for {alpha_symbol}")
            print("\nSample of the Alpha Vantage data:")
            print(alpha_data.head())
        else:
            print(f"No data retrieved from Alpha Vantage for {alpha_symbol}")
    else:
        print("\nAlpha Vantage API key not provided. Skipping Alpha Vantage data fetch.")
    
    # Fetch fundamentals for one symbol
    symbol = symbols[0]
    print(f"\nFetching fundamental data for {symbol} from Yahoo Finance...")
    yahoo_fundamentals = data_provider.get_company_fundamentals(symbol, provider='yahoo')
    
    if yahoo_fundamentals:
        print("Yahoo Finance fundamentals:")
        for key, value in yahoo_fundamentals.items():
            print(f"  {key}: {value}")
    else:
        print(f"No fundamental data retrieved for {symbol} from Yahoo Finance")

if __name__ == "__main__":
    main()
