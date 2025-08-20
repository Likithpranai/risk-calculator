import os
import sys
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_providers.market_data_provider import MarketDataProvider

def main():
    alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    data_provider = MarketDataProvider(alpha_vantage_api_key=alpha_vantage_key)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    print(f"Fetching historical price data for {symbols}")
    
    print("Using Yahoo Finance API with period parameter...")
    yahoo_data = data_provider.get_historical_prices(
        symbols=symbols,
        period='1y',  
        provider='yahoo'
    )
    
    if not yahoo_data.empty:
        print(f"Successfully retrieved {len(yahoo_data)} records from Yahoo Finance")
        print("\nSample of the data:")
        print(yahoo_data.head())
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
    if alpha_vantage_key:
        print("\nUsing Alpha Vantage API...")
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
