import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import local modules
from src.data_providers.real_market_data import RealMarketData
from src.risk_engine.risk_calculator import RiskCalculator
from src.risk_engine.monte_carlo import MonteCarloSimulator

def run_real_time_risk_analysis(symbols=None, lookback_days=365):
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']
    
    logger.info(f"Starting real-time risk analysis for {symbols}")
    os.environ['POLYGON_API_KEY'] = 'HGaV6oKbB4mbuVEzhfhaKJPeEi8blmjn'
    os.environ['ALPHA_VANTAGE_API_KEY'] = 'A95HQUB5AWCRUQ1J'
    market_data = RealMarketData(use_provider='polygon')
    logger.info(f"Fetching historical data for the past {lookback_days} days")
    chunk_size = 3  
    all_historical_data = []
    
    for i in range(0, len(symbols), chunk_size):
        symbol_chunk = symbols[i:i+chunk_size]
        logger.info(f"Processing symbols chunk: {symbol_chunk}")
        if i > 0:
            logger.info("Waiting 15 seconds to avoid API rate limits...")
            import time
            time.sleep(15)
            
        chunk_data = market_data.get_historical_market_data(
            symbols=symbol_chunk,
            lookback_days=lookback_days
        )
        
        if not chunk_data.empty:
            all_historical_data.append(chunk_data)
    
    if all_historical_data:
        historical_data = pd.concat(all_historical_data)
    else:
        historical_data = pd.DataFrame()
    
    if historical_data.empty:
        logger.error("Failed to retrieve historical market data")
        return
    
    logger.info(f"Successfully retrieved {len(historical_data)} data points")
    if not historical_data.empty:
        latest_dates = historical_data.groupby('Symbol')['Date'].max().reset_index()
        latest_market_data = pd.DataFrame()
        for _, row in latest_dates.iterrows():
            symbol_data = historical_data[
                (historical_data['Symbol'] == row['Symbol']) & 
                (historical_data['Date'] == row['Date'])
            ]
            latest_market_data = pd.concat([latest_market_data, symbol_data])
            
        if latest_market_data.empty:
            logger.error("Failed to extract latest market data")
            return
            
        logger.info("Latest market prices:")
        for symbol in symbols:
            symbol_data = latest_market_data[latest_market_data['Symbol'] == symbol]
            if not symbol_data.empty:
                price = symbol_data.iloc[0]['Price']
                logger.info(f"{symbol}: ${price:.2f}")
    else:
        logger.error("No historical data available to extract latest prices")
        return
    
    portfolio = pd.DataFrame({
        'Symbol': symbols,
        'Quantity': [100] * len(symbols),
        'Value': [0] * len(symbols)
    })
    

    for i, row in portfolio.iterrows():
        symbol = row['Symbol']
        quantity = row['Quantity']
        
        symbol_data = latest_market_data[latest_market_data['Symbol'] == symbol]
        if not symbol_data.empty:
            price = symbol_data.iloc[0]['Price']
            portfolio.at[i, 'Value'] = price * quantity
    
    total_portfolio_value = portfolio['Value'].sum()
    logger.info(f"Total portfolio value: ${total_portfolio_value:.2f}")
    
    logger.info("Calculating risk metrics...")
    
    mc_simulator = MonteCarloSimulator(iterations=10000, use_parallel=True)

    risk_calculator = RiskCalculator()
    
    logger.info("Preparing return data for risk calculation...")
    
    price_data = historical_data.pivot(index='Date', columns='Symbol', values='Price')
    
    returns = price_data.pct_change().dropna().values.flatten()
    asset_returns = {}
    available_symbols = price_data.columns.tolist()
    logger.info(f"Available symbols in price data: {available_symbols}")
    
    for symbol in symbols:
        if symbol in price_data.columns:
            symbol_data = price_data[symbol].dropna()
            if len(symbol_data) >= 2:  
                asset_returns[symbol] = symbol_data.pct_change().dropna().values
        else:
            logger.warning(f"Symbol {symbol} not found in price data - skipping")

    if asset_returns:
        returns_df = pd.DataFrame(asset_returns)
    else:
        logger.warning("No asset returns available - creating empty DataFrame")
        returns_df = pd.DataFrame()

    stats = {
        'expected_return': returns.mean(),
        'volatility': returns.std(),
        'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0
    }
    
    logger.info("\nPortfolio Risk Metrics:")
    logger.info(f"Expected Return: {stats['expected_return']*100:.2f}%")
    logger.info(f"Volatility: {stats['volatility']*100:.2f}%")
    logger.info(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    
    confidence_levels = [0.95, 0.99]
    for confidence_level in confidence_levels:
        var = np.percentile(returns, 100 * (1 - confidence_level)) 
        es = returns[returns <= var].mean() if any(returns <= var) else var
        
        var_dollar = abs(var) * total_portfolio_value
        es_dollar = abs(es) * total_portfolio_value
        
        logger.info(f"\nAt {confidence_level*100}% confidence level:")
        logger.info(f"Value at Risk (VaR): {abs(var)*100:.2f}% (${var_dollar:.2f})")
        logger.info(f"Expected Shortfall: {abs(es)*100:.2f}% (${es_dollar:.2f})")

    logger.info("\nRunning Monte Carlo stress testing...")
    if not returns_df.empty:
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        position_values = np.zeros(len(returns_df.columns))
        for i, symbol in enumerate(returns_df.columns):
            position_idx = symbols.index(symbol)
            position_values[i] = portfolio.iloc[position_idx]['Value'] 
 
        scenarios = mc_simulator.define_standard_scenarios()
        
        logger.info("Running stress tests with real-time market data...")
        stress_results = mc_simulator.run_stress_test(
            position_values=position_values,
            means=mean_returns,
            covariance_matrix=cov_matrix,
            scenarios=scenarios,
            time_periods=30
        )
        
        for scenario, result in stress_results.items():
            final_values = result['portfolio_values'][:, -1]
            mean_final = np.mean(final_values)
            worst_case = np.percentile(final_values, 5)
            pct_change = ((mean_final / total_portfolio_value) - 1) * 100
            
            logger.info(f"\nScenario: {scenario}")
            logger.info(f"Mean Final Value: ${mean_final:.2f} (Change: {pct_change:.2f}%)")
            logger.info(f"Worst Case (5th percentile): ${worst_case:.2f}")
    else:
        logger.warning("Insufficient real-time market data for Monte Carlo simulation")
        logger.warning("Need price data for multiple symbols to run simulation")
    
    logger.info("\nReal-time risk analysis complete")
    logger.info("Successfully used live market data for risk calculations")
    logger.info(f"Analyzed {len(available_symbols)} symbols with real-time price data")

if __name__ == "__main__":
    run_real_time_risk_analysis()
