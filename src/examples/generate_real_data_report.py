import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_providers.real_market_data import RealMarketData
from risk_engine.monte_carlo import MonteCarloSimulator
from risk_engine.risk_calculator import RiskCalculator
from reports.report_generator import ReportGenerator

def main():
    """Generate a PDF risk report using real market data from Polygon"""
    
    print("Fetching real market data for risk report...")
    
    # Define symbols to use for risk calculation
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    print(f"Using symbols: {', '.join(symbols)}")
    
    # Initialize real market data provider
    market_data = RealMarketData(use_provider='polygon')
    
    # Get formatted data for risk calculator
    data = market_data.get_formatted_data_for_risk_calculator(symbols, lookback_days=365)
    
    if data['prices'].empty or data['returns'].empty:
        print("Failed to retrieve necessary market data")
        return
    
    print(f"Successfully retrieved {len(data['prices'])} days of price data")
    
    # Define sample portfolio positions
    positions = {}
    portfolio_data = []
    
    # Get the latest prices
    latest_prices = data['prices'].iloc[-1]
    
    # Create a simple portfolio with equal weights
    investment_per_asset = 100000 / len(symbols)  # $100k portfolio equally divided
    total_value = 0
    
    for symbol in symbols:
        if symbol in latest_prices:
            price = latest_prices[symbol]
            shares = round(investment_per_asset / price)
            cost_basis = price * 0.9  # Assume bought at 10% less
            value = shares * price
            total_value += value
            
            positions[symbol] = {
                'shares': shares,
                'current_price': price,
                'cost_basis': cost_basis
            }
            
            # Create trade risk dataframe entry
            portfolio_data.append({
                'Symbol': symbol,
                'Quantity': shares,
                'Price': price,
                'Value': value,
                'CostBasis': cost_basis * shares,
                'UnrealizedPnL': value - (cost_basis * shares),
                'UnrealizedPnLPct': (price / cost_basis) - 1
            })
    
    # Create trade risk dataframe
    trade_risk_df = pd.DataFrame(portfolio_data)
    
    # Calculate portfolio metrics
    risk_calc = RiskCalculator(confidence_level=0.95)
    
    # Convert data to format needed for calculations - use a more explicit approach
    rows = []
    
    # Create one row at a time to ensure consistent lengths
    for symbol in symbols:
        if symbol in data['prices'].columns:
            # Get dates and prices
            for i, date in enumerate(data['prices'].index):
                price = data['prices'][symbol].iloc[i]
                # For returns, handle possible NaN for the first day
                daily_return = data['returns'][symbol].iloc[i] if i < len(data['returns']) else None
                
                # Create a row dictionary
                row = {
                    'Date': date,
                    'Symbol': symbol,
                    'Close': price,
                    'DailyReturn': daily_return
                }
                rows.append(row)
    
    # Create DataFrame from list of dictionaries
    market_data_df = pd.DataFrame(rows)
    
    # Calculate portfolio metrics using the same approach as API's risk calculator
    
    # Collect returns data for all symbols
    returns_data = {}
    for symbol in symbols:
        if symbol in data['returns'].columns:
            returns_data[symbol] = data['returns'][symbol].values
            
    # Calculate correlation and covariance matrices
    returns_df = pd.DataFrame(returns_data)
    correlation_matrix = returns_df.corr().values
    covariance_matrix = returns_df.cov().values
    
    # Get position values in same order as correlation matrix
    position_values_array = []
    for symbol in returns_df.columns:
        if symbol in positions:
            position_value = positions[symbol]['shares'] * positions[symbol]['current_price']
            position_values_array.append(position_value)
    
    position_values_array = np.array(position_values_array)
    
    # Calculate portfolio VaR using risk calculator's method
    portfolio_var = risk_calc.calculate_portfolio_var(position_values_array, covariance_matrix)
    
    # Calculate individual VaRs
    individual_vars = []
    for i, symbol in enumerate(returns_df.columns):
        if symbol in positions:
            position_value = positions[symbol]['shares'] * positions[symbol]['current_price']
            symbol_returns = returns_data[symbol]
            var = risk_calc.calculate_var(symbol_returns[~np.isnan(symbol_returns)], position_value, 'historical')
            individual_vars.append(var)
    
    # Calculate diversification benefit        
    sum_individual_vars = np.sum(individual_vars)
    diversification_benefit = 1 - (portfolio_var / sum_individual_vars) if sum_individual_vars > 0 else 0
    
    # Combine all returns for ES and loss probability calculations
    all_returns = np.concatenate([returns[~np.isnan(returns)] for returns in returns_data.values()])
    
    # Create the portfolio metrics dictionary with all required fields
    portfolio_metrics = {
        'TotalValue': total_value,
        'VaR_95': portfolio_var,
        'ES_95': risk_calc.calculate_expected_shortfall(all_returns, total_value),
        'DiversificationBenefit': diversification_benefit,
        'Loss_Prob_5pct': risk_calc.calculate_loss_probability(all_returns, 0.05),
        'Loss_Prob_10pct': risk_calc.calculate_loss_probability(all_returns, 0.10)
    }
    
    # We've already calculated all portfolio metrics above, so this block is no longer needed
    
    # Prepare for Monte Carlo simulation
    mc = MonteCarloSimulator(iterations=500)
    
    # Prepare inputs for Monte Carlo simulation
    position_values = np.array([pos['shares'] * pos['current_price'] for pos in positions.values()])
    
    # Calculate means and covariance matrix from returns
    asset_returns = data['returns'].values
    means = np.nanmean(asset_returns, axis=0)
    covariance_matrix = np.cov(asset_returns, rowvar=False)
    
    # Define standard scenarios
    scenarios = {
        'base_case': {
            'mean_shift': np.zeros_like(means),
            'volatility_multiplier': 1.0
        },
        'market_crash': {
            'mean_shift': -0.01 * np.ones_like(means),  # 1% daily loss
            'volatility_multiplier': 2.0  # Double volatility
        },
        'tech_boom': {
            'mean_shift': 0.005 * np.ones_like(means),  # 0.5% daily gain
            'volatility_multiplier': 1.2  # 20% more volatility
        }
    }
    
    # Run stress test
    stress_test_results = mc.run_stress_test(
        position_values=position_values,
        means=means,
        covariance_matrix=covariance_matrix,
        scenarios=scenarios,
        time_periods=20
    )
    
    # Generate the report
    print("\nGenerating risk report...")
    
    # Debug - check portfolio_metrics structure
    print("Portfolio metrics type:", type(portfolio_metrics))
    print("Portfolio metrics keys:", list(portfolio_metrics.keys()))
    for key, value in portfolio_metrics.items():
        print(f"  {key}: {value} (type: {type(value)})")
    
    report_gen = ReportGenerator()
    
    # Generate both HTML and PDF versions
    html_path = report_gen.generate_risk_report(
        trade_risk_df=trade_risk_df,
        portfolio_metrics=portfolio_metrics,
        market_data=market_data_df,
        stress_test_results=stress_test_results,
        output_format='html'
    )
    
    pdf_path = report_gen.generate_risk_report(
        trade_risk_df=trade_risk_df,
        portfolio_metrics=portfolio_metrics,
        market_data=market_data_df,
        stress_test_results=stress_test_results,
        output_format='pdf'
    )
    
    print(f"\nHTML report generated: {html_path}")
    print(f"PDF report generated: {pdf_path}")
    
    print("\nReal data risk report generation complete!")

if __name__ == "__main__":
    # Set environment variables for API keys
    os.environ['POLYGON_API_KEY'] = 'HGaV6oKbB4mbuVEzhfhaKJPeEi8blmjn'
    os.environ['ALPHA_VANTAGE_API_KEY'] = 'A95HQUB5AWCRUQ1J'
    
    main()
