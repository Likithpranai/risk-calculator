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

def main():
    """Test the risk calculator with real market data from Polygon"""
    
    print("Fetching real market data for risk calculator...")
    
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
    print("\nPrice data sample:")
    print(data['prices'].tail())
    
    print("\nReturns data sample:")
    print(data['returns'].tail())
    
    # Define sample portfolio positions
    positions = {}
    
    # Get the latest prices
    latest_prices = data['prices'].iloc[-1]
    
    # Create a simple portfolio with equal weights
    investment_per_asset = 100000 / len(symbols)  # $100k portfolio equally divided
    for symbol in symbols:
        if symbol in latest_prices:
            price = latest_prices[symbol]
            shares = round(investment_per_asset / price)
            positions[symbol] = {
                'shares': shares,
                'current_price': price,
                'cost_basis': price * 0.9  # Assume bought at 10% less
            }
    
    print("\nSample portfolio:")
    total_value = 0
    for symbol, pos in positions.items():
        value = pos['shares'] * pos['current_price']
        total_value += value
        print(f"{symbol}: {pos['shares']} shares, ${pos['current_price']:.2f}/share, Value: ${value:.2f}")
    print(f"Total portfolio value: ${total_value:.2f}")
    
    # Run risk calculations
    print("\nPerforming risk calculations...")
    
    # Calculate returns
    returns_array = data['returns'].values
    
    # Create risk calculator with 95% confidence level
    risk_calc = RiskCalculator(confidence_level=0.95)
    
    # Calculate VaR and ES for portfolio
    portfolio_weights = []
    for symbol in data['returns'].columns:
        if symbol in positions:
            weight = positions[symbol]['shares'] * positions[symbol]['current_price'] / total_value
            portfolio_weights.append(weight)
        else:
            portfolio_weights.append(0)
    
    portfolio_weights = np.array(portfolio_weights)
    portfolio_returns = returns_array @ portfolio_weights
    
    # Calculate VaR and ES using the correct API
    var_95 = risk_calc.calculate_var(portfolio_returns, value=total_value)
    es_95 = risk_calc.calculate_expected_shortfall(portfolio_returns, value=total_value)
    
    print(f"\nPortfolio Value at Risk (95%): ${total_value * var_95:.2f}")
    print(f"Portfolio Expected Shortfall (95%): ${total_value * es_95:.2f}")
    
    # Run Monte Carlo simulation
    print("\nRunning Monte Carlo simulation...")
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
    stress_results = mc.run_stress_test(
        position_values=position_values,
        means=means,
        covariance_matrix=covariance_matrix,
        scenarios=scenarios,
        time_periods=20
    )
    
    print("\nMonte Carlo Stress Test Results:")
    for scenario, results in stress_results.items():
        summary = results['summary']
        print(f"\n{scenario}:")
        print(f"  Mean Return: {summary['mean_return']:.2%}")
        print(f"  VaR (95%): {summary['var_95']:.2%}")
        print(f"  ES (95%): {summary['es_95']:.2%}")
        print(f"  Probability of Loss: {summary['prob_loss']:.2%}")
    
    print("\nReal data risk analysis complete!")

if __name__ == "__main__":
    main()
