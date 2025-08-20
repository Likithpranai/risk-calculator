import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_providers.real_market_data import RealMarketData
from risk_engine.monte_carlo import MonteCarloSimulator
from risk_engine.risk_calculator import RiskCalculator

def main():
    print("Fetching real market data for risk calculator...")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    print(f"Using symbols: {', '.join(symbols)}")
    
    market_data = RealMarketData(use_provider='polygon')
    
    data = market_data.get_formatted_data_for_risk_calculator(symbols, lookback_days=365)
    
    if data['prices'].empty or data['returns'].empty:
        print("Failed to retrieve necessary market data")
        return
    
    print(f"Successfully retrieved {len(data['prices'])} days of price data")
    print("\nPrice data sample:")
    print(data['prices'].tail())
    
    print("\nReturns data sample:")
    print(data['returns'].tail())
    positions = {}
    latest_prices = data['prices'].iloc[-1]
    investment_per_asset = 100000 / len(symbols) 
    for symbol in symbols:
        if symbol in latest_prices:
            price = latest_prices[symbol]
            shares = round(investment_per_asset / price)
            positions[symbol] = {
                'shares': shares,
                'current_price': price,
                'cost_basis': price * 0.9  
            }
    
    print("\nSample portfolio:")
    total_value = 0
    for symbol, pos in positions.items():
        value = pos['shares'] * pos['current_price']
        total_value += value
        print(f"{symbol}: {pos['shares']} shares, ${pos['current_price']:.2f}/share, Value: ${value:.2f}")
    print(f"Total portfolio value: ${total_value:.2f}")
    print("\nPerforming risk calculations...")
    returns_array = data['returns'].values
    risk_calc = RiskCalculator(confidence_level=0.95)
    portfolio_weights = []
    for symbol in data['returns'].columns:
        if symbol in positions:
            weight = positions[symbol]['shares'] * positions[symbol]['current_price'] / total_value
            portfolio_weights.append(weight)
        else:
            portfolio_weights.append(0)
    
    portfolio_weights = np.array(portfolio_weights)
    portfolio_returns = returns_array @ portfolio_weights
    var_95 = risk_calc.calculate_var(portfolio_returns, value=total_value)
    es_95 = risk_calc.calculate_expected_shortfall(portfolio_returns, value=total_value)
    
    print(f"\nPortfolio Value at Risk (95%): ${total_value * var_95:.2f}")
    print(f"Portfolio Expected Shortfall (95%): ${total_value * es_95:.2f}")
    print("\nRunning Monte Carlo simulation...")
    mc = MonteCarloSimulator(iterations=500)
    position_values = np.array([pos['shares'] * pos['current_price'] for pos in positions.values()])
    asset_returns = data['returns'].values
    means = np.nanmean(asset_returns, axis=0)
    covariance_matrix = np.cov(asset_returns, rowvar=False)
    scenarios = {
        'base_case': {
            'mean_shift': np.zeros_like(means),
            'volatility_multiplier': 1.0
        },
        'market_crash': {
            'mean_shift': -0.01 * np.ones_like(means), 
            'volatility_multiplier': 2.0  
        },
        'tech_boom': {
            'mean_shift': 0.005 * np.ones_like(means),  
            'volatility_multiplier': 1.2  
        }
    }
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
