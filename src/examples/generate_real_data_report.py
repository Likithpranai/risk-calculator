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
from reports.report_generator import ReportGenerator

def main():
    
    print("Fetching real market data for risk report...")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    print(f"Using symbols: {', '.join(symbols)}")
    market_data = RealMarketData(use_provider='polygon')
    data = market_data.get_formatted_data_for_risk_calculator(symbols, lookback_days=365)
    
    if data['prices'].empty or data['returns'].empty:
        print("Failed to retrieve necessary market data")
        return
    
    print(f"Successfully retrieved {len(data['prices'])} days of price data")
    positions = {}
    portfolio_data = []
    latest_prices = data['prices'].iloc[-1]
    investment_per_asset = 100000 / len(symbols)  
    total_value = 0
    
    for symbol in symbols:
        if symbol in latest_prices:
            price = latest_prices[symbol]
            shares = round(investment_per_asset / price)
            cost_basis = price * 0.9  
            value = shares * price
            total_value += value
            
            positions[symbol] = {
                'shares': shares,
                'current_price': price,
                'cost_basis': cost_basis
            }

            portfolio_data.append({
                'Symbol': symbol,
                'Quantity': shares,
                'Price': price,
                'Value': value,
                'CostBasis': cost_basis * shares,
                'UnrealizedPnL': value - (cost_basis * shares),
                'UnrealizedPnLPct': (price / cost_basis) - 1
            })

    trade_risk_df = pd.DataFrame(portfolio_data)

    risk_calc = RiskCalculator(confidence_level=0.95)
    
    rows = []
    for symbol in symbols:
        if symbol in data['prices'].columns:
            for i, date in enumerate(data['prices'].index):
                price = data['prices'][symbol].iloc[i]
                daily_return = data['returns'][symbol].iloc[i] if i < len(data['returns']) else None
                row = {
                    'Date': date,
                    'Symbol': symbol,
                    'Close': price,
                    'DailyReturn': daily_return
                }
                rows.append(row)
    market_data_df = pd.DataFrame(rows)
    returns_data = {}
    for symbol in symbols:
        if symbol in data['returns'].columns:
            returns_data[symbol] = data['returns'][symbol].values
    returns_df = pd.DataFrame(returns_data)
    correlation_matrix = returns_df.corr().values
    covariance_matrix = returns_df.cov().values
    position_values_array = []
    for symbol in returns_df.columns:
        if symbol in positions:
            position_value = positions[symbol]['shares'] * positions[symbol]['current_price']
            position_values_array.append(position_value)
    
    position_values_array = np.array(position_values_array)
    portfolio_var = risk_calc.calculate_portfolio_var(position_values_array, covariance_matrix)
    individual_vars = []
    for i, symbol in enumerate(returns_df.columns):
        if symbol in positions:
            position_value = positions[symbol]['shares'] * positions[symbol]['current_price']
            symbol_returns = returns_data[symbol]
            var = risk_calc.calculate_var(symbol_returns[~np.isnan(symbol_returns)], position_value, 'historical')
            individual_vars.append(var)    
    sum_individual_vars = np.sum(individual_vars)
    diversification_benefit = 1 - (portfolio_var / sum_individual_vars) if sum_individual_vars > 0 else 0

    all_returns = np.concatenate([returns[~np.isnan(returns)] for returns in returns_data.values()])

    portfolio_metrics = {
        'TotalValue': total_value,
        'VaR_95': portfolio_var,
        'ES_95': risk_calc.calculate_expected_shortfall(all_returns, total_value),
        'DiversificationBenefit': diversification_benefit,
        'Loss_Prob_5pct': risk_calc.calculate_loss_probability(all_returns, 0.05),
        'Loss_Prob_10pct': risk_calc.calculate_loss_probability(all_returns, 0.10)
    }
    
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
    
    stress_test_results = mc.run_stress_test(
        position_values=position_values,
        means=means,
        covariance_matrix=covariance_matrix,
        scenarios=scenarios,
        time_periods=20
    )
    
    print("\nGenerating risk report...")
    print("Portfolio metrics type:", type(portfolio_metrics))
    print("Portfolio metrics keys:", list(portfolio_metrics.keys()))
    for key, value in portfolio_metrics.items():
        print(f"  {key}: {value} (type: {type(value)})")
    
    report_gen = ReportGenerator()
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
    os.environ['POLYGON_API_KEY'] = 'HGaV6oKbB4mbuVEzhfhaKJPeEi8blmjn'
    os.environ['ALPHA_VANTAGE_API_KEY'] = 'A95HQUB5AWCRUQ1J'
    
    main()
