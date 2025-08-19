import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data.data_loader import DataLoader
from src.risk_engine.risk_calculator import RiskCalculator
from src.risk_engine.monte_carlo import MonteCarloSimulator

def load_real_market_data(symbols=None, days=252):
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    data_loader = DataLoader()
    
    market_data = data_loader.load_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        source='polygon'
    )
    
    return market_data

def generate_trade_data_from_market(market_df):
    latest_dates = market_df.groupby('Symbol')['Date'].max()
    
    trade_data = []
    for symbol, latest_date in latest_dates.items():
        try:
            symbol_data = market_df[(market_df['Symbol'] == symbol) & 
                                 (market_df['Date'] == latest_date)]
            
            if symbol_data.empty:
                continue
                
            latest_price = symbol_data['Close'].iloc[0]
            quantity = np.random.randint(50, 500)
            
            trade_data.append({
                'Symbol': symbol,
                'Quantity': quantity,
                'Price': latest_price,
                'TradeDate': latest_date
            })
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    trade_df = pd.DataFrame(trade_data)
    return trade_df

def calculate_risk_metrics(market_df, trade_df):
    risk_calculator = RiskCalculator(confidence_level=0.95)
    
    trade_risk = risk_calculator.calculate_trade_risk_metrics(trade_df, market_df)
    portfolio_metrics = risk_calculator.calculate_portfolio_metrics(trade_risk, market_df)
    
    print("\nTrade-Level Risk Metrics (Using Real Polygon Data):")
    print(trade_risk[['Symbol', 'Value', 'VaR_95', 'ES_95', 'Loss_Prob_5pct']].round(2))
    
    print("\nPortfolio-Level Risk Metrics (Using Real Polygon Data):")
    for key, value in portfolio_metrics.items():
        print(f"{key}: {value:.2f}")
    
    return trade_risk, portfolio_metrics

def run_monte_carlo_simulation(market_df, trade_risk):
    simulator = MonteCarloSimulator(iterations=1000)
    
    position_values = []
    symbols = []
    
    for _, row in trade_risk.iterrows():
        symbols.append(row['Symbol'])
        position_values.append(row['Value'])
    
    position_values = np.array(position_values)
    
    dates = sorted(market_df['Date'].unique())
    returns_by_symbol = {}
    returns_data = []
    
    for symbol in symbols:
        symbol_returns = []
        symbol_data = market_df[market_df['Symbol'] == symbol].sort_values('Date')
        returns_by_symbol[symbol] = symbol_data.set_index('Date')['DailyReturn'].to_dict()
        
        for date in dates:
            if date in symbol_data['Date'].values and not pd.isna(returns_by_symbol[symbol].get(date)):
                symbol_returns.append(returns_by_symbol[symbol][date])
        
        if symbol_returns:
            returns_data.append(np.array(symbol_returns))
    
    means = np.array([np.mean(ret) for ret in returns_data])
    
    cov_matrix = np.zeros((len(symbols), len(symbols)))
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            if i == j:
                cov_matrix[i, j] = np.std(returns_data[i]) ** 2
            else:
                common_indices = np.logical_and(
                    ~np.isnan(returns_data[i][:min(len(returns_data[i]), len(returns_data[j]))]),
                    ~np.isnan(returns_data[j][:min(len(returns_data[i]), len(returns_data[j]))])
                )
                
                if np.sum(common_indices) > 2:
                    x = returns_data[i][:min(len(returns_data[i]), len(returns_data[j]))][common_indices]
                    y = returns_data[j][:min(len(returns_data[i]), len(returns_data[j]))][common_indices]
                    corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
                else:
                    corr = 0
                    
                cov_matrix[i, j] = corr * np.std(returns_data[i]) * np.std(returns_data[j])
    
    scenarios = simulator.define_standard_scenarios()
    
    stress_results = simulator.run_stress_test(
        position_values=position_values,
        means=means,
        covariance_matrix=cov_matrix,
        scenarios=scenarios,
        time_periods=20
    )
    
    print("\nMonte Carlo Stress Test Results (Using Real Polygon Data):")
    for scenario, results in stress_results.items():
        print(f"\nScenario: {scenario}")
        print(f"Mean Return: {results['summary']['mean_return']:.2%}")
        print(f"VaR (95%): {results['summary']['var_95']:.2%}")
        print(f"ES (95%): {results['summary']['es_95']:.2%}")
        print(f"Probability of Loss: {results['summary']['prob_loss']:.2%}")
    
    return stress_results

def run_real_data_risk_assessment(symbols=None):
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    print(f"\n===== Running AI-Driven Trade Risk Assessment with Real Polygon Data =====\n")
    print(f"Using symbols: {', '.join(symbols)}")
    
    market_df = load_real_market_data(symbols)
    
    if market_df.empty:
        print("Failed to load market data from Polygon API.")
        return
    
    print(f"Loaded market data for {len(market_df['Symbol'].unique())} symbols")
    print(f"Date range: {market_df['Date'].min().date()} to {market_df['Date'].max().date()}")
    
    trade_df = generate_trade_data_from_market(market_df)
    
    if trade_df.empty:
        print("Failed to generate trade data.")
        return
    
    trade_risk, portfolio_metrics = calculate_risk_metrics(market_df, trade_df)
    stress_results = run_monte_carlo_simulation(market_df, trade_risk)
    
    print("\n===== Real Data Risk Assessment Completed Successfully =====\n")
    
    return market_df, trade_df, trade_risk, portfolio_metrics, stress_results

if __name__ == "__main__":
    run_real_data_risk_assessment()
