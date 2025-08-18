#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.data_loader import DataLoader
from src.risk_engine.risk_calculator import RiskCalculator
from src.risk_engine.monte_carlo import MonteCarloSimulator
from src.models.risk_predictor import RiskPredictor
from src.sentiment.sentiment_analyzer import SentimentAnalyzer
from src.sentiment.real_time_feed import RealTimeSentimentFeed
from src.reports.report_generator import ReportGenerator
from src.visualization.dashboard import app as dashboard_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def ensure_directories_exist():
    directories = [
        'data',
        'data/market',
        'data/trades',
        'data/sentiment',
        'reports',
        'models',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def run_dashboard():
    logger.info("Starting Risk Dashboard application...")
    dashboard_app.run_server(debug=True, host='0.0.0.0', port=8050)

def run_api():
    logger.info("Starting API server...")
    try:
        from src.api.app import app as api_app
        import uvicorn
        uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        logger.error("Failed to import API app. Make sure FastAPI and uvicorn are installed.")
        sys.exit(1)

def run_system_test():
    logger.info("Running system test...")
    
    # Add the current directory to the Python path to ensure imports work
    base_path = os.path.dirname(os.path.abspath(__file__))
    if base_path not in sys.path:
        sys.path.insert(0, base_path)
    
    # Import and execute the individual test functions directly rather than
    # importing the run_system_test function which has scope issues
    try:
        from tests.system_test import (
            generate_sample_market_data,
            generate_sample_trade_data,
            generate_sample_sentiment_data,
            test_risk_calculation,
            test_monte_carlo_simulation,
            test_ml_prediction,
            test_sentiment_analysis,
            test_report_generation
        )
        
        print("\n===== Running AI-Driven Trade Risk Assessment System Test =====")
        
        # Follow the same steps as the original run_system_test function
        market_df = generate_sample_market_data()
        trade_df = generate_sample_trade_data(market_df)
        sentiment_df = generate_sample_sentiment_data(market_df)
        
        trade_risk, portfolio_metrics = test_risk_calculation(market_df, trade_df)
        stress_results = test_monte_carlo_simulation(market_df, trade_risk)
        ml_predictions = test_ml_prediction(market_df, sentiment_df, trade_risk)
        test_sentiment_analysis(sentiment_df)
        
        report_path = test_report_generation(
            trade_risk, 
            portfolio_metrics, 
            market_df, 
            stress_results,
            sentiment_df, 
            ml_predictions
        )
        
        print("\n===== System Test Completed Successfully =====")
        print(f"Report generated at: {report_path}")
        
    except ImportError as e:
        logger.error(f"Failed to import system test modules: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"System test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_sentiment_feed(symbols=None):
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    logger.info(f"Starting real-time sentiment feed for symbols: {symbols}")
    
    sentiment_feed = RealTimeSentimentFeed()
    sentiment_feed.start()
    
    try:
        sentiment_feed.fetch_data_for_symbols(symbols)
        sentiment_feed.schedule_data_fetching(symbols, interval_minutes=60)
        
        logger.info("Press Ctrl+C to stop the sentiment feed...")
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Stopping sentiment feed...")
    finally:
        sentiment_feed.stop()

def generate_report(symbols=None, start_date=None, end_date=None, output_format="html"):
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    if start_date is None:
        start_date = (datetime.now().replace(day=1) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Generating risk report for symbols: {symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    data_loader = DataLoader()
    risk_calculator = RiskCalculator()
    monte_carlo = MonteCarloSimulator()
    risk_predictor = RiskPredictor()
    sentiment_feed = RealTimeSentimentFeed()
    report_generator = ReportGenerator()
    

    market_data = data_loader.load_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    if market_data.empty:
        logger.error("No market data available. Using system test data.")
        from tests.system_test import generate_sample_market_data
        market_data = generate_sample_market_data()
    

    latest_date = market_data['Date'].max()
    
    trade_data = []
    for symbol in symbols:
        symbol_data = market_data[market_data['Symbol'] == symbol]
        if symbol_data.empty:
            continue
        
        latest_price = symbol_data.sort_values('Date').iloc[-1]['Close']
        
        trade_data.append({
            'Symbol': symbol,
            'Quantity': 100,
            'Price': latest_price,
            'TradeDate': latest_date
        })
    
    trade_df = pd.DataFrame(trade_data)
    

    trade_risk = risk_calculator.calculate_trade_risk_metrics(trade_df, market_data)
    portfolio_metrics = risk_calculator.calculate_portfolio_metrics(trade_risk, market_data)
    

    position_values = {}
    for _, row in trade_risk.iterrows():
        position_values[row['Symbol']] = row['Value']
    
    symbols_list = list(position_values.keys())
    position_values_array = np.array(list(position_values.values()))
    
    returns_data = []
    for symbol in symbols_list:
        symbol_market = market_data[market_data['Symbol'] == symbol]
        returns_data.append(symbol_market['DailyReturn'].dropna().values)
    
    means = np.array([np.mean(ret) for ret in returns_data])
    
    cov_matrix = np.zeros((len(symbols_list), len(symbols_list)))
    for i in range(len(symbols_list)):
        for j in range(len(symbols_list)):
            if i == j:
                cov_matrix[i, j] = np.std(returns_data[i]) ** 2
            else:
                corr = np.corrcoef(returns_data[i], returns_data[j])[0, 1]
                cov_matrix[i, j] = corr * np.std(returns_data[i]) * np.std(returns_data[j])
    
    scenarios = monte_carlo.define_standard_scenarios()
    stress_results = monte_carlo.run_stress_test(
        position_values=position_values_array,
        means=means,
        covariance_matrix=cov_matrix,
        scenarios=scenarios,
        time_periods=20
    )
    

    sentiment_data = pd.concat([
        sentiment_feed.aggregate_sentiment_data(symbol, days=30)
        for symbol in symbols_list
    ])
    
    if sentiment_data.empty:
        logger.warning("No sentiment data available. Report will not include sentiment analysis.")
    

    ml_predictions = risk_predictor.predict_var_es(
        market_data=market_data,
        position_values=position_values,
        confidence_level=0.95,
        horizon=10,
        sentiment_data=sentiment_data
    )
    

    output_path = report_generator.generate_risk_report(
        trade_risk_df=trade_risk,
        portfolio_metrics=portfolio_metrics,
        market_data=market_data,
        stress_test_results=stress_results,
        sentiment_data=sentiment_data,
        ml_predictions=ml_predictions,
        output_format=output_format
    )
    
    logger.info(f"Risk report generated successfully: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="AI-Driven Trade Risk Assessment System")
    

    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    

    dashboard_parser = subparsers.add_parser('dashboard', help='Run the risk dashboard web application')
    

    api_parser = subparsers.add_parser('api', help='Run the REST API server')
    

    test_parser = subparsers.add_parser('test', help='Run the system test')
    

    sentiment_parser = subparsers.add_parser('sentiment', help='Run the real-time sentiment feed')
    sentiment_parser.add_argument('--symbols', nargs='+', help='List of stock symbols to monitor')
    

    report_parser = subparsers.add_parser('report', help='Generate a risk assessment report')
    report_parser.add_argument('--symbols', nargs='+', help='List of stock symbols to include in the report')
    report_parser.add_argument('--start-date', help='Start date for historical data (YYYY-MM-DD)')
    report_parser.add_argument('--end-date', help='End date for historical data (YYYY-MM-DD)')
    report_parser.add_argument('--format', choices=['html', 'pdf', 'json'], default='html', help='Output format')
    
    args = parser.parse_args()
    

    ensure_directories_exist()
    

    if args.command == 'dashboard':
        run_dashboard()
    elif args.command == 'api':
        run_api()
    elif args.command == 'test':
        run_system_test()
    elif args.command == 'sentiment':
        run_sentiment_feed(args.symbols)
    elif args.command == 'report':
        generate_report(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            output_format=args.format
        )
    else:

        parser.print_help()

if __name__ == "__main__":
    main()
