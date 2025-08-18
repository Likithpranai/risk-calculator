import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data.data_loader import DataLoader
from src.risk_engine.risk_calculator import RiskCalculator
from src.risk_engine.monte_carlo import MonteCarloSimulator
from src.models.risk_predictor import RiskPredictor
from src.sentiment.sentiment_analyzer import SentimentAnalyzer
from src.sentiment.real_time_feed import RealTimeSentimentFeed
from src.reports.report_generator import ReportGenerator

def generate_sample_market_data(output_dir=None):
    print("Generating sample market data...")
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    market_data = []
    
    for symbol in symbols:
        np.random.seed(symbols.index(symbol))
        
        initial_price = 100 + np.random.randint(0, 900)
        volatility = 0.01 + 0.01 * np.random.random()
        
        prices = [initial_price]
        for i in range(1, len(dates)):
            daily_return = np.random.normal(0.0002, volatility)
            price = prices[-1] * (1 + daily_return)
            prices.append(price)
        
        close_prices = np.array(prices)
        
        high_prices = close_prices * (1 + 0.005 + 0.01 * np.random.random(len(close_prices)))
        low_prices = close_prices * (1 - 0.005 - 0.01 * np.random.random(len(close_prices)))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0] * (1 + np.random.normal(0, 0.005))
        
        volumes = np.random.randint(1000000, 10000000, len(dates))
        
        symbol_data = pd.DataFrame({
            'Date': dates,
            'Symbol': symbol,
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        })
        
        market_data.append(symbol_data)
    
    market_df = pd.concat(market_data)
    market_df['DailyReturn'] = market_df.groupby('Symbol')['Close'].pct_change()
    market_df['LogReturn'] = np.log(market_df.groupby('Symbol')['Close'].pct_change() + 1)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        market_df.to_csv(os.path.join(output_dir, 'sample_market_data.csv'), index=False)
        print(f"Sample market data saved to {os.path.join(output_dir, 'sample_market_data.csv')}")
    
    return market_df

def generate_sample_trade_data(market_df, output_dir=None):
    print("Generating sample trade data...")
    
    latest_date = market_df['Date'].max()
    
    symbols = market_df['Symbol'].unique()
    
    trade_data = []
    for symbol in symbols:
        latest_price = market_df[(market_df['Symbol'] == symbol) & 
                               (market_df['Date'] == latest_date)]['Close'].values[0]
        
        quantity = np.random.randint(10, 1000)
        
        trade_data.append({
            'Symbol': symbol,
            'Quantity': quantity,
            'Price': latest_price,
            'TradeDate': latest_date
        })
    
    trade_df = pd.DataFrame(trade_data)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        trade_df.to_csv(os.path.join(output_dir, 'sample_trade_data.csv'), index=False)
        print(f"Sample trade data saved to {os.path.join(output_dir, 'sample_trade_data.csv')}")
    
    return trade_df

def generate_sample_sentiment_data(market_df, output_dir=None):
    print("Generating sample sentiment data...")
    
    sentiment_analyzer = SentimentAnalyzer()
    
    symbols = market_df['Symbol'].unique()
    dates = sorted(market_df['Date'].unique())[-30:]  # Last 30 days
    
    sentiment_data = []
    
    news_templates = [
        "{symbol} announces strong quarterly earnings, exceeding analyst expectations.",
        "{symbol} shares drop after disappointing product launch.",
        "{symbol} enters strategic partnership with major tech company.",
        "Analysts upgrade {symbol} stock rating to 'buy' citing growth potential.",
        "Investors concerned about {symbol}'s market position amid industry challenges.",
        "{symbol} reveals new AI initiative that could revolutionize their business model.",
        "{symbol} CEO discusses future growth strategies in recent interview.",
        "Market report suggests {symbol} is undervalued compared to peers.",
        "{symbol} faces regulatory scrutiny over recent business practices.",
        "New product announcement from {symbol} receives mixed reactions from experts."
    ]
    
    for symbol in symbols:
        symbol_returns = market_df[market_df['Symbol'] == symbol]['DailyReturn'].dropna().values
        
        for date in dates:
            # Generate 2-5 news articles per day
            num_articles = np.random.randint(2, 6)
            
            for _ in range(num_articles):
                # Select a random news template
                news_text = np.random.choice(news_templates).format(symbol=symbol)
                
                # Analyze sentiment
                sentiment = sentiment_analyzer.analyze(news_text)
                
                sentiment_data.append({
                    'symbol': symbol,
                    'date': date,
                    'text': news_text,
                    'source': np.random.choice(['news', 'twitter', 'blog']),
                    'compound': sentiment['compound'],
                    'positive': sentiment['positive'],
                    'negative': sentiment['negative'],
                    'neutral': sentiment['neutral']
                })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        sentiment_df.to_csv(os.path.join(output_dir, 'sample_sentiment_data.csv'), index=False)
        print(f"Sample sentiment data saved to {os.path.join(output_dir, 'sample_sentiment_data.csv')}")
    
    return sentiment_df

def test_risk_calculation(market_df, trade_df):
    print("\n--- Testing Risk Calculation ---")
    
    risk_calculator = RiskCalculator(confidence_level=0.95)
    
    trade_risk = risk_calculator.calculate_trade_risk_metrics(trade_df, market_df)
    portfolio_metrics = risk_calculator.calculate_portfolio_metrics(trade_risk, market_df)
    
    print("\nTrade-Level Risk Metrics:")
    print(trade_risk[['Symbol', 'Value', 'VaR_95', 'ES_95', 'Loss_Prob_5pct']].round(2))
    
    print("\nPortfolio-Level Risk Metrics:")
    for key, value in portfolio_metrics.items():
        print(f"{key}: {value:.2f}")
    
    return trade_risk, portfolio_metrics

def test_monte_carlo_simulation(market_df, trade_risk):
    print("\n--- Testing Monte Carlo Simulation ---")
    
    simulator = MonteCarloSimulator(iterations=1000)
    
    position_values = []
    symbols = []
    
    for _, row in trade_risk.iterrows():
        symbols.append(row['Symbol'])
        position_values.append(row['Value'])
    
    position_values = np.array(position_values)
    
    returns_data = []
    for symbol in symbols:
        symbol_market = market_df[market_df['Symbol'] == symbol]
        returns_data.append(symbol_market['DailyReturn'].dropna().values)
    
    means = np.array([np.mean(ret) for ret in returns_data])
    
    cov_matrix = np.zeros((len(symbols), len(symbols)))
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            if i == j:
                cov_matrix[i, j] = np.std(returns_data[i]) ** 2
            else:
                corr = np.corrcoef(returns_data[i], returns_data[j])[0, 1]
                cov_matrix[i, j] = corr * np.std(returns_data[i]) * np.std(returns_data[j])
    
    scenarios = simulator.define_standard_scenarios()
    
    # Test two scenarios: base_case and market_crash
    test_scenarios = {
        'base_case': scenarios['base_case'],
        'market_crash': scenarios['market_crash']
    }
    
    stress_results = simulator.run_stress_test(
        position_values=position_values,
        means=means,
        covariance_matrix=cov_matrix,
        scenarios=test_scenarios,
        time_periods=20
    )
    
    for scenario, results in stress_results.items():
        print(f"\nScenario: {scenario}")
        print(f"Mean Return: {results['summary']['mean_return']:.2%}")
        print(f"VaR (95%): {results['summary']['var_95']:.2%}")
        print(f"ES (95%): {results['summary']['es_95']:.2%}")
        print(f"Probability of Loss: {results['summary']['prob_loss']:.2%}")
    
    return stress_results

def test_ml_prediction(market_df, sentiment_df, trade_risk):
    print("\n--- Testing Machine Learning Risk Prediction ---")
    
    risk_predictor = RiskPredictor(use_sentiment=True)
    
    # Create a training dataset with historical volatility as target
    features = risk_predictor.prepare_features(market_df, sentiment_df, lookback_periods=10)
    
    # Add a target column (historical 10-day volatility)
    for symbol in market_df['Symbol'].unique():
        symbol_data = market_df[market_df['Symbol'] == symbol]
        
        if symbol_data.empty:
            continue
            
        symbol_returns = symbol_data['DailyReturn'].dropna()
        rolling_vol = symbol_returns.rolling(window=10).std() * np.sqrt(10)
        
        features.loc[features['Symbol'] == symbol, 'FutureVol_10d'] = rolling_vol.shift(-10)
    
    features = features.dropna()
    
    print("\nTraining ML model...")
    training_results = risk_predictor.train(features, 'FutureVol_10d')
    
    print(f"Training RMSE: {training_results['train_rmse']:.4f}")
    print(f"Test RMSE: {training_results['test_rmse']:.4f}")
    print(f"R² Score: {training_results['test_r2']:.4f}")
    
    print("\nTop 5 Feature Importances:")
    for i, (feature, importance) in enumerate(list(training_results['feature_importance'].items())[:5]):
        print(f"{feature}: {importance:.4f}")
    
    # Make predictions
    position_values = {}
    for _, row in trade_risk.iterrows():
        position_values[row['Symbol']] = row['Value']
    
    predictions = risk_predictor.predict_var_es(
        market_data=market_df,
        position_values=position_values,
        confidence_level=0.95,
        horizon=10,
        sentiment_data=sentiment_df
    )
    
    print("\nPredicted Risk Metrics:")
    for symbol, metrics in predictions.items():
        if symbol != 'Portfolio':
            print(f"{symbol} - Predicted Volatility: {metrics['PredictedVolatility']:.2%}, VaR: ${metrics['VaR']:.2f}")
    
    print(f"\nPortfolio - VaR: ${predictions['Portfolio']['VaR']:.2f}, ES: ${predictions['Portfolio']['ES']:.2f}")
    
    return predictions

def test_sentiment_analysis(sentiment_df):
    print("\n--- Testing Sentiment Analysis ---")
    
    sentiment_analyzer = SentimentAnalyzer()
    
    # Calculate average sentiment per symbol
    avg_sentiment = sentiment_df.groupby('symbol')['compound'].mean()
    
    print("\nAverage Sentiment by Symbol:")
    for symbol, compound in avg_sentiment.items():
        sentiment_label = "Positive" if compound > 0.1 else "Negative" if compound < -0.1 else "Neutral"
        print(f"{symbol}: {compound:.4f} ({sentiment_label})")
    
    # Test analyzing a new text
    test_text = "The company reported strong earnings, beating analyst expectations by 15%. Revenue growth accelerated to 22% year-over-year."
    sentiment = sentiment_analyzer.analyze(test_text)
    
    print("\nSample Text Sentiment Analysis:")
    print(f"Text: {test_text}")
    print(f"Compound Score: {sentiment['compound']:.4f}")
    print(f"Positive: {sentiment['positive']:.4f}, Negative: {sentiment['negative']:.4f}, Neutral: {sentiment['neutral']:.4f}")
    
    return avg_sentiment

def test_report_generation(trade_risk, portfolio_metrics, market_df, stress_results, sentiment_df, ml_predictions):
    print("\n--- Testing Report Generation ---")
    
    report_generator = ReportGenerator()
    
    output_path = report_generator.generate_risk_report(
        trade_risk_df=trade_risk,
        portfolio_metrics=portfolio_metrics,
        market_data=market_df,
        stress_test_results=stress_results,
        sentiment_data=sentiment_df,
        ml_predictions=ml_predictions,
        output_format="html"
    )
    
    print(f"\nRisk report generated successfully at: {output_path}")
    
    return output_path

def run_system_test():
    print("\n===== Running AI-Driven Trade Risk Assessment System Test =====\n")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), '../data/sample')
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate or load sample data
    market_data_file = os.path.join(data_dir, 'sample_market_data.csv')
    if os.path.exists(market_data_file):
        print("Loading existing market data...")
        market_df = pd.read_csv(market_data_file)
        market_df['Date'] = pd.to_datetime(market_df['Date'])
    else:
        market_df = generate_sample_market_data(data_dir)
    
    trade_data_file = os.path.join(data_dir, 'sample_trade_data.csv')
    if os.path.exists(trade_data_file):
        print("Loading existing trade data...")
        trade_df = pd.read_csv(trade_data_file)
        trade_df['TradeDate'] = pd.to_datetime(trade_df['TradeDate'])
    else:
        trade_df = generate_sample_trade_data(market_df, data_dir)
    
    sentiment_data_file = os.path.join(data_dir, 'sample_sentiment_data.csv')
    if os.path.exists(sentiment_data_file):
        print("Loading existing sentiment data...")
        sentiment_df = pd.read_csv(sentiment_data_file)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    else:
        sentiment_df = generate_sample_sentiment_data(market_df, data_dir)
    
    # Run the tests
    trade_risk, portfolio_metrics = test_risk_calculation(market_df, trade_df)
    stress_results = test_monte_carlo_simulation(market_df, trade_risk)
    ml_predictions = test_ml_prediction(market_df, sentiment_df, trade_risk)
    avg_sentiment = test_sentiment_analysis(sentiment_df)
    report_path = test_report_generation(trade_risk, portfolio_metrics, market_df, stress_results, sentiment_df, ml_predictions)
    
    print("\n===== System Test Completed Successfully =====\n")
    print(f"Full risk report available at: {report_path}")

if __name__ == "__main__":
    run_system_test()
