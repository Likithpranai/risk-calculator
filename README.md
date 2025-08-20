# Risk Analysis System

## Project Description

The Risk Analysis System is a comprehensive financial portfolio risk assessment tool that leverages real-time market data to calculate critical risk metrics for investment decision-making. The system fetches live stock price data from multiple sources (Polygon API and Alpha Vantage) and performs advanced risk calculations to evaluate portfolio vulnerability under various market conditions.

At its core, the system calculates essential risk metrics including Value-at-Risk (VaR) and Expected Shortfall at different confidence levels (95%, 99%), allowing investors to quantify potential losses in normal and extreme market conditions. The system's Monte Carlo simulation engine runs thousands of iterations to model potential future portfolio performance across multiple scenarios.

The stress testing framework evaluates portfolio resilience under predefined crisis scenarios including market crashes, volatility spikes, correlation breakdowns, stagflation, and liquidity crises. This provides a comprehensive view of how investments might perform during periods of market turbulence.

The modular architecture ensures separation of concerns between data acquisition, risk calculation, and reporting components, allowing for easy maintenance and future extensions. The system handles API rate limitations gracefully and provides fallback mechanisms to ensure continuous operation.

## Tech Stack

- **Python**: Core programming language
- **NumPy & Pandas**: Data manipulation and numerical computations
- **SciPy**: Statistical analysis and probability distributions
- **Polygon API & Alpha Vantage API**: Real-time market data providers
- **Concurrent Processing**: Multi-threading for parallel computations
- **Matplotlib/Plotly** (Optional): Data visualization capabilities
- **FastAPI**: API endpoints for risk metrics delivery
- **WeasyPrint**: PDF report generation

# Paths
CACHE_DIR = "./cache"
MODEL_DIR = "./models"
REPORT_DIR = "./reports"
```

## Usage

### Command Line Interface

The system provides a command-line interface with several commands:

```bash
# Run the interactive dashboard
python main.py dashboard

# Run the API server
python main.py api

# Run a system test with sample data
python main.py test

# Run the sentiment feed for specific symbols
python main.py sentiment --symbols AAPL MSFT GOOGL

# Generate a risk report
python main.py report --symbols AAPL MSFT --start-date 2023-01-01 --end-date 2023-12-31 --format html
```

### API Endpoints

Once the API server is running, the following endpoints are available:

- `GET /market-data`: Get historical market data for specified symbols
- `POST /calculate-risk`: Calculate risk metrics for trades
- `POST /stress-test`: Run stress tests on a portfolio
- `GET /sentiment/{symbol}`: Get sentiment data for a symbol
- `POST /sentiment/fetch`: Fetch sentiment data for symbols
- `POST /sentiment/schedule`: Schedule periodic sentiment data fetching
- `POST /ml/predict`: Make ML-based risk predictions
- `POST /report/generate`: Generate a risk assessment report

API documentation is available at `http://localhost:8000/docs` when the API server is running.

### Dashboard

The interactive dashboard provides visualization of:
- Portfolio overview with key risk metrics
- Trade-level risk details
- Monte Carlo simulation results
- Sentiment impact analysis
- Machine learning predictions

Access the dashboard at `http://localhost:8050` when running the dashboard command.

## System Testing

The system includes a comprehensive test suite that:
1. Generates sample market, trade, and sentiment data
2. Tests risk calculation functionality
3. Runs Monte Carlo simulations
4. Tests machine learning predictions
5. Analyzes sentiment data
6. Generates a sample risk report

Run the test with:

```bash
python main.py test
```

## Project Structure

```
risk-calculator/
├── config/
│   └── config.py         # Configuration settings
├── data/                 # Data storage
│   ├── market/           # Market data files
│   ├── sentiment/        # Sentiment data cache
│   └── sample/           # Sample data for testing
├── models/               # Trained ML models
├── reports/              # Generated risk reports
├── src/
│   ├── api/              # API endpoints
│   ├── data/             # Data loading modules
│   ├── models/           # ML risk prediction models
│   ├── reports/          # Report generation
│   ├── risk_engine/      # Core risk calculations
│   ├── sentiment/        # Sentiment analysis
│   └── visualization/    # Dashboard and visualizations
├── tests/                # Test scripts
├── main.py               # CLI entry point
└── requirements.txt      # Project dependencies
```