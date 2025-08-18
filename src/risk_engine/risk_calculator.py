"""
Risk Calculator module for the AI-Driven Trade Risk Assessment System.
Implements core risk metrics (VaR, ES) with optimizations for low latency.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import os
import sys

# Import configuration
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config.config import DEFAULT_CONFIDENCE_LEVEL, USE_PARALLEL, MAX_THREADS

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskCalculator:
    """
    Calculates various risk metrics for trades and portfolios.
    Designed for low latency operation with parallel processing where appropriate.
    """
    
    def __init__(self, confidence_level: float = DEFAULT_CONFIDENCE_LEVEL):
        """
        Initialize the RiskCalculator.
        
        Args:
            confidence_level: Confidence level for VaR and ES calculations (default: 0.95)
        """
        self.confidence_level = confidence_level
        
    def calculate_var(self, 
                     returns: np.ndarray, 
                     value: float,
                     method: str = 'historical',
                     time_horizon: int = 1) -> float:
        """
        Calculate Value at Risk (VaR) for a given series of returns.
        
        Args:
            returns: Array of historical returns
            value: Current position value
            method: Method to calculate VaR ('historical', 'parametric', 'monte_carlo')
            time_horizon: Time horizon in days
            
        Returns:
            VaR value at specified confidence level
        """
        start_time = time.time()
        
        # Remove NaN values
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            logger.warning("Empty returns array provided for VaR calculation")
            return 0.0
            
        if method == 'historical':
            # Historical simulation method (non-parametric)
            var = np.percentile(returns, 100 * (1 - self.confidence_level)) * value * np.sqrt(time_horizon)
            
        elif method == 'parametric':
            # Parametric method (assumes normal distribution)
            mean = np.mean(returns)
            std = np.std(returns)
            var = -1 * (mean + std * stats.norm.ppf(self.confidence_level)) * value * np.sqrt(time_horizon)
            
        elif method == 'monte_carlo':
            # Monte Carlo simulation method
            logger.warning("Monte Carlo VaR requested but using simpler method - use stress_test for MC simulations")
            # Fall back to parametric
            mean = np.mean(returns)
            std = np.std(returns)
            var = -1 * (mean + std * stats.norm.ppf(self.confidence_level)) * value * np.sqrt(time_horizon)
            
        else:
            logger.error(f"Unsupported VaR method: {method}")
            return 0.0
            
        logger.info(f"VaR calculation completed in {time.time() - start_time:.4f} seconds")
        return abs(var)  # Return absolute value for easier interpretation
        
    def calculate_expected_shortfall(self,
                                   returns: np.ndarray,
                                   value: float,
                                   method: str = 'historical',
                                   time_horizon: int = 1) -> float:
        """
        Calculate Expected Shortfall (ES) / Conditional VaR.
        
        Args:
            returns: Array of historical returns
            value: Current position value
            method: Method to calculate ES ('historical', 'parametric')
            time_horizon: Time horizon in days
            
        Returns:
            ES value at specified confidence level
        """
        start_time = time.time()
        
        # Remove NaN values
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            logger.warning("Empty returns array provided for ES calculation")
            return 0.0
            
        if method == 'historical':
            # Historical simulation method
            var_cutoff = np.percentile(returns, 100 * (1 - self.confidence_level))
            es = returns[returns <= var_cutoff].mean() * value * np.sqrt(time_horizon)
            
        elif method == 'parametric':
            # Parametric method (assumes normal distribution)
            mean = np.mean(returns)
            std = np.std(returns)
            var_quantile = stats.norm.ppf(1 - self.confidence_level)
            es = -1 * (mean + std * stats.norm.pdf(var_quantile) / (1 - self.confidence_level)) * value * np.sqrt(time_horizon)
            
        else:
            logger.error(f"Unsupported ES method: {method}")
            return 0.0
            
        logger.info(f"ES calculation completed in {time.time() - start_time:.4f} seconds")
        return abs(es)  # Return absolute value for easier interpretation
        
    def calculate_portfolio_var(self,
                              position_values: np.ndarray,
                              covariance_matrix: np.ndarray,
                              confidence_level: Optional[float] = None) -> float:
        """
        Calculate portfolio VaR using variance-covariance method.
        
        Args:
            position_values: Array of position values
            covariance_matrix: Covariance matrix of returns
            confidence_level: Optional confidence level (defaults to instance value)
            
        Returns:
            Portfolio VaR
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        # Calculate portfolio standard deviation
        portfolio_variance = position_values.T @ covariance_matrix @ position_values
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Calculate VaR
        z_score = stats.norm.ppf(confidence_level)
        portfolio_var = portfolio_std * z_score
        
        return portfolio_var
        
    def calculate_loss_probability(self,
                                 returns: np.ndarray,
                                 threshold: float) -> float:
        """
        Calculate probability of loss exceeding the threshold.
        
        Args:
            returns: Array of historical returns
            threshold: Loss threshold (positive value)
            
        Returns:
            Probability of loss exceeding the threshold
        """
        # Remove NaN values
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return 0.0
            
        # Count observations where loss exceeds threshold
        # Note: returns are positive for gains, negative for losses
        exceed_count = np.sum(returns < -threshold)
        
        # Calculate probability
        probability = exceed_count / len(returns)
        
        return probability
        
    def calculate_trade_risk_metrics(self,
                                   trade_data: pd.DataFrame,
                                   market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk metrics for individual trades.
        
        Args:
            trade_data: DataFrame with trade information
            market_data: DataFrame with market data including returns
            
        Returns:
            DataFrame with trade risk metrics
        """
        start_time = time.time()
        
        # Prepare results dataframe
        results = trade_data.copy()
        
        # Required columns
        required_columns = ['Symbol', 'Quantity', 'Price']
        if not all(col in trade_data.columns for col in required_columns):
            logger.error(f"Trade data missing required columns: {required_columns}")
            return results
            
        # Calculate trade values
        results['Value'] = results['Quantity'] * results['Price']
        
        # Process each trade - could be parallelized for large portfolios
        if USE_PARALLEL and len(results) > 100:
            # For large portfolios, process trades in parallel
            with ThreadPoolExecutor(max_workers=min(len(results), MAX_THREADS)) as executor:
                # Split into smaller chunks
                chunk_size = max(1, len(results) // MAX_THREADS)
                chunks = [results.iloc[i:i+chunk_size] for i in range(0, len(results), chunk_size)]
                
                # Process each chunk
                processed_chunks = list(executor.map(
                    lambda chunk: self._process_trade_chunk(chunk, market_data),
                    chunks
                ))
                
                # Combine results
                results = pd.concat(processed_chunks)
        else:
            # For smaller portfolios, process sequentially
            results = self._process_trade_chunk(results, market_data)
            
        logger.info(f"Trade risk metrics calculated in {time.time() - start_time:.2f} seconds")
        return results
        
    def _process_trade_chunk(self, trade_chunk: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process a chunk of trades to calculate risk metrics.
        
        Args:
            trade_chunk: DataFrame with trade information
            market_data: DataFrame with market data
            
        Returns:
            DataFrame with added risk metrics
        """
        result_chunk = trade_chunk.copy()
        
        # Initialize risk columns
        result_chunk['VaR_95'] = 0.0
        result_chunk['ES_95'] = 0.0
        result_chunk['Loss_Prob_5pct'] = 0.0
        result_chunk['Loss_Prob_10pct'] = 0.0
        
        # Calculate metrics for each trade
        for idx, trade in result_chunk.iterrows():
            symbol = trade['Symbol']
            value = trade['Value']
            
            # Extract returns for this symbol
            if 'Symbol' in market_data.columns:
                symbol_data = market_data[market_data['Symbol'] == symbol]
                if len(symbol_data) == 0:
                    continue
                returns = symbol_data['DailyReturn'].values if 'DailyReturn' in symbol_data.columns else None
            else:
                # Assume single-symbol market data
                returns = market_data['DailyReturn'].values if 'DailyReturn' in market_data.columns else None
                
            if returns is None or len(returns) == 0:
                continue
                
            # Calculate risk metrics
            result_chunk.at[idx, 'VaR_95'] = self.calculate_var(returns, value, 'historical')
            result_chunk.at[idx, 'ES_95'] = self.calculate_expected_shortfall(returns, value, 'historical')
            result_chunk.at[idx, 'Loss_Prob_5pct'] = self.calculate_loss_probability(returns, 0.05)
            result_chunk.at[idx, 'Loss_Prob_10pct'] = self.calculate_loss_probability(returns, 0.10)
            
        return result_chunk
        
    def calculate_portfolio_metrics(self, portfolio_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict:
        """
        Calculate aggregate risk metrics for the entire portfolio.
        
        Args:
            portfolio_data: DataFrame with portfolio positions
            market_data: DataFrame with market data
            
        Returns:
            Dictionary with portfolio risk metrics
        """
        start_time = time.time()
        
        if 'Symbol' not in portfolio_data.columns or 'Value' not in portfolio_data.columns:
            logger.error("Portfolio data missing required columns: Symbol, Value")
            return {}
            
        # Get unique symbols
        symbols = portfolio_data['Symbol'].unique()
        
        # Extract returns for relevant symbols
        returns_data = {}
        for symbol in symbols:
            if 'Symbol' in market_data.columns:
                symbol_data = market_data[market_data['Symbol'] == symbol]
                if 'DailyReturn' in symbol_data.columns:
                    returns_data[symbol] = symbol_data['DailyReturn'].values
            elif symbol == market_data.get('Symbol', None):
                if 'DailyReturn' in market_data.columns:
                    returns_data[symbol] = market_data['DailyReturn'].values
        
        # Calculate correlation matrix
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr().values
        
        # Calculate covariance matrix
        covariance_matrix = returns_df.cov().values
        
        # Get position values in same order as correlation matrix
        position_values = []
        for symbol in returns_df.columns:
            position_value = portfolio_data[portfolio_data['Symbol'] == symbol]['Value'].sum()
            position_values.append(position_value)
            
        position_values = np.array(position_values)
        
        # Calculate portfolio VaR
        portfolio_var = self.calculate_portfolio_var(position_values, covariance_matrix)
        
        # Calculate diversification benefit
        individual_vars = np.array([
            self.calculate_var(returns_data[symbol], portfolio_data[portfolio_data['Symbol'] == symbol]['Value'].sum(), 'historical')
            for symbol in returns_df.columns
        ])
        sum_individual_vars = np.sum(individual_vars)
        diversification_benefit = 1 - (portfolio_var / sum_individual_vars) if sum_individual_vars > 0 else 0
        
        # Calculate other portfolio metrics
        total_value = portfolio_data['Value'].sum()
        
        # Combine all returns for portfolio level calculations
        all_returns = np.concatenate([returns for returns in returns_data.values()])
        
        portfolio_metrics = {
            'TotalValue': total_value,
            'VaR_95': portfolio_var,
            'ES_95': self.calculate_expected_shortfall(all_returns, total_value),
            'DiversificationBenefit': diversification_benefit,
            'Loss_Prob_5pct': self.calculate_loss_probability(all_returns, 0.05),
            'Loss_Prob_10pct': self.calculate_loss_probability(all_returns, 0.10),
        }
        
        logger.info(f"Portfolio metrics calculated in {time.time() - start_time:.2f} seconds")
        return portfolio_metrics
