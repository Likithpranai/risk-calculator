
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config.config import DEFAULT_CONFIDENCE_LEVEL, USE_PARALLEL, MAX_THREADS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskCalculator:
    def __init__(self, confidence_level: float = DEFAULT_CONFIDENCE_LEVEL):
        self.confidence_level = confidence_level
        
    def calculate_var(self, 
                     returns: np.ndarray, 
                     value: float,
                     method: str = 'historical',
                     time_horizon: int = 1) -> float:
        start_time = time.time()
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            logger.warning("Empty returns array provided for VaR calculation")
            return 0.0
            
        if method == 'historical':

            var = np.percentile(returns, 100 * (1 - self.confidence_level)) * value * np.sqrt(time_horizon)
            
        elif method == 'parametric':

            mean = np.mean(returns)
            std = np.std(returns)
            var = -1 * (mean + std * stats.norm.ppf(self.confidence_level)) * value * np.sqrt(time_horizon)
            
        elif method == 'monte_carlo':

            logger.warning("Monte Carlo VaR requested but using simpler method - use stress_test for MC simulations")

            mean = np.mean(returns)
            std = np.std(returns)
            var = -1 * (mean + std * stats.norm.ppf(self.confidence_level)) * value * np.sqrt(time_horizon)
            
        else:
            logger.error(f"Unsupported VaR method: {method}")
            return 0.0
            
        logger.info(f"VaR calculation completed in {time.time() - start_time:.4f} seconds")
        return abs(var)
        
    def calculate_expected_shortfall(self,
                                   returns: np.ndarray,
                                   value: float,
                                   method: str = 'historical',
                                   time_horizon: int = 1) -> float:
        start_time = time.time()
        

        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            logger.warning("Empty returns array provided for ES calculation")
            return 0.0
            
        if method == 'historical':
            var_cutoff = np.percentile(returns, 100 * (1 - self.confidence_level))
            es = returns[returns <= var_cutoff].mean() * value * np.sqrt(time_horizon)
            
        elif method == 'parametric':

            mean = np.mean(returns)
            std = np.std(returns)
            var_quantile = stats.norm.ppf(1 - self.confidence_level)
            es = -1 * (mean + std * stats.norm.pdf(var_quantile) / (1 - self.confidence_level)) * value * np.sqrt(time_horizon)
            
        else:
            logger.error(f"Unsupported ES method: {method}")
            return 0.0
            
        logger.info(f"ES calculation completed in {time.time() - start_time:.4f} seconds")
        return abs(es)
        
    def calculate_portfolio_var(self,
                              position_values: np.ndarray,
                              covariance_matrix: np.ndarray,
                              confidence_level: Optional[float] = None) -> float:
        if confidence_level is None:
            confidence_level = self.confidence_level
            

        portfolio_variance = position_values.T @ covariance_matrix @ position_values
        portfolio_std = np.sqrt(portfolio_variance)
        

        z_score = stats.norm.ppf(confidence_level)
        portfolio_var = portfolio_std * z_score
        
        return portfolio_var
        
    def calculate_loss_probability(self,
                                 returns: np.ndarray,
                                 threshold: float) -> float:

        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return 0.0
            

        exceed_count = np.sum(returns < -threshold)
        

        probability = exceed_count / len(returns)
        
        return probability
        
    def calculate_trade_risk_metrics(self,
                                   trade_data: pd.DataFrame,
                                   market_data: pd.DataFrame) -> pd.DataFrame:

        start_time = time.time()
        results = trade_data.copy()
        

        required_columns = ['Symbol', 'Quantity', 'Price']
        if not all(col in trade_data.columns for col in required_columns):
            logger.error(f"Trade data missing required columns: {required_columns}")
            return results
            

        results['Value'] = results['Quantity'] * results['Price']
        

        if USE_PARALLEL and len(results) > 100:

            with ThreadPoolExecutor(max_workers=min(len(results), MAX_THREADS)) as executor:

                chunk_size = max(1, len(results) // MAX_THREADS)
                chunks = [results.iloc[i:i+chunk_size] for i in range(0, len(results), chunk_size)]
                
                processed_chunks = list(executor.map(
                    lambda chunk: self._process_trade_chunk(chunk, market_data),
                    chunks
                ))
                
                results = pd.concat(processed_chunks)
        else:

            results = self._process_trade_chunk(results, market_data)
            
        logger.info(f"Trade risk metrics calculated in {time.time() - start_time:.2f} seconds")
        return results
        
    def _process_trade_chunk(self, trade_chunk: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        result_chunk = trade_chunk.copy()
        
        result_chunk['VaR_95'] = 0.0
        result_chunk['ES_95'] = 0.0
        result_chunk['Loss_Prob_5pct'] = 0.0
        result_chunk['Loss_Prob_10pct'] = 0.0
        
        for idx, trade in result_chunk.iterrows():
            symbol = trade['Symbol']
            value = trade['Value']
        
            if 'Symbol' in market_data.columns:
                symbol_data = market_data[market_data['Symbol'] == symbol]
                if len(symbol_data) == 0:
                    continue
                returns = symbol_data['DailyReturn'].values if 'DailyReturn' in symbol_data.columns else None
            else:

                returns = market_data['DailyReturn'].values if 'DailyReturn' in market_data.columns else None
                
            if returns is None or len(returns) == 0:
                continue
                

            result_chunk.at[idx, 'VaR_95'] = self.calculate_var(returns, value, 'historical')
            result_chunk.at[idx, 'ES_95'] = self.calculate_expected_shortfall(returns, value, 'historical')
            result_chunk.at[idx, 'Loss_Prob_5pct'] = self.calculate_loss_probability(returns, 0.05)
            result_chunk.at[idx, 'Loss_Prob_10pct'] = self.calculate_loss_probability(returns, 0.10)
            
        return result_chunk
        
    def calculate_portfolio_metrics(self, portfolio_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict:
        start_time = time.time()
        
        if 'Symbol' not in portfolio_data.columns or 'Value' not in portfolio_data.columns:
            logger.error("Portfolio data missing required columns: Symbol, Value")
            return {}
            

        symbols = portfolio_data['Symbol'].unique()
        

        returns_data = {}
        for symbol in symbols:
            if 'Symbol' in market_data.columns:
                symbol_data = market_data[market_data['Symbol'] == symbol]
                if 'DailyReturn' in symbol_data.columns:
                    returns_data[symbol] = symbol_data['DailyReturn'].values
            elif symbol == market_data.get('Symbol', None):
                if 'DailyReturn' in market_data.columns:
                    returns_data[symbol] = market_data['DailyReturn'].values
        
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr().values
        covariance_matrix = returns_df.cov().values
        position_values = []
        for symbol in returns_df.columns:
            position_value = portfolio_data[portfolio_data['Symbol'] == symbol]['Value'].sum()
            position_values.append(position_value)
            
        position_values = np.array(position_values)
        

        portfolio_var = self.calculate_portfolio_var(position_values, covariance_matrix)
        

        individual_vars = np.array([
            self.calculate_var(returns_data[symbol], portfolio_data[portfolio_data['Symbol'] == symbol]['Value'].sum(), 'historical')
            for symbol in returns_df.columns
        ])
        sum_individual_vars = np.sum(individual_vars)
        diversification_benefit = 1 - (portfolio_var / sum_individual_vars) if sum_individual_vars > 0 else 0
        

        total_value = portfolio_data['Value'].sum()
        

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
