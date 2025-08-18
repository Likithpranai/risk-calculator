"""
Monte Carlo simulation module for stress testing portfolios under various scenarios.
Optimized for performance with parallel processing capabilities.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys

# Import configuration
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config.config import SIMULATION_ITERATIONS, MAX_THREADS, USE_PARALLEL

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """
    Monte Carlo simulator for portfolio stress testing.
    Uses parallel processing for improved performance on multi-core systems.
    """
    
    def __init__(self, 
                iterations: int = SIMULATION_ITERATIONS, 
                random_seed: Optional[int] = None,
                use_parallel: bool = USE_PARALLEL,
                max_workers: int = MAX_THREADS):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            iterations: Number of simulation iterations
            random_seed: Optional random seed for reproducibility
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of worker processes
        """
        self.iterations = iterations
        self.random_seed = random_seed
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        
    def simulate_returns(self, 
                       mean: float, 
                       std: float, 
                       time_periods: int = 1, 
                       distribution: str = 'normal') -> np.ndarray:
        """
        Simulate asset returns for a single asset.
        
        Args:
            mean: Mean daily return
            std: Standard deviation of daily returns
            time_periods: Number of time periods to simulate
            distribution: Distribution type ('normal', 'student-t', 'skewed')
            
        Returns:
            Array of simulated returns
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        if distribution == 'normal':
            # Standard normal distribution
            returns = np.random.normal(mean, std, (self.iterations, time_periods))
            
        elif distribution == 'student-t':
            # Student's t-distribution (heavy tails)
            dof = 5  # Degrees of freedom (lower = fatter tails)
            returns = np.random.standard_t(dof, (self.iterations, time_periods))
            # Scale and shift to match desired mean and std
            returns = mean + std * returns * np.sqrt((dof - 2) / dof)
            
        elif distribution == 'skewed':
            # Skewed distribution using skewed normal
            from scipy.stats import skewnorm
            skewness = -2  # Negative skew (more extreme negative returns)
            returns = skewnorm.rvs(skewness, size=(self.iterations, time_periods))
            # Scale and shift to match desired mean and std
            returns = (returns - np.mean(returns)) / np.std(returns)
            returns = mean + std * returns
            
        else:
            logger.error(f"Unsupported distribution: {distribution}")
            returns = np.random.normal(mean, std, (self.iterations, time_periods))
            
        return returns
        
    def simulate_multivariate_returns(self, 
                                    means: np.ndarray, 
                                    covariance_matrix: np.ndarray,
                                    time_periods: int = 1) -> np.ndarray:
        """
        Simulate correlated returns for multiple assets.
        
        Args:
            means: Array of mean returns
            covariance_matrix: Covariance matrix
            time_periods: Number of time periods to simulate
            
        Returns:
            Array of simulated returns (shape: iterations x assets x time_periods)
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        num_assets = len(means)
        
        # Generate multivariate normal random variables
        # For each time period, we need iterations x assets random variables
        returns = np.zeros((self.iterations, num_assets, time_periods))
        
        for t in range(time_periods):
            # Simulate one time period
            period_returns = np.random.multivariate_normal(
                means, covariance_matrix, size=self.iterations
            )
            returns[:, :, t] = period_returns
            
        return returns
        
    def _simulate_scenario_batch(self, 
                             batch_id: int,
                             batch_size: int,
                             means: np.ndarray,
                             covariance_matrix: np.ndarray,
                             position_values: np.ndarray,
                             scenarios: Dict[str, Dict],
                             time_periods: int) -> Dict[str, np.ndarray]:
        """
        Simulate a batch of scenarios for parallel processing.
        
        Args:
            batch_id: Batch identifier
            batch_size: Number of simulations in this batch
            means: Array of mean returns
            covariance_matrix: Covariance matrix
            position_values: Array of position values
            scenarios: Dictionary of stress test scenarios
            time_periods: Number of time periods
            
        Returns:
            Dictionary with simulated portfolio values for each scenario
        """
        # Set different seed for each batch
        if self.random_seed is not None:
            batch_seed = self.random_seed + batch_id
            np.random.seed(batch_seed)
        
        # Initialize results
        results = {}
        for scenario_name in scenarios.keys():
            results[scenario_name] = np.zeros((batch_size, time_periods + 1))
            # Initial portfolio value at time 0
            results[scenario_name][:, 0] = np.sum(position_values)
            
        # Simulate base returns for this batch
        base_returns = np.random.multivariate_normal(
            means, covariance_matrix, size=(batch_size, time_periods)
        )
        
        # Apply scenario adjustments and calculate portfolio values
        for scenario_name, scenario_params in scenarios.items():
            # Create a copy of base returns for this scenario
            scenario_returns = base_returns.copy()
            
            # Apply mean shift if specified
            if 'mean_shift' in scenario_params:
                scenario_returns += scenario_params['mean_shift']
                
            # Apply volatility multiplier if specified
            if 'volatility_multiplier' in scenario_params:
                # Center returns around zero, multiply by volatility factor, then shift back
                centered_returns = scenario_returns - means
                scenario_returns = means + (centered_returns * scenario_params['volatility_multiplier'])
                
            # Apply correlation break if specified
            if 'correlation_break' in scenario_params:
                # For correlation break, we mix in some independent returns
                indep_returns = np.zeros_like(scenario_returns)
                for i in range(scenario_returns.shape[1]):
                    indep_returns[:, i] = np.random.normal(
                        means[i], 
                        np.sqrt(covariance_matrix[i, i]), 
                        size=batch_size
                    )
                # Mix correlated and independent returns
                mix_factor = scenario_params['correlation_break']
                scenario_returns = (1 - mix_factor) * scenario_returns + mix_factor * indep_returns
                
            # Apply specific asset shocks if specified
            if 'asset_shocks' in scenario_params:
                for asset_idx, shock in scenario_params['asset_shocks'].items():
                    scenario_returns[:, asset_idx] += shock
                    
            # Calculate portfolio values through time
            for t in range(time_periods):
                # Previous portfolio value
                prev_values = results[scenario_name][:, t]
                
                # Calculate new values based on returns at time t
                for i in range(batch_size):
                    # Asset values at time t
                    asset_returns_t = scenario_returns[i, :, t] if len(scenario_returns.shape) > 2 else scenario_returns[i, :]
                    asset_values_t = position_values * (1 + asset_returns_t)
                    
                    # New portfolio value
                    results[scenario_name][i, t + 1] = np.sum(asset_values_t)
        
        return results
        
    def run_stress_test(self,
                      position_values: np.ndarray,
                      means: np.ndarray,
                      covariance_matrix: np.ndarray,
                      scenarios: Dict[str, Dict],
                      time_periods: int = 20) -> Dict[str, Dict]:
        """
        Run Monte Carlo stress tests on a portfolio under different scenarios.
        
        Args:
            position_values: Array of position values
            means: Array of mean returns
            covariance_matrix: Covariance matrix
            scenarios: Dictionary of stress test scenarios
            time_periods: Number of time periods to simulate
            
        Returns:
            Dictionary with stress test results
        """
        start_time = time.time()
        logger.info(f"Starting Monte Carlo stress test with {self.iterations} iterations")
        
        if len(position_values) != len(means):
            logger.error("Mismatch between position_values and means dimensions")
            return {}
            
        # Initialize results dictionary
        results = {}
        for scenario_name in scenarios.keys():
            results[scenario_name] = {
                'portfolio_values': np.zeros((self.iterations, time_periods + 1)),
                'portfolio_returns': np.zeros((self.iterations, time_periods)),
                'var_95': np.zeros(time_periods),
                'es_95': np.zeros(time_periods),
                'mean_path': np.zeros(time_periods + 1),
                'lower_bound': np.zeros(time_periods + 1),
                'upper_bound': np.zeros(time_periods + 1)
            }
            
        if self.use_parallel and self.iterations >= 1000:
            # Parallel implementation for large iteration counts
            # Split work into batches
            num_workers = min(self.max_workers, os.cpu_count() or 4)
            batch_size = self.iterations // num_workers
            
            logger.info(f"Using {num_workers} workers with batch size {batch_size}")
            
            # Create batches
            batches = []
            remaining = self.iterations
            batch_id = 0
            
            while remaining > 0:
                current_batch_size = min(batch_size, remaining)
                batches.append((batch_id, current_batch_size))
                remaining -= current_batch_size
                batch_id += 1
                
            # Process batches in parallel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        self._simulate_scenario_batch,
                        batch_id,
                        batch_size,
                        means,
                        covariance_matrix,
                        position_values,
                        scenarios,
                        time_periods
                    )
                    for batch_id, batch_size in batches
                ]
                
                # Collect results
                batch_results = []
                for future in as_completed(futures):
                    try:
                        batch_result = future.result()
                        batch_results.append(batch_result)
                    except Exception as e:
                        logger.error(f"Error in worker process: {e}")
                        
                # Combine batch results
                for scenario_name in scenarios.keys():
                    scenario_values = np.vstack([
                        batch_result[scenario_name] for batch_result in batch_results
                    ])
                    results[scenario_name]['portfolio_values'] = scenario_values
        else:
            # Sequential implementation for smaller iteration counts
            # Initial portfolio value
            initial_value = np.sum(position_values)
            
            for scenario_name, scenario_params in scenarios.items():
                # Simulate multivariate returns
                adjusted_means = means.copy()
                adjusted_cov = covariance_matrix.copy()
                
                # Apply scenario adjustments to mean and covariance
                if 'mean_shift' in scenario_params:
                    adjusted_means += scenario_params['mean_shift']
                    
                if 'volatility_multiplier' in scenario_params:
                    # Scale diagonal of covariance matrix
                    for i in range(len(adjusted_means)):
                        adjusted_cov[i, i] *= scenario_params['volatility_multiplier']**2
                        
                # Simulate returns with adjustments
                returns = self.simulate_multivariate_returns(
                    adjusted_means, adjusted_cov, time_periods
                )
                
                # Calculate portfolio values through time
                results[scenario_name]['portfolio_values'][:, 0] = initial_value
                
                for t in range(time_periods):
                    # Asset values at time t
                    asset_returns_t = returns[:, :, t]
                    asset_values_t = position_values * (1 + asset_returns_t)
                    
                    # New portfolio value
                    for i in range(self.iterations):
                        results[scenario_name]['portfolio_values'][i, t + 1] = np.sum(asset_values_t[i])
                        
                    # Calculate portfolio returns
                    results[scenario_name]['portfolio_returns'][:, t] = (
                        results[scenario_name]['portfolio_values'][:, t + 1] / 
                        results[scenario_name]['portfolio_values'][:, t] - 1
                    )
        
        # Calculate statistics for each scenario
        for scenario_name in scenarios.keys():
            portfolio_values = results[scenario_name]['portfolio_values']
            
            # Calculate portfolio returns from values
            for t in range(time_periods):
                prev_values = portfolio_values[:, t]
                curr_values = portfolio_values[:, t + 1]
                results[scenario_name]['portfolio_returns'][:, t] = curr_values / prev_values - 1
                
                # Calculate VaR and ES at each time step
                returns_t = results[scenario_name]['portfolio_returns'][:, t]
                results[scenario_name]['var_95'][t] = np.percentile(returns_t, 5)
                results[scenario_name]['es_95'][t] = returns_t[returns_t <= results[scenario_name]['var_95'][t]].mean()
                
            # Calculate path statistics
            results[scenario_name]['mean_path'] = np.mean(portfolio_values, axis=0)
            results[scenario_name]['lower_bound'] = np.percentile(portfolio_values, 5, axis=0)
            results[scenario_name]['upper_bound'] = np.percentile(portfolio_values, 95, axis=0)
            
            # Calculate final summary statistics
            final_values = portfolio_values[:, -1]
            total_returns = final_values / portfolio_values[:, 0] - 1
            
            results[scenario_name]['summary'] = {
                'mean_return': np.mean(total_returns),
                'median_return': np.median(total_returns),
                'std_return': np.std(total_returns),
                'var_95': np.percentile(total_returns, 5),
                'es_95': total_returns[total_returns <= np.percentile(total_returns, 5)].mean(),
                'max_loss': np.min(total_returns),
                'max_gain': np.max(total_returns),
                'prob_loss': np.mean(total_returns < 0)
            }
        
        logger.info(f"Monte Carlo stress test completed in {time.time() - start_time:.2f} seconds")
        return results
        
    def define_standard_scenarios(self) -> Dict[str, Dict]:
        """
        Define standard stress test scenarios.
        
        Returns:
            Dictionary with standard scenario definitions
        """
        scenarios = {
            'base_case': {
                'description': 'Base case scenario with historical parameters',
                'mean_shift': 0,
                'volatility_multiplier': 1
            },
            'market_crash': {
                'description': 'Severe market crash scenario',
                'mean_shift': -0.03,  # 3% daily loss
                'volatility_multiplier': 2.5  # 2.5x volatility
            },
            'volatility_spike': {
                'description': 'Volatility spike without directional shift',
                'mean_shift': 0,
                'volatility_multiplier': 3.0  # 3x volatility
            },
            'correlation_breakdown': {
                'description': 'Breakdown in asset correlations during crisis',
                'mean_shift': -0.01,  # 1% daily loss
                'volatility_multiplier': 2.0,  # 2x volatility
                'correlation_break': 0.5  # 50% correlation reduction
            },
            'stagflation': {
                'description': 'Stagflation scenario: higher inflation, lower growth',
                'mean_shift': -0.005,  # 0.5% daily loss
                'volatility_multiplier': 1.5,  # 1.5x volatility
                'asset_shocks': {
                    0: -0.02,  # Stocks down 2%
                    1: -0.015,  # Corporate bonds down 1.5%
                    2: 0.01,  # Commodities up 1%
                    3: -0.01   # Real estate down 1%
                }
            },
            'liquidity_crisis': {
                'description': 'Liquidity crisis with correlation breakdown',
                'mean_shift': -0.02,  # 2% daily loss
                'volatility_multiplier': 2.0,  # 2x volatility
                'correlation_break': 0.7,  # 70% correlation reduction
                'asset_shocks': {
                    0: -0.03,  # Stocks down 3%
                    1: -0.03,  # Corporate bonds down 3%
                    2: 0.0,    # Treasuries unchanged
                    3: -0.04   # High yield down 4%
                }
            }
        }
        
        return scenarios
