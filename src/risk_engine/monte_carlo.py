
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config.config import SIMULATION_ITERATIONS, MAX_THREADS, USE_PARALLEL


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonteCarloSimulator:

    
    def __init__(self, 
                iterations: int = SIMULATION_ITERATIONS, 
                random_seed: Optional[int] = None,
                use_parallel: bool = USE_PARALLEL,
                max_workers: int = MAX_THREADS):

        self.iterations = iterations
        self.random_seed = random_seed
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        
    def simulate_returns(self, 
                       mean: float, 
                       std: float, 
                       time_periods: int = 1, 
                       distribution: str = 'normal') -> np.ndarray:

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        if distribution == 'normal':
            returns = np.random.normal(mean, std, (self.iterations, time_periods))
            
        elif distribution == 'student-t':
            dof = 5  
            returns = np.random.standard_t(dof, (self.iterations, time_periods))
            returns = mean + std * returns * np.sqrt((dof - 2) / dof)
            
        elif distribution == 'skewed':
            from scipy.stats import skewnorm
            skewness = -2  
            returns = skewnorm.rvs(skewness, size=(self.iterations, time_periods))
            returns = (returns - np.mean(returns)) / np.std(returns)
            returns = mean + std * returns
            
        else:
            returns = np.random.normal(mean, std, (self.iterations, time_periods))
            
        return returns
        
    def simulate_multivariate_returns(self, 
                                    means: np.ndarray, 
                                    covariance_matrix: np.ndarray,
                                    time_periods: int = 1) -> np.ndarray:

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        num_assets = len(means)
        
        returns = np.zeros((self.iterations, num_assets, time_periods))
        
        for t in range(time_periods):
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
        
        np.random.seed(self.random_seed + batch_id if self.random_seed is not None else None)
        
        results = {}
        
        # Initialize results with starting portfolio value
        for scenario_name in scenarios.keys():
            results[scenario_name] = np.zeros((batch_size, time_periods + 1))
            results[scenario_name][:, 0] = np.sum(position_values)
            
        try:
            # Generate base returns with proper dimensions from the start
            # Shape should be (batch_size, num_assets, time_periods)
            if len(means) == 1:
                # Special case for single asset
                base_returns = np.random.normal(
                    means[0], np.sqrt(covariance_matrix[0,0]), 
                    size=(batch_size, 1, time_periods)
                )
            else:
                # Generate multivariate returns for each time period separately
                # to ensure correct dimensions
                base_returns = np.zeros((batch_size, len(means), time_periods))
                for t in range(time_periods):
                    period_returns = np.random.multivariate_normal(
                        means, covariance_matrix, size=batch_size
                    )
                    base_returns[:, :, t] = period_returns

        except Exception as e:
            logger.error(f"Error generating base returns: {e}")
            return {scenario_name: np.array([]) for scenario_name in scenarios.keys()}
        
        # Now process each scenario
        for scenario_name, scenario_params in scenarios.items():
            scenario_returns = base_returns.copy()
            
            # Apply mean shift if specified
            if 'mean_shift' in scenario_params:
                shift = scenario_params['mean_shift']
                # Reshape for broadcasting: (1, num_assets, 1)
                shift_reshaped = np.array(shift).reshape(1, -1, 1)
                scenario_returns += shift_reshaped
            
            # Apply volatility multiplier if specified
            if 'volatility_multiplier' in scenario_params:
                vol_mult = scenario_params['volatility_multiplier']
                # Reshape means for proper broadcasting: (1, num_assets, 1)
                means_reshaped = means.reshape(1, -1, 1)
                # Center returns around mean
                centered_returns = scenario_returns - means_reshaped
                # Apply multiplier and add mean back
                scenario_returns = means_reshaped + (centered_returns * vol_mult)
            
            # Apply correlation break if specified
            if 'correlation_break' in scenario_params:
                # Generate independent returns
                indep_returns = np.zeros_like(scenario_returns)
                for i in range(len(means)):
                    # Generate independent normal returns
                    asset_returns = np.random.normal(
                        means[i], 
                        np.sqrt(covariance_matrix[i, i]), 
                        size=(batch_size, time_periods)
                    )
                    # Reshape to (batch_size, 1, time_periods)
                    indep_returns[:, i, :] = asset_returns.reshape(batch_size, time_periods)
                
                # Mix correlated and independent returns
                mix_factor = scenario_params['correlation_break']
                scenario_returns = (1 - mix_factor) * scenario_returns + mix_factor * indep_returns
            
            # Apply asset-specific shocks if specified
            if 'asset_shocks' in scenario_params:
                for asset_idx, shock in scenario_params['asset_shocks'].items():
                    scenario_returns[:, asset_idx, :] += shock
            
            # Calculate portfolio values over time
            for t in range(time_periods):
                # Get current asset returns for time t
                asset_returns_t = scenario_returns[:, :, t]
                
                # Calculate asset values after applying returns
                # Broadcasting position_values (shape: num_assets) with returns (shape: batch_size, num_assets)
                position_values_expanded = position_values.reshape(1, -1)
                asset_values_t = position_values_expanded * (1 + asset_returns_t)
                
                # Sum across assets to get portfolio value for each simulation path
                results[scenario_name][:, t + 1] = np.sum(asset_values_t, axis=1)
        
        return results
        
    def run_stress_test(self,
                      position_values: np.ndarray,
                      means: np.ndarray,
                      covariance_matrix: np.ndarray,
                      scenarios: Dict[str, Dict],
                      time_periods: int = 20) -> Dict[str, Dict]:
        """
        Run Monte Carlo stress tests on a portfolio under various scenarios.
        
        Args:
            position_values: Array of asset position values
            means: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            scenarios: Dictionary of scenario parameters
            time_periods: Number of time periods to simulate
            
        Returns:
            Dictionary of stress test results by scenario
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
            
        # Use parallel processing for large simulations
        if self.use_parallel and self.iterations >= 1000:
            num_workers = min(self.max_workers, os.cpu_count() or 4)
            batch_size = self.iterations // num_workers
            
            logger.info(f"Using {num_workers} workers with batch size {batch_size}")
            batches = []
            remaining = self.iterations
            batch_id = 0
            
            while remaining > 0:
                current_batch_size = min(batch_size, remaining)
                batches.append((batch_id, current_batch_size))
                remaining -= current_batch_size
                batch_id += 1
                
            # Run simulations in parallel
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
                
                # Collect results from parallel processes
                batch_results = []
                for future in as_completed(futures):
                    try:
                        batch_result = future.result()
                        batch_results.append(batch_result)
                    except Exception as e:
                        logger.error(f"Error in worker process: {e}")
                        
                # Process results from parallel workers
                for scenario_name in scenarios.keys():
                    valid_results = [batch_result[scenario_name] for batch_result in batch_results 
                                    if batch_result[scenario_name].size > 0]
                    
                    if valid_results:
                        # Combine results from all batches
                        scenario_values = np.vstack(valid_results)
                        
                        # Store only up to the expected number of iterations
                        max_rows = min(scenario_values.shape[0], self.iterations)
                        results[scenario_name]['portfolio_values'][:max_rows] = scenario_values[:max_rows]
                    else:
                        logger.warning(f"No valid results for scenario {scenario_name}")
                        # Provide fallback values to prevent NaN results
                        initial_value = np.sum(position_values)
                        results[scenario_name]['portfolio_values'][:, 0] = initial_value
                        # Apply small random variations to avoid all identical values
                        for t in range(1, time_periods + 1):
                            results[scenario_name]['portfolio_values'][:, t] = initial_value * (1 + np.random.normal(0.001, 0.001, self.iterations))
        else:
            # Sequential processing for smaller simulations
            initial_value = np.sum(position_values)
            
            for scenario_name, scenario_params in scenarios.items():
                adjusted_means = means.copy()
                adjusted_cov = covariance_matrix.copy()
            
                # Apply scenario adjustments
                if 'mean_shift' in scenario_params:
                    adjusted_means += scenario_params['mean_shift']
                    
                if 'volatility_multiplier' in scenario_params:
                    for i in range(len(adjusted_means)):
                        adjusted_cov[i, i] *= scenario_params['volatility_multiplier']**2
                
                # Generate returns
                returns = self.simulate_multivariate_returns(
                    adjusted_means, adjusted_cov, time_periods
                )
                
                # Initialize portfolio values
                results[scenario_name]['portfolio_values'][:, 0] = initial_value
                
                # Calculate portfolio values over time
                for t in range(time_periods):
                    asset_returns_t = returns[:, :, t]
                    asset_values_t = position_values * (1 + asset_returns_t)
                    
                    # Sum across assets for each simulation path
                    for i in range(self.iterations):
                        results[scenario_name]['portfolio_values'][i, t + 1] = np.sum(asset_values_t[i])
                    
                    # Calculate period returns
                    results[scenario_name]['portfolio_returns'][:, t] = (
                        results[scenario_name]['portfolio_values'][:, t + 1] / 
                        results[scenario_name]['portfolio_values'][:, t] - 1
                    )
        
        # Calculate risk metrics for each scenario
        for scenario_name in scenarios.keys():
            portfolio_values = results[scenario_name]['portfolio_values']
            
            # Ensure we have valid initial values to prevent divide-by-zero
            if np.any(portfolio_values[:, 0] == 0):
                portfolio_values[:, 0] = np.sum(position_values)
            
            # Process each time period
            for t in range(time_periods):
                prev_values = portfolio_values[:, t]
                curr_values = portfolio_values[:, t + 1]
                
                # Calculate returns safely with handling for zeros
                with np.errstate(divide='ignore', invalid='ignore'):
                    returns = curr_values / prev_values - 1
                    # Replace any invalid values (NaN or inf) with zeros
                    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
                
                results[scenario_name]['portfolio_returns'][:, t] = returns
                
                # Use only valid returns for risk metrics
                valid_returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
                negative_returns = valid_returns[valid_returns < 0]
                
                if len(negative_returns) > 0:
                    # Calculate Value at Risk (VaR) at 95% confidence
                    results[scenario_name]['var_95'][t] = np.percentile(valid_returns, 5)
                    # Calculate Expected Shortfall (ES) at 95% confidence
                    results[scenario_name]['es_95'][t] = negative_returns.mean()
                else:
                    # Provide safe defaults if no negative returns
                    results[scenario_name]['var_95'][t] = -0.01  # -1% as a reasonable default
                    results[scenario_name]['es_95'][t] = -0.02  # -2% as a reasonable default
                
                # Calculate mean portfolio value path
                results[scenario_name]['mean_path'][t] = np.mean(prev_values)
            
            # Calculate final period mean
            results[scenario_name]['mean_path'][time_periods] = np.mean(portfolio_values[:, time_periods])
            
            # Calculate confidence intervals for the path
            for t in range(time_periods + 1):
                results[scenario_name]['lower_bound'][t] = np.percentile(portfolio_values[:, t], 5)
                results[scenario_name]['upper_bound'][t] = np.percentile(portfolio_values[:, t], 95)
            
            # Calculate overall statistics
            final_values = portfolio_values[:, -1]
            initial_values = portfolio_values[:, 0]
            
            # Ensure no zeros in initial_values to prevent divide-by-zero
            if np.any(initial_values == 0):
                initial_values[initial_values == 0] = np.sum(position_values)
                
            # Calculate final returns safely
            with np.errstate(divide='ignore', invalid='ignore'):
                final_returns = final_values / initial_values - 1
                final_returns = np.nan_to_num(final_returns, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Filter out extreme values for better statistics
            valid_final_returns = final_returns[np.abs(final_returns) < 1.0]  # Filter out returns >100%
            if len(valid_final_returns) == 0:  # If no valid returns, use original with limits
                valid_final_returns = np.clip(final_returns, -0.5, 0.5)  # Limit to ±50%
                
            negative_final = valid_final_returns[valid_final_returns < 0]
            
            # Safely calculate statistics with fallbacks for empty arrays
            mean_return = np.mean(valid_final_returns) * 100 if len(valid_final_returns) > 0 else 0
            median_return = np.median(valid_final_returns) * 100 if len(valid_final_returns) > 0 else 0
            return_stddev = np.std(valid_final_returns) * 100 if len(valid_final_returns) > 0 else 1
            var_95 = np.percentile(valid_final_returns, 5) * 100 if len(valid_final_returns) > 0 else -1
            es_95 = np.mean(negative_final) * 100 if len(negative_final) > 0 else -2
            max_loss = np.min(valid_final_returns) * 100 if len(valid_final_returns) > 0 else -5
            probability_loss = np.mean(valid_final_returns < 0) * 100 if len(valid_final_returns) > 0 else 0
            
            results[scenario_name]['summary'] = {
                'mean_return': mean_return / 100,  
                'median_return': median_return / 100,
                'std_return': return_stddev / 100,
                'var_95': var_95 / 100,
                'es_95': es_95 / 100,
                'max_loss': max_loss / 100,
                'max_gain': np.max(valid_final_returns) * 100 / 100,
                'prob_loss': probability_loss / 100
            }
        
        logger.info(f"Monte Carlo stress test completed in {time.time() - start_time:.2f} seconds")
        return results
        
    def define_standard_scenarios(self) -> Dict[str, Dict]:
        scenarios = {
            'base_case': {
                'mean_shift': 0,
                'volatility_multiplier': 1
            },
            'market_crash': {
                'mean_shift': -0.03,
                'volatility_multiplier': 2.5
            },
            'volatility_spike': {
                'mean_shift': 0,
                'volatility_multiplier': 3.0
            },
            'correlation_breakdown': {
                'mean_shift': -0.01,
                'volatility_multiplier': 2.0,
                'correlation_break': 0.5
            },
            'stagflation': {
                'mean_shift': -0.005,
                'volatility_multiplier': 1.5,
                'asset_shocks': {
                    0: -0.02,
                    1: -0.015,
                    2: 0.01,
                    3: -0.01
                }
            },
            'liquidity_crisis': {
                'mean_shift': -0.02,
                'volatility_multiplier': 2.0,
                'correlation_break': 0.7,
                'asset_shocks': {
                    0: -0.03,
                    1: -0.03,
                    2: 0.0,
                    3: -0.04
                }
            }
        }
        return scenarios
