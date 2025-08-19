import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys
import joblib
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config.config import ML_MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskPredictor:
    def __init__(self, model_type: str = 'random_forest', use_sentiment: bool = True):
        self.model_type = model_type
        self.use_sentiment = use_sentiment
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_cols = []
        self.sentiment_cols = ['compound', 'positive', 'negative'] if use_sentiment else []
        
    def prepare_features(self, market_data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None,
                        lookback_periods: int = 10) -> pd.DataFrame:
        start_time = time.time()
        
        if 'Symbol' in market_data.columns:
            symbols = market_data['Symbol'].unique()
            all_features = []
            
            for symbol in symbols:
                symbol_data = market_data[market_data['Symbol'] == symbol].copy()
                symbol_features = self._create_features(symbol_data, lookback_periods)
                
                if self.use_sentiment and sentiment_data is not None:
                    symbol_sentiment = sentiment_data[sentiment_data['symbol'] == symbol].copy()
                    if not symbol_sentiment.empty:
                        symbol_features = self._add_sentiment_features(symbol_features, symbol_sentiment)
                
                all_features.append(symbol_features)
                
            features = pd.concat(all_features)
        else:
            features = self._create_features(market_data.copy(), lookback_periods)
            
            if self.use_sentiment and sentiment_data is not None:
                features = self._add_sentiment_features(features, sentiment_data)
        
        logger.info(f"Features prepared in {time.time() - start_time:.2f} seconds")
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate the Relative Strength Index (RSI) for a price series.
        
        Args:
            prices: Series of price values
            window: The RSI window period (default: 14)
            
        Returns:
            Series containing RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and average loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss.replace(0, float('1e-9'))  # Avoid division by zero
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _create_features(self, df: pd.DataFrame, lookback_periods: int = 10) -> pd.DataFrame:
        df = df.sort_values('Date')
        
        feature_df = pd.DataFrame(index=df.index)
        feature_df['Date'] = df['Date']
        
        if 'Symbol' in df.columns:
            feature_df['Symbol'] = df['Symbol']
        
        if 'Close' in df.columns:
            for i in range(1, lookback_periods + 1):
                feature_df[f'Return_Lag_{i}'] = df['DailyReturn'].shift(i)
                feature_df[f'LogReturn_Lag_{i}'] = df['LogReturn'].shift(i)
                
            feature_df['MA_5'] = df['Close'].rolling(window=5).mean() / df['Close'] - 1
            feature_df['MA_10'] = df['Close'].rolling(window=10).mean() / df['Close'] - 1
            feature_df['MA_20'] = df['Close'].rolling(window=20).mean() / df['Close'] - 1
            
            feature_df['Vol_5'] = df['DailyReturn'].rolling(window=5).std()
            feature_df['Vol_10'] = df['DailyReturn'].rolling(window=10).std()
            feature_df['Vol_20'] = df['DailyReturn'].rolling(window=20).std()
            
            feature_df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            feature_df['RSI'] = self._calculate_rsi(df['Close'])
            
            feature_df['TargetVolatility'] = df['DailyReturn'].rolling(window=20).std().shift(-20)
        
        if all(x in df.columns for x in ['High', 'Low', 'Close']):
            feature_df['HL_Ratio'] = (df['High'] - df['Low']) / df['Close']
            feature_df['HL_Ratio_MA5'] = feature_df['HL_Ratio'].rolling(window=5).mean()
            
        feature_df['Day_of_Week'] = df['Date'].dt.dayofweek
        feature_df['Month'] = df['Date'].dt.month
        
        feature_df = feature_df.dropna()
        return feature_df
    
    def _add_sentiment_features(self, feature_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        sentiment_df = sentiment_df.copy()
        
        if 'date' in sentiment_df.columns:
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            sentiment_df = sentiment_df.rename(columns={'date': 'Date'})
            
        feature_df = pd.merge_asof(
            feature_df.sort_values('Date'),
            sentiment_df.sort_values('Date')[['Date'] + self.sentiment_cols],
            on='Date',
            direction='backward'
        )
        
        return feature_df
    
    def train(self, features: pd.DataFrame, target_col: str, test_size: float = 0.2, 
             time_series_split: bool = True, random_state: int = 42) -> Dict:
        start_time = time.time()
        
        self.target_col = target_col
        
        X = features.drop(['Date', target_col], axis=1)
        if 'Symbol' in X.columns:
            X = X.drop('Symbol', axis=1)
        y = features[target_col]
        
        self.feature_cols = X.columns.tolist()
        
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        if time_series_split:
            tscv = TimeSeriesSplit(n_splits=5)
            train_indices, test_indices = [], []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                train_indices = train_idx
                test_indices = test_idx
                
            X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
            y_train, y_test = y_scaled[train_indices], y_scaled[test_indices]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=test_size, random_state=random_state
            )
        
        if self.model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1)
        elif self.model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=random_state)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1)
        
        model.fit(X_train, y_train)
        self.model = model
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_cols, model.feature_importances_))
            sorted_importance = {k: v for k, v in sorted(
                feature_importance.items(), key=lambda item: item[1], reverse=True
            )}
        else:
            sorted_importance = {}
        
        train_time = time.time() - start_time
        logger.info(f"Model trained in {train_time:.2f} seconds")
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': sorted_importance,
            'training_time': train_time
        }
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            logger.error("Model not trained, cannot make predictions")
            return np.array([])
        
        X = features.copy()
        
        if 'Date' in X.columns:
            X = X.drop('Date', axis=1)
        if 'Symbol' in X.columns:
            X = X.drop('Symbol', axis=1)
        if self.target_col in X.columns:
            X = X.drop(self.target_col, axis=1)
        
        missing_cols = set(self.feature_cols) - set(X.columns)
        if missing_cols:
            logger.error(f"Missing features: {missing_cols}")
            return np.array([])
        
        X = X[self.feature_cols]
        
        X_scaled = self.feature_scaler.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred
    
    def predict_volatility(self, market_data: pd.DataFrame, 
                         horizon: int = 10, 
                         sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if 'Symbol' not in market_data.columns:
            market_data['Symbol'] = 'MARKET'
        
        features = self.prepare_features(market_data, sentiment_data)
        
        X = features.copy()
        if self.target_col in X.columns:
            X = X.drop(self.target_col, axis=1)
        
        symbols = X['Symbol'].unique()
        predictions = []
        
        for symbol in symbols:
            symbol_features = X[X['Symbol'] == symbol].sort_values('Date')
            symbol_dates = symbol_features['Date']
            
            if len(symbol_features) < horizon:
                continue
            
            symbol_pred = self.predict(symbol_features)
            
            pred_df = pd.DataFrame({
                'Symbol': symbol,
                'Date': symbol_dates,
                'PredictedVolatility': symbol_pred
            })
            
            predictions.append(pred_df)
        
        if not predictions:
            return pd.DataFrame()
        
        return pd.concat(predictions)
    
    def predict_var_es(self, market_data: pd.DataFrame, 
                     position_values: Dict[str, float],
                     confidence_level: float = 0.95,
                     horizon: int = 10,
                     sentiment_data: Optional[pd.DataFrame] = None) -> Dict:
        pred_vol = self.predict_volatility(market_data, horizon, sentiment_data)
        
        if pred_vol.empty:
            logger.error("No volatility predictions available")
            return {}
        
        latest_predictions = pred_vol.groupby('Symbol').last().reset_index()
        
        results = {}
        total_value = sum(position_values.values())
        
        for symbol, value in position_values.items():
            symbol_pred = latest_predictions[latest_predictions['Symbol'] == symbol]
            
            if symbol_pred.empty:
                continue
            
            pred_volatility = symbol_pred['PredictedVolatility'].values[0]
            
            z_score = np.abs(np.percentile(np.random.normal(0, 1, 10000), 100 * (1 - confidence_level)))
            
            var = value * pred_volatility * z_score * np.sqrt(horizon)
            es = value * pred_volatility * np.exp(0.5) / (1 - confidence_level) * z_score * np.sqrt(horizon)
            
            weight = value / total_value
            
            results[symbol] = {
                'Value': value,
                'Weight': weight,
                'PredictedVolatility': pred_volatility,
                'VaR': var,
                'ES': es
            }
        
        if not results:
            return {}
        
        portfolio_var = sum(asset['VaR'] for asset in results.values())
        portfolio_es = sum(asset['ES'] for asset in results.values())
        
        results['Portfolio'] = {
            'Value': total_value,
            'Weight': 1.0,
            'VaR': portfolio_var,
            'ES': portfolio_es
        }
        
        return results
    
    def save_model(self, filename: Optional[str] = None):
        if self.model is None:
            logger.error("No model to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"risk_model_{self.model_type}_{timestamp}.pkl"
        
        os.makedirs(ML_MODEL_PATH, exist_ok=True)
        model_path = os.path.join(ML_MODEL_PATH, filename)
        
        model_data = {
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'feature_cols': self.feature_cols,
            'sentiment_cols': self.sentiment_cols,
            'model_type': self.model_type,
            'use_sentiment': self.use_sentiment
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filename: str):
        model_path = os.path.join(ML_MODEL_PATH, filename)
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.feature_scaler = model_data['feature_scaler']
            self.target_scaler = model_data['target_scaler']
            self.feature_cols = model_data['feature_cols']
            self.sentiment_cols = model_data['sentiment_cols']
            self.model_type = model_data['model_type']
            self.use_sentiment = model_data['use_sentiment']
            
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
