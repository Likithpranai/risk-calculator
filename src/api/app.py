import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, Query, Path, Body, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import bcrypt
import uvicorn
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from config.config import JWT_SECRET, CORS_ORIGINS, JWT_ALGORITHM, JWT_EXPIRATION_SECONDS
from config.config import RATE_LIMIT_ENABLED, RATE_LIMIT_REQUESTS, RATE_LIMIT_PERIOD
from src.data.data_loader import DataLoader
from src.risk_engine.risk_calculator import RiskCalculator
from src.risk_engine.monte_carlo import MonteCarloSimulator
from src.models.risk_predictor import RiskPredictor
from src.sentiment.sentiment_analyzer import SentimentAnalyzer
from src.sentiment.real_time_feed import RealTimeSentimentFeed
from src.reports.report_generator import ReportGenerator

app = FastAPI(title="AI-Driven Trade Risk Assessment API",
             description="API for trade-level risk assessment, portfolio stress testing, and sentiment analysis",
             version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
data_loader = DataLoader()
risk_calculator = RiskCalculator()
monte_carlo = MonteCarloSimulator()
risk_predictor = RiskPredictor()
sentiment_analyzer = SentimentAnalyzer()
sentiment_feed = RealTimeSentimentFeed()
report_generator = ReportGenerator()

user_store = {
    "admin": {
        "password_hash": bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode(),
        "role": "admin"
    },
    "user": {
        "password_hash": bcrypt.hashpw("user123".encode(), bcrypt.gensalt()).decode(),
        "role": "user"
    }
}

rate_limits = {}

class TradeData(BaseModel):
    symbol: str
    quantity: float
    price: float
    trade_date: Optional[str] = None

class MarketDataQuery(BaseModel):
    symbols: List[str] = Field(..., min_items=1)
    start_date: str
    end_date: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class StressTestRequest(BaseModel):
    position_values: Dict[str, float]
    scenario: str = "base_case"
    time_periods: int = 20

class SentimentRequest(BaseModel):
    symbols: List[str]
    days: int = 7

class ReportRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: Optional[str] = None
    include_stress_test: bool = True
    include_sentiment: bool = True
    include_ml_predictions: bool = True
    output_format: str = "html"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        exp = payload.get("exp")
        
        if not username or username not in user_store:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        if datetime.utcnow().timestamp() > exp:
            raise HTTPException(status_code=401, detail="Token expired")
        
        return {"username": username, "role": user_store[username]["role"]}
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def check_rate_limit(ip_address: str = Header(None, alias="X-Forwarded-For")):
    if not RATE_LIMIT_ENABLED:
        return True
    
    now = datetime.utcnow().timestamp()
    
    if ip_address not in rate_limits:
        rate_limits[ip_address] = {
            "requests": 1,
            "window_start": now
        }
        return True
    
    limit_data = rate_limits[ip_address]
    
    if now - limit_data["window_start"] > RATE_LIMIT_PERIOD:
        limit_data["window_start"] = now
        limit_data["requests"] = 1
        return True
    
    if limit_data["requests"] >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_PERIOD} seconds."
        )
    
    limit_data["requests"] += 1
    return True

@app.get("/", tags=["Health"])
async def root():
    return {"message": "AI-Driven Trade Risk Assessment API is running"}

@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(request: LoginRequest):
    if request.username not in user_store:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    stored_hash = user_store[request.username]["password_hash"]
    if not bcrypt.checkpw(request.password.encode(), stored_hash.encode()):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    expiration = datetime.utcnow() + timedelta(seconds=JWT_EXPIRATION_SECONDS)
    
    payload = {
        "sub": request.username,
        "role": user_store[request.username]["role"],
        "exp": expiration.timestamp()
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    return {
        "access_token": token,
        "token_type": "bearer"
    }

@app.get("/market-data", tags=["Data"])
async def get_market_data(
    symbols: List[str] = Query(..., min_items=1),
    start_date: str = Query(...),
    end_date: Optional[str] = Query(None),
    user: dict = Depends(verify_token),
    _: bool = Depends(check_rate_limit)
):
    try:
        market_data = data_loader.load_market_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        if market_data.empty:
            return JSONResponse(
                status_code=404,
                content={"message": "No market data found for the specified symbols and date range"}
            )
        
        market_data_dict = market_data.to_dict(orient="records")
        return {"market_data": market_data_dict}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate-risk", tags=["Risk"])
async def calculate_risk(
    trades: List[TradeData],
    market_query: MarketDataQuery,
    confidence_level: float = Query(0.95, ge=0.8, le=0.99),
    user: dict = Depends(verify_token),
    _: bool = Depends(check_rate_limit)
):
    try:
        trade_data = pd.DataFrame([
            {
                "Symbol": t.symbol,
                "Quantity": t.quantity,
                "Price": t.price,
                "TradeDate": t.trade_date or datetime.now().strftime("%Y-%m-%d")
            } for t in trades
        ])
        
        market_data = data_loader.load_market_data(
            symbols=market_query.symbols,
            start_date=market_query.start_date,
            end_date=market_query.end_date
        )
        
        if market_data.empty:
            return JSONResponse(
                status_code=404,
                content={"message": "No market data found for the specified symbols and date range"}
            )
        
        risk_calc = RiskCalculator(confidence_level=confidence_level)
        trade_risk = risk_calc.calculate_trade_risk_metrics(trade_data, market_data)
        portfolio_metrics = risk_calc.calculate_portfolio_metrics(trade_risk, market_data)
        
        return {
            "trade_risk": trade_risk.to_dict(orient="records"),
            "portfolio_metrics": portfolio_metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stress-test", tags=["Risk"])
async def run_stress_test(
    request: StressTestRequest,
    user: dict = Depends(verify_token),
    _: bool = Depends(check_rate_limit)
):
    try:
        symbols = list(request.position_values.keys())
        
        if len(symbols) == 0:
            raise HTTPException(status_code=400, detail="Position values must be provided")
        
        position_values = np.array(list(request.position_values.values()))
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=252)).strftime("%Y-%m-%d")
        
        market_data = data_loader.load_market_data(symbols=symbols, start_date=start_date, end_date=end_date)
        
        if market_data.empty:
            raise HTTPException(status_code=404, detail="No market data found for the specified symbols")
        
        returns_data = []
        for symbol in symbols:
            symbol_market = market_data[market_data["Symbol"] == symbol]
            returns_data.append(symbol_market["DailyReturn"].dropna().values)
        
        means = np.array([np.mean(ret) for ret in returns_data])
        
        cov_matrix = np.zeros((len(symbols), len(symbols)))
        for i in range(len(symbols)):
            for j in range(len(symbols)):
                if i == j:
                    cov_matrix[i, j] = np.std(returns_data[i]) ** 2
                else:
                    corr = np.corrcoef(returns_data[i], returns_data[j])[0, 1]
                    cov_matrix[i, j] = corr * np.std(returns_data[i]) * np.std(returns_data[j])
        
        scenarios = monte_carlo.define_standard_scenarios()
        
        if request.scenario not in scenarios:
            scenarios = {request.scenario: scenarios["base_case"]}
        else:
            scenarios = {request.scenario: scenarios[request.scenario]}
        
        stress_results = monte_carlo.run_stress_test(
            position_values=position_values,
            means=means,
            covariance_matrix=cov_matrix,
            scenarios=scenarios,
            time_periods=request.time_periods
        )
        
        return {
            "scenario": request.scenario,
            "time_periods": request.time_periods,
            "results": stress_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentiment/{symbol}", tags=["Sentiment"])
async def get_sentiment(
    symbol: str = Path(...),
    days: int = Query(7, ge=1, le=30),
    user: dict = Depends(verify_token),
    _: bool = Depends(check_rate_limit)
):
    try:
        sentiment_data = sentiment_feed.aggregate_sentiment_data(symbol, days)
        
        if sentiment_data.empty:
            return JSONResponse(
                status_code=404,
                content={"message": f"No sentiment data found for {symbol}"}
            )
        
        latest_sentiment = sentiment_feed.get_latest_sentiment(symbol, days)
        
        return {
            "symbol": symbol,
            "latest_sentiment": latest_sentiment,
            "historical_sentiment": sentiment_data.to_dict(orient="records")
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment/fetch", tags=["Sentiment"])
async def fetch_sentiment(
    request: SentimentRequest,
    user: dict = Depends(verify_token),
    _: bool = Depends(check_rate_limit)
):
    try:
        if user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Admin role required")
        
        sentiment_feed.fetch_data_for_symbols(request.symbols)
        
        return {
            "status": "success",
            "message": f"Sentiment data fetching initiated for {len(request.symbols)} symbols"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment/schedule", tags=["Sentiment"])
async def schedule_sentiment(
    request: SentimentRequest,
    interval_minutes: int = Query(60, ge=15, le=1440),
    user: dict = Depends(verify_token),
    _: bool = Depends(check_rate_limit)
):
    try:
        if user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Admin role required")
        
        sentiment_feed.schedule_data_fetching(request.symbols, interval_minutes)
        
        return {
            "status": "success",
            "message": f"Scheduled sentiment data fetching for {len(request.symbols)} symbols every {interval_minutes} minutes"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/predict", tags=["Machine Learning"])
async def predict_risk(
    trades: List[TradeData],
    market_query: MarketDataQuery,
    horizon: int = Query(10, ge=1, le=60),
    use_sentiment: bool = Query(True),
    user: dict = Depends(verify_token),
    _: bool = Depends(check_rate_limit)
):
    try:
        trade_data = pd.DataFrame([
            {
                "Symbol": t.symbol,
                "Quantity": t.quantity,
                "Price": t.price,
                "TradeDate": t.trade_date or datetime.now().strftime("%Y-%m-%d")
            } for t in trades
        ])
        
        market_data = data_loader.load_market_data(
            symbols=market_query.symbols,
            start_date=market_query.start_date,
            end_date=market_query.end_date
        )
        
        if market_data.empty:
            raise HTTPException(status_code=404, detail="No market data found for the specified symbols")
        
        position_values = {}
        for _, row in trade_data.iterrows():
            position_values[row["Symbol"]] = row["Quantity"] * row["Price"]
        
        sentiment_data = None
        if use_sentiment:
            sentiment_data = pd.concat([
                sentiment_feed.aggregate_sentiment_data(symbol, 30)
                for symbol in position_values.keys()
            ])
        
        predictions = risk_predictor.predict_var_es(
            market_data=market_data,
            position_values=position_values,
            confidence_level=0.95,
            horizon=horizon,
            sentiment_data=sentiment_data
        )
        
        return {
            "predictions": predictions,
            "horizon": horizon,
            "used_sentiment": use_sentiment
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report/generate", tags=["Reports"])
async def generate_report(
    request: ReportRequest,
    user: dict = Depends(verify_token),
    _: bool = Depends(check_rate_limit)
):
    try:
        market_data = data_loader.load_market_data(
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if market_data.empty:
            raise HTTPException(status_code=404, detail="No market data found for the specified symbols")
        
        last_date = market_data["Date"].max()
        
        trade_data = pd.DataFrame([
            {
                "Symbol": symbol,
                "Quantity": 100,  
                "Price": market_data[market_data["Symbol"] == symbol]["Close"].iloc[-1],
                "TradeDate": last_date
            } for symbol in request.symbols
        ])
        
        risk_calc = RiskCalculator(confidence_level=0.95)
        trade_risk = risk_calc.calculate_trade_risk_metrics(trade_data, market_data)
        portfolio_metrics = risk_calc.calculate_portfolio_metrics(trade_risk, market_data)
        
        stress_test_results = None
        if request.include_stress_test:
            position_values = {}
            for _, row in trade_risk.iterrows():
                position_values[row["Symbol"]] = row["Value"]
            
            returns_data = []
            for symbol in request.symbols:
                symbol_market = market_data[market_data["Symbol"] == symbol]
                returns_data.append(symbol_market["DailyReturn"].dropna().values)
            
            means = np.array([np.mean(ret) for ret in returns_data])
            
            cov_matrix = np.zeros((len(request.symbols), len(request.symbols)))
            for i in range(len(request.symbols)):
                for j in range(len(request.symbols)):
                    if i == j:
                        cov_matrix[i, j] = np.std(returns_data[i]) ** 2
                    else:
                        corr = np.corrcoef(returns_data[i], returns_data[j])[0, 1]
                        cov_matrix[i, j] = corr * np.std(returns_data[i]) * np.std(returns_data[j])
            
            scenarios = monte_carlo.define_standard_scenarios()
            stress_results = monte_carlo.run_stress_test(
                position_values=np.array(list(position_values.values())),
                means=means,
                covariance_matrix=cov_matrix,
                scenarios=scenarios,
                time_periods=20
            )
            stress_test_results = stress_results
        
        sentiment_data = None
        if request.include_sentiment:
            all_sentiment = []
            for symbol in request.symbols:
                symbol_sentiment = sentiment_feed.aggregate_sentiment_data(symbol, 30)
                if not symbol_sentiment.empty:
                    all_sentiment.append(symbol_sentiment)
            
            if all_sentiment:
                sentiment_data = pd.concat(all_sentiment)
        
        ml_predictions = None
        if request.include_ml_predictions:
            position_values = {}
            for _, row in trade_risk.iterrows():
                position_values[row["Symbol"]] = row["Value"]
            
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
            stress_test_results=stress_test_results,
            sentiment_data=sentiment_data,
            ml_predictions=ml_predictions,
            output_format=request.output_format
        )
        
        return FileResponse(
            path=output_path,
            filename=os.path.basename(output_path),
            media_type="text/html" if request.output_format == "html" else "application/pdf"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    sentiment_feed.start()

@app.on_event("shutdown")
async def shutdown_event():
    sentiment_feed.stop()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
