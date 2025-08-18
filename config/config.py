"""
Configuration settings for the AI-Driven Trade Risk Assessment System
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")  # Should be set in .env file in production
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-jwt-secret-key-here")  # Should be set in .env file
JWT_EXPIRATION_DELTA = 24 * 60 * 60  # Token validity in seconds (24 hours)
HASH_ALGORITHM = 'bcrypt'  # Password hashing algorithm
BCRYPT_LOG_ROUNDS = 13  # Higher number = more secure but slower

# API Security
RATE_LIMIT = 100  # Max requests per minute per user
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
API_KEY_HEADER = "X-API-Key"
SSL_REQUIRED = os.getenv("SSL_REQUIRED", "False").lower() in ('true', '1', 't')

# Performance optimization
CACHE_TYPE = "SimpleCache"  # For development
CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
WORKERS = int(os.getenv("WORKERS", 4))  # Number of workers for API service
CHUNK_SIZE = 10000  # Process data in chunks of this size
USE_PARALLEL = True  # Use parallel processing where appropriate
MAX_THREADS = int(os.getenv("MAX_THREADS", os.cpu_count() or 4))
COMPRESSION_LEVEL = 1  # 0-9, higher = better compression but slower

# Model settings
DEFAULT_CONFIDENCE_LEVEL = 0.95  # For VaR, ES calculations
SIMULATION_ITERATIONS = 10000  # For Monte Carlo simulations
ML_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models")
SENTIMENT_BATCH_SIZE = 16  # For sentiment analysis models

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
REPORT_TEMPLATES = os.path.join(os.path.dirname(__file__), "../src/reports/templates")

# Create directories if they don't exist
for directory in [DATA_DIR, LOG_DIR, REPORT_TEMPLATES]:
    os.makedirs(directory, exist_ok=True)
