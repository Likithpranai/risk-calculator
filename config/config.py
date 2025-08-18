
import os
from dotenv import load_dotenv


load_dotenv()


SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-jwt-secret-key-here")
JWT_EXPIRATION_DELTA = 24 * 60 * 60
HASH_ALGORITHM = 'bcrypt'
BCRYPT_LOG_ROUNDS = 13


RATE_LIMIT = 100
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
API_KEY_HEADER = "X-API-Key"
SSL_REQUIRED = os.getenv("SSL_REQUIRED", "False").lower() in ('true', '1', 't')


CACHE_TYPE = "SimpleCache"
CACHE_DEFAULT_TIMEOUT = 300
WORKERS = int(os.getenv("WORKERS", 4))
CHUNK_SIZE = 10000
USE_PARALLEL = True
MAX_THREADS = int(os.getenv("MAX_THREADS", os.cpu_count() or 4))
COMPRESSION_LEVEL = 1


DEFAULT_CONFIDENCE_LEVEL = 0.95
SIMULATION_ITERATIONS = 10000
ML_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models")
SENTIMENT_BATCH_SIZE = 16


DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
REPORT_TEMPLATES = os.path.join(os.path.dirname(__file__), "../src/reports/templates")

# API keys for sentiment analysis services
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
SENTIMENT_CACHE_DIR = os.path.join(DATA_DIR, "sentiment/cache")


for directory in [DATA_DIR, LOG_DIR, REPORT_TEMPLATES]:
    os.makedirs(directory, exist_ok=True)
