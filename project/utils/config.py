
import os
from pathlib import Path
from dotenv import load_dotenv

# Define base directory (project root)
# Assumes config.py is in project/utils/config.py
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from project root or parent directory
env_path = BASE_DIR.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    # Try project root
    env_path = BASE_DIR / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        print(f"Warning: .env file not found at {env_path}")

# Export constants
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("gemini_api")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
