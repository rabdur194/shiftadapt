"""Configuration for ShiftAdapt."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OLD_DOMAIN_PATH = DATA_DIR / "old_domain" / "examples.json"
NEW_DOMAIN_PATH = DATA_DIR / "new_domain" / "examples.json"
ADAPT_PATH = DATA_DIR / "adaptation_examples" / "examples.json"

# Embedding model (pretrained foundation model)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "true").lower() == "true"  # True = no API key needed

# Shift detection
SHIFT_THRESHOLD = float(os.getenv("SHIFT_THRESHOLD", "0.25"))  # cosine distance threshold

# Retrieval
TOP_K = int(os.getenv("TOP_K", "3"))

# Paths for persisted knowledge
KB_PATH = BASE_DIR / "artifacts" / "knowledge_base"
KB_PATH.mkdir(parents=True, exist_ok=True)
