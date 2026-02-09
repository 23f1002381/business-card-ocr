import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, UPLOAD_DIR, PROCESSED_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# OCR Settings
OCR_ENGINE = "easyocr"  # Options: 'easyocr', 'tesseract'
LANGUAGES = ['en']  # Primary language for OCR

# Model settings
USE_LAYOUTLM = True
LAYOUTLM_MODEL_NAME = "microsoft/layoutlm-base-uncased"
SPACY_MODEL = "en_core_web_sm"

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.7
MIN_TEXT_CONFIDENCE = 0.5

# Field extraction settings
PHONE_PATTERNS = [
    r'\+?[\d\s-]{10,}',  # International format
    r'\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}',  # US/CAN format
]

EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
URL_PATTERN = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

# Export settings
EXPORT_FORMATS = ['xlsx', 'csv', 'json']
DEFAULT_EXPORT_FORMAT = 'xlsx'

# UI Settings
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = BASE_DIR / 'logs' / 'app.log'
