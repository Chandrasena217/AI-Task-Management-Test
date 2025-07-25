import os
from pathlib import Path

# Base configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Dataset"
MODELS_DIR = BASE_DIR / "models"
FEATURES_DIR = BASE_DIR / "Feature Extraction Files"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
FEATURES_DIR.mkdir(exist_ok=True)

# App configuration
APP_TITLE = "AI Task Manager"
APP_ICON = "🤖"
PAGE_LAYOUT = "wide"

# ML Model settings
MODEL_CONFIG = {
    'tfidf_max_features': 5000,
    'tfidf_ngram_range': (1, 2),
    'w2v_vector_size': 100,
    'w2v_window': 5,
    'w2v_min_count': 2,
    'bert_model': 'bert-base-uncased',
    'sentence_transformer_model': 'all-MiniLM-L6-v2'
}

# Database configuration (if using)
DATABASE_CONFIG = {
    'url': os.getenv('DATABASE_URL', 'sqlite:///tasks.db'),
    'echo': False
}
