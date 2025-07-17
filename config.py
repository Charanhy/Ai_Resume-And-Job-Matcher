"""
Configuration settings for AI Resume Job Matcher
"""
import os
from pathlib import Path

# ================================
# STREAMLIT SETTINGS
# ================================

# Page configuration
STREAMLIT_CONFIG = {
    'page_title': 'AI Resume Job Matcher',
    'page_icon': '🤖',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Theme colors
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#ff7f0e',
    'danger': '#d62728',
    'info': '#17a2b8'
}

# ================================
# LOGGING SETTINGS
# ================================

import logging

# Logging configuration
LOGGING_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': DATA_DIR / 'app.log'
}

# ================================
# CACHING SETTINGS
# ================================

# Cache settings for AI models and embeddings
CACHE_SETTINGS = {
    'enable_model_cache': True,
    'cache_dir': DATA_DIR / 'cache',
    'max_cache_size_gb': 2,
    'cache_expiry_hours': 24
}

# Create cache directory
(DATA_DIR / 'cache').mkdir(exist_ok=True)

# ================================
# ERROR HANDLING
# ================================

# Error messages
ERROR_MESSAGES = {
    'file_too_large': f'File size exceeds {MAX_FILE_SIZE_MB}MB limit',
    'unsupported_format': f'Supported formats: {", ".join(SUPPORTED_RESUME_FORMATS)}',
    'text_too_short': f'Resume must contain at least {MIN_TEXT_LENGTH} characters',
    'model_load_error': 'Failed to load AI model. Please try again.',
    'api_error': 'Job search API is currently unavailable',
    'processing_error': 'Error processing your resume. Please check the format.'
}

# ================================
# SAMPLE DATA (for testing)
# ================================

# Sample job data for testing when APIs are not available
SAMPLE_JOBS = [
    {
        'title': 'Senior Software Engineer',
        'company': 'TechCorp Inc.',
        'location': 'San Francisco, CA',
        'type': 'Full-time',
        'experience': '3-5 years',
        'salary': '$120,000 - $150,000',
        'description': 'We are seeking a skilled software engineer with experience in Python, JavaScript, and cloud technologies...',
        'skills': ['Python', 'JavaScript', 'AWS', 'Docker', 'React'],
        'requirements': ['Bachelor\'s degree in CS', '3+ years experience', 'Strong problem-solving skills']
    },
    {
        'title': 'Data Scientist',
        'company': 'DataTech Solutions',
        'location': 'Remote',
        'type': 'Full-time',
        'experience': '2-4 years',
        'salary': '$100,000 - $130,000',
        'description': 'Join our data science team to build machine learning models and extract insights from data...',
        'skills': ['Python', 'Machine Learning', 'SQL', 'Pandas', 'Scikit-learn'],
        'requirements': ['MS in Data Science or related field', 'ML experience', 'Statistical knowledge']
    },
    {
        'title': 'Frontend Developer',
        'company': 'WebDev Agency',
        'location': 'New York, NY',
        'type': 'Full-time',
        'experience': '1-3 years',
        'salary': '$80,000 - $100,000',
        'description': 'Looking for a creative frontend developer to build responsive web applications...',
        'skills': ['JavaScript', 'React', 'HTML', 'CSS', 'TypeScript'],
        'requirements': ['Portfolio of web projects', 'React experience', 'Design sense']
    }
]

# ================================
# UTILITY FUNCTIONS
# ================================

def get_model_path(model_name: str) -> Path:
    """Get the path where a model should be cached"""
    return CACHE_SETTINGS['cache_dir'] / model_name.replace('/', '_')

def is_valid_file_size(file_size: int) -> bool:
    """Check if file size is within limits"""
    return file_size <= MAX_FILE_SIZE_BYTES

def is_valid_text_length(text: str) -> bool:
    """Check if text length is sufficient for processing"""
    return MIN_TEXT_LENGTH <= len(text) <= MAX_TEXT_LENGTH

def get_supported_formats_string() -> str:
    """Get comma-separated string of supported formats"""
    return ", ".join(SUPPORTED_RESUME_FORMATS)

# ================================
# ENVIRONMENT VARIABLES
# ================================

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

# Debug mode
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# API Keys (set these in your .env file or environment)
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')  # Optional, for future use

# ================================
# VERSION INFO
# ================================

VERSION = '1.0.0'
LAST_UPDATED = '2024-12-26'
AUTHOR = 'AI Resume Matcher Team'
LICENSE = 'MIT'

# ================================
# FEATURE FLAGS
# ================================

# Enable/disable features during development
FEATURES = {
    'pdf_processing': True,
    'word_processing': True,
    'ai_summarization': True,
    'job_api_integration': True,
    'email_notifications': False,  # Not implemented yet
    'advanced_analytics': False,   # Future feature
    'user_accounts': False,        # Future feature
}

# Print configuration status when imported
if __name__ == '__main__':
    print("=== AI Resume Job Matcher Configuration ===")
    print(f"Version: {VERSION}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Cache Directory: {CACHE_SETTINGS['cache_dir']}")
    print(f"Default Model: {DEFAULT_EMBEDDING_MODEL}")
    print(f"Debug Mode: {DEBUG_MODE}")
    print("=" * 45)
# PROJECT PATHS
# ================================
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
UTILS_DIR = BASE_DIR / 'utils'
TEMPLATES_DIR = BASE_DIR / 'templates'
TESTS_DIR = BASE_DIR / 'tests'

# Create directories if they don't exist
directories_to_create = [
    DATA_DIR,
    MODELS_DIR,
    UTILS_DIR,
    TEMPLATES_DIR,
    TESTS_DIR,
    DATA_DIR / 'resumes',
    DATA_DIR / 'jobs',
    DATA_DIR / 'processed',
    DATA_DIR / 'chroma_db'
]

for directory in directories_to_create:
    directory.mkdir(parents=True, exist_ok=True)

# ================================
# AI MODEL SETTINGS
# ================================

# Embedding Models (choose one)
EMBEDDING_MODELS = {
    'fast': 'all-MiniLM-L6-v2',           # Fastest, good quality
    'balanced': 'all-mpnet-base-v2',       # Best balance of speed/quality
    'multilingual': 'paraphrase-multilingual-MiniLM-L12-v2'  # Multi-language support
}

# Default embedding model
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODELS['fast']

# Summarization Models
SUMMARIZER_MODELS = {
    'bart': 'facebook/bart-large-cnn',
    'pegasus': 'google/pegasus-xsum',
    't5': 't5-small'
}

# Default summarization model
DEFAULT_SUMMARIZER_MODEL = SUMMARIZER_MODELS['bart']

# ================================
# DATABASE SETTINGS
# ================================

# ChromaDB settings
CHROMA_DB_PATH = DATA_DIR / 'chroma_db'
CHROMA_COLLECTION_NAME = 'job_embeddings'
CHROMA_DISTANCE_FUNCTION = 'cosine'  # cosine, l2, ip (inner product)

# ================================
# SEARCH SETTINGS
# ================================

# Default search parameters
DEFAULT_MAX_RESULTS = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.5
MIN_SIMILARITY_THRESHOLD = 0.3
MAX_SIMILARITY_THRESHOLD = 0.9

# Similarity weights for different aspects
SIMILARITY_WEIGHTS = {
    'skills': 0.4,
    'experience': 0.3,
    'education': 0.2,
    'location': 0.1
}

# ================================
# JOB API SETTINGS
# ================================

# Free Job APIs (you'll need to get API keys)
JOB_APIS = {
    'adzuna': {
        'base_url': 'https://api.adzuna.com/v1/api/jobs',
        'app_id': os.getenv('ADZUNA_APP_ID', ''),
        'app_key': os.getenv('ADZUNA_APP_KEY', ''),
        'free_limit': 1000  # requests per month
    },
    'jsearch': {
        'base_url': 'https://jsearch.p.rapidapi.com/search',
        'api_key': os.getenv('JSEARCH_API_KEY', ''),
        'free_limit': 150  # requests per month
    }
}

# ================================
# FILE PROCESSING SETTINGS
# ================================

# Supported file types
SUPPORTED_RESUME_FORMATS = ['pdf', 'docx', 'txt']
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Text processing settings
MAX_TEXT_LENGTH = 10000  # Maximum characters to process
MIN_TEXT_LENGTH = 100    # Minimum characters required

# ================================