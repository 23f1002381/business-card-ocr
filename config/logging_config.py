import logging
import logging.handlers
import os
from pathlib import Path
from config.settings import LOG_LEVEL, LOG_FILE

def setup_logging():
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Log format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Suppress third-party loggers
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    return logger
