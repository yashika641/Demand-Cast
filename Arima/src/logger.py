# logger.py
import logging
import os
from logging.handlers import RotatingFileHandler

# Create logs directory if not exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file path
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Configure logger
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)  # Capture everything (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Console Handler (prints logs to terminal)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Show INFO and above in console
console_handler.setFormatter(formatter)

# File Handler (rotates log file when size exceeds 5MB)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)  # Store all logs in file
file_handler.setFormatter(formatter)

# Add handlers to logger
if not logger.hasHandlers():  # Prevent duplicate handlers on reload
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def get_logger(name: str = None) -> logging.Logger:
    """
    Returns a logger instance.
    If name is provided, returns a child logger (e.g., module-specific).
    """
    if name:
        return logger.getChild(name)
    return logger
