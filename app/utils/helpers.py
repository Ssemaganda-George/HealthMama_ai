"""
Helper functions for HealthMama AI
"""
import os
import logging
from typing import Optional


def setup_logging(app_name: str = "healthmama", log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(app_name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, log_level.upper()))
    
    return logger


def ensure_directory_exists(directory_path: str) -> bool:
    """Ensure a directory exists, create if it doesn't"""
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False


def get_file_size(file_path: str) -> Optional[int]:
    """Get file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except Exception:
        return None


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def is_safe_path(path: str, base_path: str) -> bool:
    """Check if a file path is safe (within base directory)"""
    try:
        abs_base = os.path.abspath(base_path)
        abs_path = os.path.abspath(os.path.join(base_path, path))
        return abs_path.startswith(abs_base)
    except Exception:
        return False