"""
Validation utilities for HealthMama AI
"""
import re
from typing import Optional, List


class InputValidator:
    """Validates user inputs for security and format"""
    
    # Security patterns to block
    BLOCKED_PATTERNS = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'onclick\s*=',
        r'onload\s*=',
        r'onerror\s*=',
        r'<iframe.*?>',
        r'<object.*?>',
        r'<embed.*?>',
        r'eval\s*\(',
        r'document\.cookie',
        r'window\.location'
    ]
    
    # File patterns to block
    BLOCKED_FILE_PATTERNS = [
        r'\.exe$',
        r'\.bat$',
        r'\.cmd$',
        r'\.scr$',
        r'\.com$',
        r'\.pif$',
        r'\.vbs$',
        r'\.js$',
        r'\.jar$'
    ]
    
    # Allowed image MIME types
    ALLOWED_IMAGE_TYPES = [
        'image/jpeg',
        'image/jpg', 
        'image/png',
        'image/gif',
        'image/webp'
    ]
    
    @classmethod
    def validate_text_input(cls, text: str, max_length: int = 1000) -> tuple[bool, Optional[str]]:
        """
        Validate text input for security issues
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not text:
            return False, "Text cannot be empty"
        
        if len(text) > max_length:
            return False, f"Text too long (max {max_length} characters)"
        
        # Check for blocked patterns
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Text contains potentially harmful content"
        
        return True, None
    
    @classmethod
    def validate_filename(cls, filename: str) -> tuple[bool, Optional[str]]:
        """
        Validate filename for security
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not filename:
            return False, "Filename cannot be empty"
        
        # Check for blocked file patterns
        for pattern in cls.BLOCKED_FILE_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                return False, "File type not allowed"
        
        return True, None
    
    @classmethod
    def validate_file_type(cls, mime_type: str) -> tuple[bool, Optional[str]]:
        """
        Validate file MIME type
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if mime_type not in cls.ALLOWED_IMAGE_TYPES:
            return False, f"File type {mime_type} not allowed. Only images are permitted."
        
        return True, None
    
    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """
        Sanitize text by removing potentially harmful content
        
        Returns:
            str: Sanitized text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove script content
        text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove javascript: protocols
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        # Remove event handlers
        text = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
        
        return text.strip()


class ModelValidator:
    """Validates model-related inputs"""
    
    VALID_MODELS = ['diabetes', 'preeclampsia']
    
    @classmethod
    def validate_model(cls, model: str) -> tuple[bool, Optional[str]]:
        """
        Validate model name
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not model:
            return False, "Model cannot be empty"
        
        if model not in cls.VALID_MODELS:
            return False, f"Invalid model. Must be one of: {', '.join(cls.VALID_MODELS)}"
        
        return True, None