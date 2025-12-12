"""
Comprehensive Input Validation and Sanitization Module

Provides strict validation and sanitization for all user inputs across services.
Prevents injection attacks, XSS, path traversal, and other security vulnerabilities.
"""

import re
import html
import base64
import hashlib
from typing import Any, Optional, List, Dict
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    # Dangerous patterns for injection attacks
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bDROP\b.*\bTABLE\b)",
        r"(\bINSERT\b.*\bINTO\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(\bUPDATE\b.*\bSET\b)",
        r"(--|\#|\/\*|\*\/)",
        r"(\bEXEC\b|\bEXECUTE\b)",
        r"(\bxp_cmdshell\b)",
    ]
    
    CYPHER_INJECTION_PATTERNS = [
        r"(\bMATCH\b.*\bDELETE\b)",
        r"(\bCREATE\b.*\bNODE\b)",
        r"(\bDROP\b.*\bINDEX\b)",
        r"(\bMERGE\b.*\bON\b.*\bCREATE\b)",
        r"(//.*\n)",  # Cypher comments
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers like onclick=
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.",
        r"%2e%2e",
        r"\.\.\\",
    ]
    
    # Allowed URL schemes
    ALLOWED_URL_SCHEMES = {"http", "https"}
    
    # Maximum lengths for different input types
    MAX_LENGTHS = {
        "material_name": 100,
        "query_text": 5000,
        "url": 2048,
        "email": 254,
        "phone": 20,
        "name": 200,
        "description": 10000,
        "base64_image": 10 * 1024 * 1024,  # 10MB
    }
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize string input
        
        Args:
            value: Input string
            max_length: Maximum allowed length
        
        Returns:
            Sanitized string
        
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value)}")
        
        # Strip whitespace
        value = value.strip()
        
        # Check length
        if max_length and len(value) > max_length:
            raise ValueError(f"Input exceeds maximum length of {max_length}")
        
        # HTML escape to prevent XSS
        value = html.escape(value)
        
        return value
    
    @classmethod
    def validate_no_injection(cls, value: str, input_type: str = "general") -> str:
        """
        Validate that input doesn't contain injection patterns
        
        Args:
            value: Input string
            input_type: Type of input ("sql", "cypher", "general")
        
        Returns:
            Validated string
        
        Raises:
            ValueError: If injection pattern detected
        """
        if not value:
            return value
        
        # Check for SQL injection
        if input_type in ("sql", "general"):
            for pattern in cls.SQL_INJECTION_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    logger.warning(f"SQL injection attempt detected: {pattern}")
                    raise ValueError("Invalid input: potential SQL injection detected")
        
        # Check for Cypher injection
        if input_type in ("cypher", "general"):
            for pattern in cls.CYPHER_INJECTION_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    logger.warning(f"Cypher injection attempt detected: {pattern}")
                    raise ValueError("Invalid input: potential Cypher injection detected")
        
        # Check for XSS
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"XSS attempt detected: {pattern}")
                raise ValueError("Invalid input: potential XSS detected")
        
        # Check for path traversal
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Path traversal attempt detected: {pattern}")
                raise ValueError("Invalid input: potential path traversal detected")
        
        return value

    @classmethod
    def validate_material_name(cls, name: str) -> str:
        """
        Validate material name

        Args:
            name: Material name

        Returns:
            Validated and sanitized material name
        """
        name = cls.sanitize_string(name, max_length=cls.MAX_LENGTHS["material_name"])
        name = cls.validate_no_injection(name, input_type="cypher")

        # Material names should only contain alphanumeric, spaces, hyphens, underscores
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', name):
            raise ValueError("Material name contains invalid characters")

        return name

    @classmethod
    def validate_url(cls, url: str) -> str:
        """
        Validate URL

        Args:
            url: URL string

        Returns:
            Validated URL

        Raises:
            ValueError: If URL is invalid
        """
        if not url:
            raise ValueError("URL cannot be empty")

        if len(url) > cls.MAX_LENGTHS["url"]:
            raise ValueError(f"URL exceeds maximum length of {cls.MAX_LENGTHS['url']}")

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValueError(f"Invalid URL format: {e}")

        # Check scheme
        if parsed.scheme not in cls.ALLOWED_URL_SCHEMES:
            raise ValueError(f"URL scheme must be one of: {cls.ALLOWED_URL_SCHEMES}")

        # Check for suspicious patterns
        if any(pattern in url.lower() for pattern in ["javascript:", "data:", "file:"]):
            raise ValueError("URL contains suspicious protocol")

        return url

    @classmethod
    def validate_base64_image(cls, b64_string: str) -> str:
        """
        Validate base64 encoded image

        Args:
            b64_string: Base64 encoded image string

        Returns:
            Validated base64 string

        Raises:
            ValueError: If image is invalid
        """
        if not b64_string:
            raise ValueError("Base64 image cannot be empty")

        # Remove data URL prefix if present
        if b64_string.startswith("data:"):
            try:
                header, b64_string = b64_string.split(",", 1)
                # Validate MIME type
                if not any(mime in header for mime in ["image/jpeg", "image/png", "image/jpg", "image/webp"]):
                    raise ValueError("Invalid image MIME type")
            except ValueError:
                raise ValueError("Invalid data URL format")

        # Validate base64 encoding
        try:
            decoded = base64.b64decode(b64_string, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {e}")

        # Check size
        if len(decoded) > cls.MAX_LENGTHS["base64_image"]:
            raise ValueError(f"Image size exceeds maximum of {cls.MAX_LENGTHS['base64_image'] / 1024 / 1024}MB")

        # Validate image magic bytes (basic check)
        if not (
            decoded.startswith(b'\xff\xd8\xff') or  # JPEG
            decoded.startswith(b'\x89PNG\r\n\x1a\n') or  # PNG
            decoded.startswith(b'RIFF') and b'WEBP' in decoded[:12]  # WebP
        ):
            raise ValueError("Invalid image format (must be JPEG, PNG, or WebP)")

        return b64_string

    @classmethod
    def validate_numeric_range(
        cls,
        value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        field_name: str = "value"
    ) -> float:
        """
        Validate numeric value is within range

        Args:
            value: Numeric value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            field_name: Field name for error messages

        Returns:
            Validated value

        Raises:
            ValueError: If value is out of range
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be a number")

        if min_value is not None and value < min_value:
            raise ValueError(f"{field_name} must be >= {min_value}")

        if max_value is not None and value > max_value:
            raise ValueError(f"{field_name} must be <= {max_value}")

        return value

    @classmethod
    def validate_email(cls, email: str) -> str:
        """
        Validate email address

        Args:
            email: Email address

        Returns:
            Validated email

        Raises:
            ValueError: If email is invalid
        """
        email = cls.sanitize_string(email, max_length=cls.MAX_LENGTHS["email"])

        # Basic email regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise ValueError("Invalid email format")

        return email.lower()

    @classmethod
    def validate_list_items(
        cls,
        items: List[Any],
        max_items: int = 100,
        item_validator: Optional[callable] = None
    ) -> List[Any]:
        """
        Validate list of items

        Args:
            items: List of items
            max_items: Maximum number of items
            item_validator: Optional validator function for each item

        Returns:
            Validated list

        Raises:
            ValueError: If list is invalid
        """
        if not isinstance(items, list):
            raise ValueError("Expected list")

        if len(items) > max_items:
            raise ValueError(f"List exceeds maximum of {max_items} items")

        if item_validator:
            validated_items = []
            for i, item in enumerate(items):
                try:
                    validated_items.append(item_validator(item))
                except Exception as e:
                    raise ValueError(f"Invalid item at index {i}: {e}")
            return validated_items

        return items

