#!/usr/bin/env python3
"""
Utility functions for the Image Downloader system.
"""

import re
import logging
import hashlib
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Optional, Tuple
import time
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for filesystem compatibility.

    Args:
        filename: Original filename
        max_length: Maximum allowed filename length

    Returns:
        Sanitized filename safe for filesystem
    """
    if not filename:
        return "unnamed_file"

    # Remove or replace invalid characters
    # Windows invalid chars: < > : " | ? * \ /
    # Also remove control characters
    sanitized = re.sub(r'[<>:"|?*\\/\x00-\x1f\x7f]', "_", filename)

    # Replace multiple underscores with single underscore
    sanitized = re.sub(r"_+", "_", sanitized)

    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")

    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed_file"

    # Truncate if too long, preserving extension
    if len(sanitized) > max_length:
        name_part, ext_part = Path(sanitized).stem, Path(sanitized).suffix
        max_name_length = max_length - len(ext_part)
        if max_name_length > 0:
            sanitized = name_part[:max_name_length] + ext_part
        else:
            sanitized = sanitized[:max_length]

    return sanitized


def extract_relative_path_from_url(url: str, base_url: str) -> str:
    """
    Extract relative path from a full URL by removing the base URL.

    Args:
        url: Full URL
        base_url: Base URL to remove

    Returns:
        Relative path
    """
    if not url or not base_url:
        return ""

    # Remove base URL
    if url.startswith(base_url):
        relative_path = url[len(base_url) :]
    else:
        # Try to extract path from URL
        parsed = urlparse(url)
        relative_path = parsed.path

    # Clean up the path
    relative_path = relative_path.lstrip("/")

    # URL decode to handle encoded characters
    relative_path = unquote(relative_path)

    return relative_path


def create_directory_structure(file_path: Path) -> None:
    """
    Create directory structure for a file path.

    Args:
        file_path: Path to file (directories will be created)
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)


def get_file_hash(file_path: Path) -> Optional[str]:
    """
    Calculate MD5 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        MD5 hash string or None if error
    """
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return None


def is_valid_image_extension(filename: str, allowed_extensions: set) -> bool:
    """
    Check if filename has a valid image extension.

    Args:
        filename: Filename to check
        allowed_extensions: Set of allowed extensions (with dots)

    Returns:
        True if valid image extension
    """
    if not filename:
        return False

    ext = Path(filename).suffix.lower()
    return ext in allowed_extensions


def generate_unique_filename(base_path: Path, filename: str) -> str:
    """
    Generate unique filename if file already exists.

    Args:
        base_path: Directory where file will be saved
        filename: Desired filename

    Returns:
        Unique filename
    """
    file_path = base_path / filename

    if not file_path.exists():
        return filename

    # File exists, generate unique name
    name_part = Path(filename).stem
    ext_part = Path(filename).suffix
    counter = 1

    while True:
        new_filename = f"{name_part}_{counter}{ext_part}"
        new_path = base_path / new_filename

        if not new_path.exists():
            return new_filename

        counter += 1

        # Safety check to avoid infinite loop
        if counter > 1000:
            # Use timestamp as fallback
            timestamp = int(time.time())
            return f"{name_part}_{timestamp}{ext_part}"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0

    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f} {size_names[i]}"


def create_backup_filename(original_path: Path) -> Path:
    """
    Create backup filename with timestamp.

    Args:
        original_path: Original file path

    Returns:
        Backup file path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_part = original_path.stem
    ext_part = original_path.suffix

    backup_filename = f"{name_part}_backup_{timestamp}{ext_part}"
    return original_path.parent / backup_filename


def validate_url(url: str) -> bool:
    """
    Basic URL validation.

    Args:
        url: URL to validate

    Returns:
        True if URL appears valid
    """
    if not url:
        return False

    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator for retrying functions on failure.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between attempts in seconds
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}"
                        )
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )

            raise last_exception

        return wrapper

    return decorator


def clean_url_for_filename(url: str) -> str:
    """
    Clean URL to create a safe filename component.

    Args:
        url: URL to clean

    Returns:
        Cleaned string safe for use in filenames
    """
    if not url:
        return "unknown"

    # Extract path from URL
    parsed = urlparse(url)
    path = parsed.path

    # Remove leading/trailing slashes
    path = path.strip("/")

    # Replace path separators with underscores
    path = path.replace("/", "_")

    # URL decode
    path = unquote(path)

    # Sanitize for filename
    return sanitize_filename(path)


def estimate_download_time(
    total_files: int, avg_file_size: int, bandwidth_mbps: float = 10.0
) -> Tuple[int, str]:
    """
    Estimate download time based on file count and average size.

    Args:
        total_files: Number of files to download
        avg_file_size: Average file size in bytes
        bandwidth_mbps: Available bandwidth in Mbps

    Returns:
        Tuple of (seconds, formatted_time_string)
    """
    total_bytes = total_files * avg_file_size
    total_bits = total_bytes * 8
    bandwidth_bps = bandwidth_mbps * 1_000_000

    # Add 20% overhead for network latency and processing
    estimated_seconds = int((total_bits / bandwidth_bps) * 1.2)

    # Format time string
    hours = estimated_seconds // 3600
    minutes = (estimated_seconds % 3600) // 60
    seconds = estimated_seconds % 60

    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds}s"
    else:
        time_str = f"{seconds}s"

    return estimated_seconds, time_str
