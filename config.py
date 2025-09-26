#!/usr/bin/env python3
"""
Configuration settings for the Image Downloader system.
"""

import os
from pathlib import Path

# GitHub Repository Configuration (for Phase 3 cache updates)
GITHUB_REPO_URL = "https://github.com/raodurgesh325-wq/flexit-rag-images.git"
GITHUB_REPO_NAME = "flexit-rag-images"
GITHUB_USERNAME = "raodurgesh325-wq"
GITHUB_RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO_NAME}/main"
)

# Local Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOWNLOAD_IMAGES_DIR = PROJECT_ROOT / "crawler" / "download_images"
IMAGE_MAPPING_CACHE_FILE = DOWNLOAD_IMAGES_DIR / "image_mapping_cache.json"
TEMP_DOWNLOAD_DIR = DOWNLOAD_IMAGES_DIR / "images"

# CloudFront Configuration
CLOUDFRONT_BASE_URL = "https://d3u2d4xznamk2r.cloudfront.net"

# Download Settings
DOWNLOAD_BATCH_SIZE = 10  # Number of concurrent downloads
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds
REQUEST_TIMEOUT = 30  # seconds
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = DOWNLOAD_IMAGES_DIR / "download_log.txt"

# Directory Structure in GitHub Repo (for Phase 3 URL generation)
GITHUB_IMAGES_DIR = "images"

# File Extensions to Process
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"}

# Cache Backup Settings
BACKUP_DIR = DOWNLOAD_IMAGES_DIR / "backups"
MAX_BACKUPS = 5  # Keep last 5 backups

# Progress Tracking
PROGRESS_UPDATE_INTERVAL = 10  # Update progress every N downloads
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N downloads

# URL Validation
VALIDATE_URLS_BEFORE_DOWNLOAD = True
VALIDATE_GITHUB_URLS_AFTER_UPLOAD = True

# Error Handling
CONTINUE_ON_ERROR = True  # Continue processing even if some downloads fail
MAX_CONSECUTIVE_ERRORS = 10  # Stop if too many consecutive errors


def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [TEMP_DOWNLOAD_DIR, BACKUP_DIR, DOWNLOAD_IMAGES_DIR]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_github_image_url(relative_path: str) -> str:
    """
    Generate GitHub raw URL for an image.

    Args:
        relative_path: Relative path within the images directory

    Returns:
        Full GitHub raw URL
    """
    # Clean the path and ensure it doesn't start with /
    clean_path = relative_path.lstrip("/")
    return f"{GITHUB_RAW_BASE_URL}/{GITHUB_IMAGES_DIR}/{clean_path}"


def get_local_image_path(relative_path: str) -> Path:
    """
    Get local file path for an image in the temp download directory.

    Args:
        relative_path: Relative path within the images directory

    Returns:
        Full local path
    """
    return TEMP_DOWNLOAD_DIR / relative_path
