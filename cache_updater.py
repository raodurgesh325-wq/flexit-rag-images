#!/usr/bin/env python3
"""
Cache Updater module for updating image mapping cache with GitHub URLs.
"""

import json
import logging
from typing import Dict
from datetime import datetime

from .config import (
    IMAGE_MAPPING_CACHE_FILE,
    BACKUP_DIR,
    MAX_BACKUPS,
    get_github_image_url,
)
from .utils import (
    extract_relative_path_from_url,
    create_backup_filename
)

# Configure logging
logger = logging.getLogger(__name__)


class CacheUpdater:
    """Main class for updating image mapping cache with GitHub URLs."""

    def __init__(self):
        """Initialize the cache updater."""
        self.cache_data = None
        self.original_cache_data = None
        self.backup_file = None
        self.updated_images_count = 0

        # Ensure backup directory exists
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    def load_cache(self) -> bool:
        """
        Load the image mapping cache from file.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not IMAGE_MAPPING_CACHE_FILE.exists():
                logger.error(f"Cache file not found: {IMAGE_MAPPING_CACHE_FILE}")
                return False

            with open(IMAGE_MAPPING_CACHE_FILE, "r", encoding="utf-8") as f:
                self.cache_data = json.load(f)

            # Keep a copy of original data for backup
            self.original_cache_data = json.loads(json.dumps(self.cache_data))

            logger.info(f"Loaded cache with {self._count_total_images()} images")
            return True

        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return False

    def _count_total_images(self) -> int:
        """Count total number of images in the cache."""
        if not self.cache_data:
            return 0

        total = 0
        filename_to_images = self.cache_data.get("filename_to_images", {})

        for images in filename_to_images.values():
            total += len(images)

        return total

    def create_backup(self) -> bool:
        """
        Create a backup of the original cache file.

        Returns:
            True if backup created successfully, False otherwise
        """
        try:
            if not IMAGE_MAPPING_CACHE_FILE.exists():
                logger.warning("No cache file to backup")
                return False

            # Create backup filename with timestamp
            self.backup_file = create_backup_filename(IMAGE_MAPPING_CACHE_FILE)
            backup_path = BACKUP_DIR / self.backup_file.name

            # Copy original file to backup
            import shutil

            shutil.copy2(IMAGE_MAPPING_CACHE_FILE, backup_path)

            logger.info(f"Created backup: {backup_path}")

            # Clean up old backups
            self._cleanup_old_backups()

            return True

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False

    def _cleanup_old_backups(self):
        """Remove old backup files, keeping only the most recent ones."""
        try:
            # Find all backup files
            backup_pattern = f"{IMAGE_MAPPING_CACHE_FILE.stem}_backup_*.json"
            backup_files = list(BACKUP_DIR.glob(backup_pattern))

            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove excess backups
            if len(backup_files) > MAX_BACKUPS:
                for old_backup in backup_files[MAX_BACKUPS:]:
                    try:
                        old_backup.unlink()
                        logger.debug(f"Removed old backup: {old_backup}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old backup {old_backup}: {e}")

        except Exception as e:
            logger.warning(f"Error cleaning up old backups: {e}")

    def generate_github_url_from_enhance_url(
        self, enhance_url: str, cloudfront_base: str
    ) -> str:
        """
        Generate GitHub URL from CloudFront enhance_url with proper URL encoding.

        Args:
            enhance_url: Original CloudFront URL
            cloudfront_base: CloudFront base URL to remove

        Returns:
            Corresponding GitHub raw URL with proper encoding
        """
        try:
            # Import URL validation
            from crawler.url.url_encoder import validate_encoded_url

            # Extract relative path from CloudFront URL
            relative_path = extract_relative_path_from_url(enhance_url, cloudfront_base)

            if not relative_path:
                logger.warning(f"Could not extract relative path from: {enhance_url}")
                return ""

            # Generate GitHub URL (now with encoding via updated config)
            github_url = get_github_image_url(relative_path)

            # Validate that the URL is properly encoded
            if github_url and not validate_encoded_url(github_url):
                logger.warning(f"Generated URL may have encoding issues: {github_url}")

            return github_url

        except Exception as e:
            logger.error(f"Error generating GitHub URL for {enhance_url}: {e}")
            return ""

    def update_cache_with_github_urls(self, upload_results: Dict) -> Dict:
        """
        Update cache with GitHub URLs based on upload results.

        Args:
            upload_results: Results from GitHub upload operation

        Returns:
            Dictionary with update results
        """
        if not self.cache_data:
            return {"success": False, "error": "No cache data loaded"}

        logger.info("Updating cache with GitHub URLs")

        # Get successful uploads with their GitHub URLs
        successful_uploads = upload_results.get("copy_results", {}).get(
            "successful_copies_list", []
        )

        if not successful_uploads:
            logger.warning("No successful uploads to process")
            return {"success": False, "error": "No successful uploads found"}

        # Create mapping from local paths to GitHub URLs
        path_to_github_url = {}
        for upload in successful_uploads:
            local_path = upload.get("source", "")
            github_url = upload.get("github_url", "")
            if local_path and github_url:
                path_to_github_url[local_path] = github_url

        # Update cache data
        updated_count = 0
        filename_to_images = self.cache_data.get("filename_to_images", {})

        for filename, images in filename_to_images.items():
            for image in images:
                # Try to find matching GitHub URL for this image
                github_url = self._find_github_url_for_image(image, path_to_github_url)

                if github_url:
                    # Add the public_enhance_url field
                    image["public_enhance_url"] = github_url
                    updated_count += 1
                    logger.debug(
                        f"Updated {filename}: {image.get('enhance_url', '')} -> {github_url}"
                    )
                else:
                    # Generate GitHub URL from enhance_url as fallback
                    enhance_url = image.get("enhance_url", "")
                    if enhance_url:
                        github_url = self.generate_github_url_from_enhance_url(
                            enhance_url, "https://d3u2d4xznamk2r.cloudfront.net"
                        )
                        if github_url:
                            image["public_enhance_url"] = github_url
                            updated_count += 1
                            logger.debug(
                                f"Generated GitHub URL for {filename}: {github_url}"
                            )

        self.updated_images_count = updated_count

        # Update metadata
        metadata = self.cache_data.get("metadata", {})
        metadata["last_updated"] = datetime.now().isoformat()
        metadata["github_urls_added"] = updated_count
        metadata["schema_version"] = (
            "3.1"  # Increment version to indicate GitHub URLs added
        )

        result = {
            "success": True,
            "updated_images": updated_count,
            "total_images": self._count_total_images(),
            "update_percentage": (
                (updated_count / self._count_total_images() * 100)
                if self._count_total_images() > 0
                else 0
            ),
        }

        logger.info(f"Updated {updated_count} images with GitHub URLs")
        return result

    def _find_github_url_for_image(self, image: Dict, path_to_github_url: Dict) -> str:
        """
        Find the corresponding GitHub URL for an image.

        Args:
            image: Image dictionary from cache
            path_to_github_url: Mapping from local paths to GitHub URLs

        Returns:
            GitHub URL if found, empty string otherwise
        """
        # Try to match by enhance_url or src
        enhance_url = image.get("enhance_url", "")
        src = image.get("src", "")

        # Look for exact matches in the path mapping
        for local_path, github_url in path_to_github_url.items():
            # Check if the local path contains parts of the enhance_url or src
            if enhance_url and (
                enhance_url in local_path
                or any(part in local_path for part in enhance_url.split("/") if part)
            ):
                return github_url
            if src and (
                src in local_path
                or any(part in local_path for part in src.split("/") if part)
            ):
                return github_url

        return ""

    def update_cache_directly_from_enhance_urls(self) -> Dict:
        """
        Update cache with GitHub URLs directly from enhance_url patterns.
        This method works without upload results by generating GitHub URLs from CloudFront URLs.

        Returns:
            Dictionary with update results
        """
        if not self.cache_data:
            return {"success": False, "error": "No cache data loaded"}

        logger.info("Updating cache with GitHub URLs from enhance_url patterns")

        # Update cache data
        updated_count = 0
        filename_to_images = self.cache_data.get("filename_to_images", {})

        for filename, images in filename_to_images.items():
            for image in images:
                # Generate GitHub URL from enhance_url
                enhance_url = image.get("enhance_url", "")
                if enhance_url:
                    github_url = self.generate_github_url_from_enhance_url(
                        enhance_url, "https://d3u2d4xznamk2r.cloudfront.net"
                    )
                    if github_url:
                        # Add the public_enhance_url field
                        image["public_enhance_url"] = github_url
                        updated_count += 1
                        logger.debug(
                            f"Generated GitHub URL for {filename}: {enhance_url} -> {github_url}"
                        )
                    else:
                        logger.warning(
                            f"Could not generate GitHub URL for {filename}: {enhance_url}"
                        )

        self.updated_images_count = updated_count

        # Update metadata
        metadata = self.cache_data.get("metadata", {})
        metadata["last_updated"] = datetime.now().isoformat()
        metadata["github_urls_added"] = updated_count
        metadata["schema_version"] = (
            "3.1"  # Increment version to indicate GitHub URLs added
        )

        result = {
            "success": True,
            "updated_images": updated_count,
            "total_images": self._count_total_images(),
            "update_percentage": (
                (updated_count / self._count_total_images() * 100)
                if self._count_total_images() > 0
                else 0
            ),
        }

        logger.info(
            f"Updated {updated_count} images with GitHub URLs from enhance_url patterns"
        )
        return result

    def save_updated_cache(self) -> bool:
        """
        Save the updated cache back to file.

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if not self.cache_data:
                logger.error("No cache data to save")
                return False

            # Write updated cache to file
            with open(IMAGE_MAPPING_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Saved updated cache with {self.updated_images_count} GitHub URLs"
            )
            return True

        except Exception as e:
            logger.error(f"Error saving updated cache: {e}")
            return False

    def rollback_cache(self) -> bool:
        """
        Rollback cache to the backup version.

        Returns:
            True if rollback successful, False otherwise
        """
        try:
            if not self.backup_file:
                logger.error("No backup file available for rollback")
                return False

            backup_path = BACKUP_DIR / self.backup_file.name

            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            # Copy backup back to original location
            import shutil

            shutil.copy2(backup_path, IMAGE_MAPPING_CACHE_FILE)

            logger.info(f"Rolled back cache from backup: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False

    def validate_updated_cache(self) -> Dict:
        """
        Validate the updated cache for consistency.

        Returns:
            Dictionary with validation results
        """
        if not self.cache_data:
            return {"valid": False, "error": "No cache data to validate"}

        try:
            total_images = 0
            images_with_github_urls = 0
            validation_errors = []

            filename_to_images = self.cache_data.get("filename_to_images", {})

            for filename, images in filename_to_images.items():
                for i, image in enumerate(images):
                    total_images += 1

                    # Check required fields
                    required_fields = ["image_url", "enhance_url", "description", "src"]
                    for field in required_fields:
                        if field not in image:
                            validation_errors.append(
                                f"{filename}[{i}]: Missing required field '{field}'"
                            )

                    # Check if GitHub URL was added
                    if "public_enhance_url" in image:
                        images_with_github_urls += 1

                        # Validate GitHub URL format
                        github_url = image["public_enhance_url"]
                        if not github_url.startswith(
                            "https://raw.githubusercontent.com/"
                        ):
                            validation_errors.append(
                                f"{filename}[{i}]: Invalid GitHub URL format: {github_url}"
                            )

            # Check metadata
            metadata = self.cache_data.get("metadata", {})
            if "last_updated" not in metadata:
                validation_errors.append("Missing 'last_updated' in metadata")

            result = {
                "valid": len(validation_errors) == 0,
                "total_images": total_images,
                "images_with_github_urls": images_with_github_urls,
                "github_url_coverage": (
                    (images_with_github_urls / total_images * 100)
                    if total_images > 0
                    else 0
                ),
                "validation_errors": validation_errors[:10],  # Limit to first 10 errors
                "total_errors": len(validation_errors),
            }

            if result["valid"]:
                logger.info(
                    f"Cache validation passed: {images_with_github_urls}/{total_images} images have GitHub URLs"
                )
            else:
                logger.warning(
                    f"Cache validation failed with {len(validation_errors)} errors"
                )

            return result

        except Exception as e:
            logger.error(f"Error during cache validation: {e}")
            return {"valid": False, "error": str(e)}

    def get_update_summary(self) -> Dict:
        """
        Get summary of the cache update operation.

        Returns:
            Dictionary with update summary
        """
        if not self.cache_data:
            return {"error": "No cache data available"}

        metadata = self.cache_data.get("metadata", {})

        return {
            "total_images": self._count_total_images(),
            "updated_images": self.updated_images_count,
            "update_percentage": (
                (self.updated_images_count / self._count_total_images() * 100)
                if self._count_total_images() > 0
                else 0
            ),
            "last_updated": metadata.get("last_updated", "Unknown"),
            "schema_version": metadata.get("schema_version", "Unknown"),
            "backup_file": str(self.backup_file) if self.backup_file else None,
            "cache_file": str(IMAGE_MAPPING_CACHE_FILE),
        }


def create_cache_updater() -> CacheUpdater:
    """
    Create and return a new CacheUpdater instance.

    Returns:
        Configured CacheUpdater instance
    """
    return CacheUpdater()


def update_cache_with_github_urls(upload_results: Dict) -> Dict:
    """
    Convenience function to update cache with GitHub URLs.

    Args:
        upload_results: Results from GitHub upload operation

    Returns:
        Update results dictionary
    """
    updater = create_cache_updater()

    # Load cache
    if not updater.load_cache():
        return {"success": False, "error": "Failed to load cache"}

    # Create backup
    if not updater.create_backup():
        return {"success": False, "error": "Failed to create backup"}

    # Update cache
    update_result = updater.update_cache_with_github_urls(upload_results)

    if not update_result["success"]:
        return update_result

    # Save updated cache
    if not updater.save_updated_cache():
        return {"success": False, "error": "Failed to save updated cache"}

    # Validate updated cache
    validation_result = updater.validate_updated_cache()

    # Combine results
    final_result = update_result.copy()
    final_result["validation"] = validation_result
    final_result["summary"] = updater.get_update_summary()

    return final_result


def update_cache_directly() -> Dict:
    """
    Convenience function to update cache with GitHub URLs directly from enhance_url patterns.
    This function works without upload results by generating GitHub URLs from CloudFront URLs.

    Returns:
        Update results dictionary
    """
    updater = create_cache_updater()

    # Load cache
    if not updater.load_cache():
        return {"success": False, "error": "Failed to load cache"}

    # Create backup
    if not updater.create_backup():
        return {"success": False, "error": "Failed to create backup"}

    # Update cache directly from enhance_url patterns
    update_result = updater.update_cache_directly_from_enhance_urls()

    if not update_result["success"]:
        return update_result

    # Save updated cache
    if not updater.save_updated_cache():
        return {"success": False, "error": "Failed to save updated cache"}

    # Validate updated cache
    validation_result = updater.validate_updated_cache()

    # Combine results
    final_result = update_result.copy()
    final_result["validation"] = validation_result
    final_result["summary"] = updater.get_update_summary()

    return final_result
