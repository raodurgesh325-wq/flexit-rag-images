#!/usr/bin/env python3
"""
Image Downloader module for downloading images from CloudFront URLs using async Playwright.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import threading

from playwright.async_api import async_playwright, Browser, BrowserContext

from .config import (
    IMAGE_MAPPING_CACHE_FILE,
    TEMP_DOWNLOAD_DIR,
    CLOUDFRONT_BASE_URL,
    DOWNLOAD_BATCH_SIZE,
    MAX_RETRY_ATTEMPTS,
    RETRY_DELAY,
    REQUEST_TIMEOUT,
    MAX_FILE_SIZE,
    ALLOWED_IMAGE_EXTENSIONS,
    PROGRESS_UPDATE_INTERVAL,
    CHECKPOINT_INTERVAL,
    CONTINUE_ON_ERROR,
    MAX_CONSECUTIVE_ERRORS,
    ensure_directories,
)
from .utils import (
    sanitize_filename,
    extract_relative_path_from_url,
    create_directory_structure,
    is_valid_image_extension,
    generate_unique_filename,
    format_file_size,
    validate_url,
)

# Configure logging
logger = logging.getLogger(__name__)


class DownloadProgress:
    """Thread-safe progress tracker for downloads."""

    def __init__(self, total_images: int):
        self.total_images = total_images
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.total_bytes = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def update(
        self, success: bool = True, bytes_downloaded: int = 0, skipped: bool = False
    ):
        """Update progress counters."""
        with self.lock:
            if skipped:
                self.skipped += 1
            elif success:
                self.completed += 1
                self.total_bytes += bytes_downloaded
            else:
                self.failed += 1

    def get_stats(self) -> Dict:
        """Get current progress statistics."""
        with self.lock:
            elapsed_time = time.time() - self.start_time
            processed = self.completed + self.failed + self.skipped

            return {
                "total": self.total_images,
                "completed": self.completed,
                "failed": self.failed,
                "skipped": self.skipped,
                "processed": processed,
                "remaining": self.total_images - processed,
                "success_rate": (
                    (self.completed / processed * 100) if processed > 0 else 0
                ),
                "total_bytes": self.total_bytes,
                "elapsed_time": elapsed_time,
                "avg_speed": (self.completed / elapsed_time) if elapsed_time > 0 else 0,
            }


class ImageDownloader:
    """Main class for downloading images from the mapping cache using async Playwright."""

    def __init__(self):
        """Initialize the image downloader."""
        self.cache_data = None
        self.progress = None

        # Download tracking
        self.download_results = []
        self.consecutive_errors = 0

        # Ensure directories exist
        ensure_directories()

    def load_image_mapping_cache(self) -> bool:
        """
        Load the image mapping cache from JSON file.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not IMAGE_MAPPING_CACHE_FILE.exists():
                logger.error(
                    f"Image mapping cache file not found: {IMAGE_MAPPING_CACHE_FILE}"
                )
                return False

            with open(IMAGE_MAPPING_CACHE_FILE, "r", encoding="utf-8") as f:
                self.cache_data = json.load(f)

            logger.info(
                f"Loaded image mapping cache with {self._count_total_images()} images"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading image mapping cache: {e}")
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

    def _get_all_images(self) -> List[Tuple[str, Dict]]:
        """
        Get all images from cache with their source filenames.

        Returns:
            List of tuples (filename, image_dict)
        """
        images = []
        filename_to_images = self.cache_data.get("filename_to_images", {})

        for filename, image_list in filename_to_images.items():
            for image in image_list:
                images.append((filename, image))

        return images

    def _create_local_path(self, filename: str, image: Dict) -> Path:
        """
        Create local file path for an image.

        Args:
            filename: Source HTML filename
            image: Image dictionary

        Returns:
            Local file path where image should be saved
        """
        # Extract relative path from enhance_url
        enhance_url = image.get("enhance_url", "")
        relative_path = extract_relative_path_from_url(enhance_url, CLOUDFRONT_BASE_URL)

        if not relative_path:
            # Fallback: use src attribute
            src = image.get("src", "")
            if src:
                relative_path = src.lstrip("/")
            else:
                # Last resort: generate from URL
                parsed = urlparse(enhance_url)
                relative_path = parsed.path.lstrip("/")

        # Sanitize the path components
        path_parts = relative_path.split("/")
        sanitized_parts = [sanitize_filename(part) for part in path_parts if part]

        # Create the local path
        local_path = TEMP_DOWNLOAD_DIR
        for part in sanitized_parts:
            local_path = local_path / part

        return local_path

    async def _download_single_image_async(
        self,
        context: BrowserContext,
        url: str,
        local_path: Path,
        max_retries: int = MAX_RETRY_ATTEMPTS,
    ) -> Tuple[bool, int, str]:
        """
        Download a single image using Playwright with retry logic.

        Args:
            context: Browser context for the download
            url: Image URL to download
            local_path: Local path to save the image
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (success, bytes_downloaded, error_message)
        """
        for attempt in range(max_retries + 1):
            try:
                # Import URL encoder for proper encoding
                from crawler.url.url_encoder import safe_encode_url

                # Ensure URL is properly encoded
                encoded_url = safe_encode_url(url)

                # Validate URL
                if not validate_url(encoded_url):
                    return False, 0, f"Invalid URL: {encoded_url}"

                # Check if file already exists
                if local_path.exists():
                    file_size = local_path.stat().st_size
                    logger.debug(
                        f"File already exists: {local_path} ({format_file_size(file_size)})"
                    )
                    return True, file_size, ""

                # Create directory structure
                create_directory_structure(local_path)

                # Create a new page for this download
                page = await context.new_page()

                try:
                    # Set timeout and navigate to the image URL
                    page.set_default_timeout(
                        REQUEST_TIMEOUT * 1000
                    )  # Convert to milliseconds

                    # Navigate to the image URL (use encoded URL)
                    response = await page.goto(encoded_url, wait_until="networkidle")

                    if not response:
                        return False, 0, f"Failed to get response for {encoded_url}"

                    if not response.ok:
                        return (
                            False,
                            0,
                            f"HTTP {response.status}: {response.status_text}",
                        )

                    # Get the response body
                    content = await response.body()

                    # Check content length
                    if len(content) > MAX_FILE_SIZE:
                        return (
                            False,
                            0,
                            f"File too large: {format_file_size(len(content))}",
                        )

                    # Check content type
                    content_type = response.headers.get("content-type", "").lower()
                    if not any(
                        img_type in content_type
                        for img_type in ["image/", "application/octet-stream"]
                    ):
                        logger.warning(
                            f"Unexpected content type for {url}: {content_type}"
                        )

                    # Save the file
                    with open(local_path, "wb") as f:
                        f.write(content)

                    bytes_downloaded = len(content)
                    logger.debug(
                        f"Downloaded: {url} -> {local_path} ({format_file_size(bytes_downloaded)})"
                    )
                    return True, bytes_downloaded, ""

                finally:
                    await page.close()

            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}"
                logger.debug(error_msg)

                if attempt < max_retries:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                else:
                    return False, 0, f"All retry attempts failed. Last error: {str(e)}"

        return False, 0, "Maximum retry attempts exceeded"

    async def _download_image_worker_async(
        self, context: BrowserContext, args: Tuple[str, str, Dict]
    ) -> Dict:
        """
        Async worker function for downloading a single image.

        Args:
            context: Browser context for downloads
            args: Tuple of (filename, enhance_url, image_dict)

        Returns:
            Download result dictionary
        """
        filename, enhance_url, image = args

        try:
            # Create local path
            local_path = self._create_local_path(filename, image)

            # Check if it's a valid image extension
            if not is_valid_image_extension(str(local_path), ALLOWED_IMAGE_EXTENSIONS):
                self.progress.update(skipped=True)
                return {
                    "success": False,
                    "skipped": True,
                    "filename": filename,
                    "url": enhance_url,
                    "local_path": str(local_path),
                    "error": "Invalid image extension",
                }

            # Generate unique filename if needed
            if local_path.exists():
                unique_filename = generate_unique_filename(
                    local_path.parent, local_path.name
                )
                local_path = local_path.parent / unique_filename

            # Download the image
            success, bytes_downloaded, error_msg = (
                await self._download_single_image_async(
                    context, enhance_url, local_path
                )
            )

            # Update progress
            self.progress.update(success=success, bytes_downloaded=bytes_downloaded)

            # Track consecutive errors
            if success:
                self.consecutive_errors = 0
            else:
                self.consecutive_errors += 1

            result = {
                "success": success,
                "skipped": False,
                "filename": filename,
                "url": enhance_url,
                "local_path": str(local_path),
                "bytes_downloaded": bytes_downloaded,
                "error": error_msg,
            }

            return result

        except Exception as e:
            self.progress.update(success=False)
            self.consecutive_errors += 1

            return {
                "success": False,
                "skipped": False,
                "filename": filename,
                "url": enhance_url,
                "local_path": "",
                "error": f"Worker error: {str(e)}",
            }

    async def _process_images_in_batches(
        self, browser: Browser, download_tasks: List[Tuple[str, str, Dict]]
    ):
        """
        Process images in concurrent batches using async Playwright.

        Args:
            browser: Playwright browser instance
            download_tasks: List of download tasks
        """
        # Create a browser context for downloads
        context = await browser.new_context()

        try:
            # Create semaphore to limit concurrent downloads
            semaphore = asyncio.Semaphore(DOWNLOAD_BATCH_SIZE)

            async def download_with_semaphore(task):
                async with semaphore:
                    return await self._download_image_worker_async(context, task)

            # Process all tasks concurrently with semaphore limiting
            tasks = [download_with_semaphore(task) for task in download_tasks]

            # Process tasks and collect results
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                if not CONTINUE_ON_ERROR and self._should_stop_on_errors():
                    logger.error(
                        f"Stopping due to {self.consecutive_errors} consecutive errors"
                    )
                    break

                try:
                    result = await coro
                    self.download_results.append(result)

                    # Log progress periodically
                    self._log_progress()

                    # Save checkpoint periodically
                    if len(self.download_results) % CHECKPOINT_INTERVAL == 0:
                        self._save_checkpoint()

                except Exception as e:
                    logger.error(f"Error processing download result: {e}")
                    self.progress.update(success=False)

        finally:
            await context.close()

    def _should_stop_on_errors(self) -> bool:
        """Check if we should stop due to too many consecutive errors."""
        return self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS

    def _log_progress(self, force: bool = False):
        """Log current progress."""
        stats = self.progress.get_stats()

        if force or stats["processed"] % PROGRESS_UPDATE_INTERVAL == 0:
            logger.info(
                f"Progress: {stats['processed']}/{stats['total']} "
                f"(Success: {stats['completed']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}) "
                f"Success Rate: {stats['success_rate']:.1f}% "
                f"Speed: {stats['avg_speed']:.1f} images/sec "
                f"Downloaded: {format_file_size(stats['total_bytes'])}"
            )

    async def download_all_images_async(self) -> Dict:
        """
        Download all images from the mapping cache using async Playwright.

        Returns:
            Dictionary with download results and statistics
        """
        if not self.cache_data:
            logger.error("No cache data loaded. Call load_image_mapping_cache() first.")
            return {"success": False, "error": "No cache data loaded"}

        # Get all images to download
        all_images = self._get_all_images()
        total_images = len(all_images)

        if total_images == 0:
            logger.warning("No images found in cache")
            return {"success": True, "total_images": 0, "results": []}

        logger.info(f"Starting download of {total_images} images")

        # Initialize progress tracker
        self.progress = DownloadProgress(total_images)

        # Prepare download tasks
        download_tasks = []
        for filename, image in all_images:
            enhance_url = image.get("enhance_url", "")
            if enhance_url:
                download_tasks.append((filename, enhance_url, image))

        logger.info(f"Prepared {len(download_tasks)} download tasks")

        # Download images using async Playwright
        self.download_results = []

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=False)

                try:
                    # Process images in concurrent batches
                    await self._process_images_in_batches(browser, download_tasks)
                except Exception as e:
                    logger.error(f"Error during batch processing: {e}")
                finally:
                    await browser.close()

        except KeyboardInterrupt:
            logger.warning("Download interrupted by user")
        except Exception as e:
            logger.error(f"Error during download process: {e}")

        # Final progress log
        self._log_progress(force=True)

        # Generate final statistics
        stats = self.progress.get_stats()

        # Categorize results
        successful_downloads = [r for r in self.download_results if r["success"]]
        failed_downloads = [
            r for r in self.download_results if not r["success"] and not r["skipped"]
        ]
        skipped_downloads = [r for r in self.download_results if r["skipped"]]

        result = {
            "success": True,
            "total_images": total_images,
            "completed": stats["completed"],
            "failed": stats["failed"],
            "skipped": stats["skipped"],
            "success_rate": stats["success_rate"],
            "total_bytes": stats["total_bytes"],
            "elapsed_time": stats["elapsed_time"],
            "avg_speed": stats["avg_speed"],
            "successful_downloads": successful_downloads,
            "failed_downloads": failed_downloads,
            "skipped_downloads": skipped_downloads,
            "download_directory": str(TEMP_DOWNLOAD_DIR),
        }

        logger.info(
            f"Download completed: {stats['completed']}/{total_images} images successfully downloaded"
        )
        return result

    def download_all_images(self) -> Dict:
        """
        Synchronous wrapper for the async download method.

        Returns:
            Dictionary with download results and statistics
        """
        return asyncio.run(self.download_all_images_async())

    def _save_checkpoint(self):
        """Save download progress checkpoint."""
        try:
            checkpoint_file = TEMP_DOWNLOAD_DIR / "download_checkpoint.json"
            checkpoint_data = {
                "timestamp": time.time(),
                "completed_downloads": len(
                    [r for r in self.download_results if r["success"]]
                ),
                "total_results": len(self.download_results),
                "stats": self.progress.get_stats() if self.progress else {},
            }

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)

            logger.debug(
                f"Checkpoint saved: {checkpoint_data['completed_downloads']} downloads completed"
            )

        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def get_download_summary(self) -> Dict:
        """
        Get summary of the last download operation.

        Returns:
            Dictionary with download summary
        """
        if not self.progress:
            return {"error": "No download operation completed"}

        stats = self.progress.get_stats()

        return {
            "total_images": stats["total"],
            "completed": stats["completed"],
            "failed": stats["failed"],
            "skipped": stats["skipped"],
            "success_rate": stats["success_rate"],
            "total_bytes_downloaded": stats["total_bytes"],
            "total_size_formatted": format_file_size(stats["total_bytes"]),
            "elapsed_time": stats["elapsed_time"],
            "elapsed_time_formatted": f"{stats['elapsed_time']:.1f} seconds",
            "average_speed": stats["avg_speed"],
            "download_directory": str(TEMP_DOWNLOAD_DIR),
        }


def create_image_downloader() -> ImageDownloader:
    """
    Create and return a new ImageDownloader instance.

    Returns:
        Configured ImageDownloader instance
    """
    return ImageDownloader()


def download_images_from_cache() -> Dict:
    """
    Convenience function to download all images from cache.

    Returns:
        Download results dictionary
    """
    downloader = create_image_downloader()

    if not downloader.load_image_mapping_cache():
        return {"success": False, "error": "Failed to load image mapping cache"}

    return downloader.download_all_images()
