#!/usr/bin/env python3
"""
Main orchestrator for the Image Downloader system.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import glob

from .config import (
    LOG_LEVEL,
    LOG_FILE,
    ensure_directories,
    TEMP_DOWNLOAD_DIR,
    ALLOWED_IMAGE_EXTENSIONS,
    GITHUB_REPO_URL,
    GITHUB_REPO_NAME,
)
from .image_downloader import create_image_downloader, download_images_from_cache
from .cache_updater import create_cache_updater, update_cache_directly
from .utils import format_file_size, estimate_download_time


# Configure logging
def setup_logging():
    """Setup logging configuration."""
    # Ensure log directory exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


logger = logging.getLogger(__name__)


class ImageDownloadOrchestrator:
    """Main orchestrator class for the complete image download and upload process."""

    def __init__(self):
        """Initialize the orchestrator."""
        self.downloader = None
        self.uploader = None
        self.cache_updater = None
        self.start_time = None
        self.results = {}

        # Ensure all directories exist
        ensure_directories()

    def _check_existing_downloads(self) -> Optional[Dict]:
        """
        Simplified check for existing downloaded images.

        Returns:
            Mock download results if images exist, None otherwise
        """
        if not TEMP_DOWNLOAD_DIR.exists():
            return None

        # Count image files and calculate total size
        image_count = 0
        total_bytes = 0

        for ext in ALLOWED_IMAGE_EXTENSIONS:
            pattern = str(TEMP_DOWNLOAD_DIR / "**" / f"*{ext}")
            files = glob.glob(pattern, recursive=True)
            image_count += len(files)

            for file_path in files:
                try:
                    total_bytes += Path(file_path).stat().st_size
                except OSError:
                    continue

        if image_count == 0:
            return None

        # Return simplified mock results
        logger.info(
            f"Found {image_count} existing images ({format_file_size(total_bytes)})"
        )

        return {
            "success": True,
            "total_images": image_count,
            "completed": image_count,
            "failed": 0,
            "skipped": 0,
            "success_rate": 100.0,
            "total_bytes": total_bytes,
            "successful_downloads": [
                {"local_path": "existing"} for _ in range(image_count)
            ],
            "failed_downloads": [],
            "skipped_downloads": [],
        }

    def run_complete_process(self) -> Dict:
        """
        Run the complete image download and cache update process.

        Returns:
            Dictionary with complete process results
        """
        logger.info("=" * 60)
        logger.info("STARTING FLEXIT RAG IMAGE DOWNLOADER")
        logger.info("=" * 60)

        self.start_time = time.time()

        try:
            # Check for existing downloads first
            existing_downloads = self._check_existing_downloads()

            if existing_downloads:
                # Phase 1: Skip download - images already exist
                logger.info(
                    "🔥 **PHASE 1: DOWNLOAD IMAGES - SKIPPED (IMAGES ALREADY EXIST)** 🔥"
                )
                logger.info("=" * 60)
                download_results = existing_downloads
            else:
                # Phase 1: Download images
                logger.info("Phase 1: Downloading images from CloudFront")
                download_results = self._download_images()

                if not download_results.get("success", False):
                    return self._create_final_results(
                        success=False,
                        error="Download phase failed",
                        download_results=download_results,
                    )

            # Phase 2: Manual GitHub Upload Instructions
            self._show_manual_upload_instructions(download_results)

            # Phase 3: Update cache with GitHub URLs
            logger.info("Phase 3: Updating cache with GitHub URLs")
            cache_update_results = self._update_cache_directly()

            if not cache_update_results.get("success", False):
                logger.warning("Cache update failed")

            # Generate final results
            final_results = self._create_final_results(
                success=True,
                download_results=download_results,
                cache_update_results=cache_update_results,
            )

            self._log_completion_summary(final_results)
            return final_results

        except KeyboardInterrupt:
            logger.warning("Process interrupted by user")
            return self._create_final_results(
                success=False, error="Process interrupted by user"
            )

        except Exception as e:
            logger.error(f"Unexpected error during process: {e}")
            return self._create_final_results(
                success=False, error=f"Unexpected error: {str(e)}"
            )

    def _download_images(self) -> Dict:
        """Download images from CloudFront URLs."""
        try:
            self.downloader = create_image_downloader()

            # Load cache and get image count for estimation
            if not self.downloader.load_image_mapping_cache():
                return {"success": False, "error": "Failed to load image mapping cache"}

            total_images = self.downloader._count_total_images()
            logger.info(f"Found {total_images} images to download")

            # Estimate download time
            avg_file_size = 500 * 1024  # Assume 500KB average
            estimated_seconds, time_str = estimate_download_time(
                total_images, avg_file_size
            )
            logger.info(f"Estimated download time: {time_str}")

            # Start download
            download_start = time.time()
            results = self.downloader.download_all_images()
            download_time = time.time() - download_start

            # Add timing information
            results["download_time"] = download_time
            results["download_time_formatted"] = f"{download_time:.1f} seconds"

            return results

        except Exception as e:
            logger.error(f"Error during image download: {e}")
            return {"success": False, "error": str(e)}

    def _show_manual_upload_instructions(self, download_results: Dict):
        """Show instructions for manual GitHub upload."""
        successful_downloads = download_results.get("successful_downloads", [])
        total_files = len(successful_downloads)
        total_size = download_results.get("total_bytes", 0)

        logger.info("=" * 80)
        logger.info("🔥 **PHASE 2: MANUAL GITHUB UPLOAD REQUIRED** 🔥")
        logger.info("=" * 80)
        logger.info(f"Repository: {GITHUB_REPO_URL}")
        logger.info(f"Repository Name: {GITHUB_REPO_NAME}")
        logger.info(f"Source Directory: {TEMP_DOWNLOAD_DIR}")
        logger.info("Target Directory: images/ (in the GitHub repo)")
        logger.info(
            f"Files to Upload: {total_files} files ({format_file_size(total_size)})"
        )
        logger.info(
            "Phase 3: Started to created 'public_enhance_url' using the enhance_url patterns in the cache"
        )
        logger.info("=" * 80)

    def _update_cache_directly(self) -> Dict:
        """Update cache with GitHub URLs directly from enhance_url patterns."""
        try:
            self.cache_updater = create_cache_updater()

            logger.info("Updating image mapping cache with GitHub URLs")

            # Start cache update
            cache_start = time.time()
            results = update_cache_directly()
            cache_time = time.time() - cache_start

            # Add timing information
            results["cache_update_time"] = cache_time
            results["cache_update_time_formatted"] = f"{cache_time:.1f} seconds"

            return results

        except Exception as e:
            logger.error(f"Error during cache update: {e}")
            return {"success": False, "error": str(e)}

    def _create_final_results(
        self, success: bool, error: Optional[str] = None, **phase_results
    ) -> Dict:
        """Create final results dictionary."""
        total_time = time.time() - self.start_time if self.start_time else 0

        results = {
            "success": success,
            "total_time": total_time,
            "total_time_formatted": f"{total_time:.1f} seconds",
            "timestamp": datetime.now().isoformat(),
            "phases": {},
        }

        if error:
            results["error"] = error

        # Add phase results
        for phase_name, phase_result in phase_results.items():
            if phase_result:
                results["phases"][phase_name] = phase_result

        # Calculate summary statistics
        if success:
            results["summary"] = self._calculate_summary_stats(phase_results)

        return results

    def _calculate_summary_stats(self, phase_results: Dict) -> Dict:
        """Calculate summary statistics from all phases."""
        summary = {
            "total_images_processed": 0,
            "total_images_downloaded": 0,
            "total_images_uploaded": 0,
            "total_images_with_github_urls": 0,
            "total_bytes_downloaded": 0,
            "total_bytes_uploaded": 0,
            "success_rate": 0.0,
        }

        # Download statistics
        download_results = phase_results.get("download_results", {})
        if download_results:
            summary["total_images_processed"] = download_results.get("total_images", 0)
            summary["total_images_downloaded"] = download_results.get("completed", 0)
            summary["total_bytes_downloaded"] = download_results.get("total_bytes", 0)

            if summary["total_images_processed"] > 0:
                summary["success_rate"] = (
                    summary["total_images_downloaded"]
                    / summary["total_images_processed"]
                ) * 100

        # Upload statistics
        upload_results = phase_results.get("upload_results", {})
        if upload_results:
            summary["total_images_uploaded"] = upload_results.get(
                "total_files_uploaded", 0
            )
            summary["total_bytes_uploaded"] = upload_results.get(
                "total_bytes_uploaded", 0
            )

        # Cache update statistics
        cache_results = phase_results.get("cache_update_results", {})
        if cache_results:
            summary["total_images_with_github_urls"] = cache_results.get(
                "updated_images", 0
            )

        # Format file sizes
        summary["total_bytes_downloaded_formatted"] = format_file_size(
            summary["total_bytes_downloaded"]
        )
        summary["total_bytes_uploaded_formatted"] = format_file_size(
            summary["total_bytes_uploaded"]
        )

        return summary

    def _log_completion_summary(self, results: Dict):
        """Log completion summary."""
        logger.info("=" * 60)
        logger.info("PROCESS COMPLETED")
        logger.info("=" * 60)

        if results["success"]:
            summary = results.get("summary", {})
            logger.info(
                f"✓ Total images processed: {summary.get('total_images_processed', 0)}"
            )
            logger.info(
                f"✓ Images downloaded: {summary.get('total_images_downloaded', 0)}"
            )
            logger.info(
                f"✓ Images uploaded to GitHub: {summary.get('total_images_uploaded', 0)}"
            )
            logger.info(
                f"✓ Images with GitHub URLs: {summary.get('total_images_with_github_urls', 0)}"
            )
            logger.info(
                f"✓ Data downloaded: {summary.get('total_bytes_downloaded_formatted', '0 B')}"
            )
            logger.info(
                f"✓ Data uploaded: {summary.get('total_bytes_uploaded_formatted', '0 B')}"
            )
            logger.info(f"✓ Success rate: {summary.get('success_rate', 0):.1f}%")
            logger.info(f"✓ Total time: {results['total_time_formatted']}")
        else:
            logger.error(f"✗ Process failed: {results.get('error', 'Unknown error')}")

        logger.info("=" * 60)

    def get_process_status(self) -> Dict:
        """Get current process status."""
        if not self.start_time:
            return {"status": "not_started"}

        elapsed_time = time.time() - self.start_time

        status = {
            "status": "running",
            "elapsed_time": elapsed_time,
            "elapsed_time_formatted": f"{elapsed_time:.1f} seconds",
        }

        # Add phase-specific status
        if (
            self.downloader
            and hasattr(self.downloader, "progress")
            and self.downloader.progress
        ):
            status["download_progress"] = self.downloader.progress.get_stats()

        if self.uploader:
            status["upload_summary"] = self.uploader.get_upload_summary()

        if self.cache_updater:
            status["cache_summary"] = self.cache_updater.get_update_summary()

        return status


def main():
    """Main entry point."""
    # Setup logging
    setup_logging()

    try:
        # Create and run orchestrator
        orchestrator = ImageDownloadOrchestrator()
        results = orchestrator.run_complete_process()

        # Exit with appropriate code
        exit_code = 0 if results["success"] else 1
        sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


def run_download_only() -> Dict:
    """
    Convenience function to run only the download phase.

    Returns:
        Download results dictionary
    """
    setup_logging()
    logger.info("Running download-only mode")

    orchestrator = ImageDownloadOrchestrator()
    return orchestrator._download_images()


def run_cache_update_only() -> Dict:
    """
    Convenience function to run only the cache update phase.

    Returns:
        Cache update results dictionary
    """
    setup_logging()
    logger.info("Running cache-update-only mode")

    orchestrator = ImageDownloadOrchestrator()
    return orchestrator._update_cache_directly()


if __name__ == "__main__":
    main()
