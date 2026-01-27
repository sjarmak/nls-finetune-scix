"""HTTP downloader with caching and checksum verification.

This module handles downloading HTTP(S) sources declared in sources.yaml:
1. Downloads files into raw/ directory with deterministic filenames
2. Computes and records SHA256 checksums
3. Refuses to overwrite cached files if checksum matches
4. Supports If-Modified-Since/ETag conditional requests
5. Records response headers in source manifest

Example usage:
    downloader = HTTPDownloader(cache_dir=Path("./cache"))
    result = downloader.download(source_config, raw_dir=Path("./raw"))
    # result contains: local_path, checksum, etag, last_modified, etc.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import httpx

from finetune.dataset_agent.schemas import SourceEntry, SourceType
from finetune.dataset_agent.sources import (
    SourceConfig,
    create_source_entry,
    get_deterministic_filename,
)

logger = logging.getLogger(__name__)


class HTTPDownloadError(Exception):
    """Error during HTTP download."""

    def __init__(
        self,
        message: str,
        source_id: str | None = None,
        url: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.source_id = source_id
        self.url = url
        self.status_code = status_code
        context = []
        if source_id:
            context.append(f"source_id={source_id}")
        if url:
            context.append(f"url={url}")
        if status_code:
            context.append(f"status={status_code}")
        full_message = f"{message} [{', '.join(context)}]" if context else message
        super().__init__(full_message)


@dataclass
class DownloadResult:
    """Result of a download operation.

    Contains all metadata needed to populate a SourceEntry.
    """

    local_path: Path  # Path where file was saved
    checksum_sha256: str  # SHA256 of downloaded content
    size_bytes: int  # File size in bytes
    retrieved_at: str  # ISO 8601 timestamp
    etag: str | None = None  # HTTP ETag header if present
    last_modified: str | None = None  # HTTP Last-Modified header if present
    content_type: str | None = None  # HTTP Content-Type header if present
    from_cache: bool = False  # Whether file was served from cache


@dataclass
class CacheMetadata:
    """Metadata stored alongside cached files for conditional requests."""

    checksum_sha256: str
    etag: str | None = None
    last_modified: str | None = None
    downloaded_at: str | None = None


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_stream_sha256(response: httpx.Response, output_file: BinaryIO) -> tuple[str, int]:
    """Stream response content to file while computing SHA256.

    Args:
        response: HTTP response with streaming enabled
        output_file: File handle to write to

    Returns:
        Tuple of (SHA256 hex digest, total bytes written)
    """
    hasher = hashlib.sha256()
    total_bytes = 0

    for chunk in response.iter_bytes(chunk_size=8192):
        output_file.write(chunk)
        hasher.update(chunk)
        total_bytes += len(chunk)

    return hasher.hexdigest(), total_bytes


class HTTPDownloader:
    """Downloader for HTTP(S) sources with caching and checksum verification.

    Features:
    - SHA256 checksum computation for all downloads
    - Cache hit detection: skips download if local file checksum matches
    - Conditional requests using If-Modified-Since and If-None-Match headers
    - Configurable timeout and retry behavior
    """

    def __init__(
        self,
        timeout: float = 60.0,
        follow_redirects: bool = True,
        user_agent: str = "dataset-agent/1.0",
    ) -> None:
        """Initialize the HTTP downloader.

        Args:
            timeout: Request timeout in seconds
            follow_redirects: Whether to follow HTTP redirects
            user_agent: User-Agent header value
        """
        self.timeout = timeout
        self.follow_redirects = follow_redirects
        self.user_agent = user_agent

    def _get_client(self) -> httpx.Client:
        """Create a configured HTTP client."""
        return httpx.Client(
            timeout=self.timeout,
            follow_redirects=self.follow_redirects,
            headers={"User-Agent": self.user_agent},
        )

    def _build_conditional_headers(
        self,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> dict[str, str]:
        """Build conditional request headers for cache validation.

        Args:
            etag: Previous ETag value (for If-None-Match)
            last_modified: Previous Last-Modified value (for If-Modified-Since)

        Returns:
            Dict of headers to send
        """
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        return headers

    def _check_cache_valid(
        self,
        local_path: Path,
        expected_checksum: str | None = None,
    ) -> bool:
        """Check if a cached file is valid.

        Args:
            local_path: Path to the cached file
            expected_checksum: Expected SHA256 checksum (if known)

        Returns:
            True if cache is valid and should be used
        """
        if not local_path.exists():
            return False

        if expected_checksum:
            actual_checksum = compute_file_sha256(local_path)
            return actual_checksum == expected_checksum

        # If no expected checksum, cache is valid if file exists and is non-empty
        return local_path.stat().st_size > 0

    def download(
        self,
        source_config: SourceConfig,
        raw_dir: Path,
        expected_checksum: str | None = None,
        cached_etag: str | None = None,
        cached_last_modified: str | None = None,
    ) -> DownloadResult:
        """Download an HTTP source to the raw directory.

        Args:
            source_config: Source configuration from sources.yaml
            raw_dir: Directory to store downloaded files
            expected_checksum: If provided, skip download if local file matches this checksum
            cached_etag: Previous ETag for conditional request
            cached_last_modified: Previous Last-Modified for conditional request

        Returns:
            DownloadResult with download metadata

        Raises:
            HTTPDownloadError: On download failure
        """
        if source_config.type != SourceType.HTTP:
            raise HTTPDownloadError(
                f"Cannot download non-HTTP source type: {source_config.type.value}",
                source_id=source_config.id,
            )

        # Determine local path
        filename = get_deterministic_filename(source_config)
        local_path = raw_dir / filename

        # Ensure raw directory exists
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Check if we can use cached file based on checksum
        if expected_checksum and self._check_cache_valid(local_path, expected_checksum):
            logger.info(
                f"Cache hit for {source_config.id}: checksum matches {expected_checksum[:16]}..."
            )
            return DownloadResult(
                local_path=local_path,
                checksum_sha256=expected_checksum,
                size_bytes=local_path.stat().st_size,
                retrieved_at=datetime.now(UTC).isoformat(),
                etag=cached_etag,
                last_modified=cached_last_modified,
                from_cache=True,
            )

        # Build conditional headers
        conditional_headers = self._build_conditional_headers(
            etag=cached_etag,
            last_modified=cached_last_modified,
        )

        logger.info(f"Downloading {source_config.id} from {source_config.url}")

        try:
            with self._get_client() as client:
                # Use streaming to handle large files efficiently
                with client.stream("GET", source_config.url, headers=conditional_headers) as response:
                    # Handle 304 Not Modified
                    if response.status_code == 304:
                        if not local_path.exists():
                            raise HTTPDownloadError(
                                "Server returned 304 Not Modified but local file doesn't exist",
                                source_id=source_config.id,
                                url=source_config.url,
                                status_code=304,
                            )

                        actual_checksum = compute_file_sha256(local_path)
                        logger.info(f"Not modified (304) for {source_config.id}")

                        return DownloadResult(
                            local_path=local_path,
                            checksum_sha256=actual_checksum,
                            size_bytes=local_path.stat().st_size,
                            retrieved_at=datetime.now(UTC).isoformat(),
                            etag=cached_etag,
                            last_modified=cached_last_modified,
                            from_cache=True,
                        )

                    # Check for HTTP errors
                    if response.status_code >= 400:
                        raise HTTPDownloadError(
                            f"HTTP {response.status_code}: {response.reason_phrase}",
                            source_id=source_config.id,
                            url=source_config.url,
                            status_code=response.status_code,
                        )

                    # Extract response headers
                    etag = response.headers.get("ETag")
                    last_modified = response.headers.get("Last-Modified")
                    content_type = response.headers.get("Content-Type")

                    # Stream to file while computing checksum
                    with open(local_path, "wb") as f:
                        checksum, size_bytes = compute_stream_sha256(response, f)

                    logger.info(
                        f"Downloaded {source_config.id}: {size_bytes} bytes, "
                        f"checksum={checksum[:16]}..."
                    )

                    return DownloadResult(
                        local_path=local_path,
                        checksum_sha256=checksum,
                        size_bytes=size_bytes,
                        retrieved_at=datetime.now(UTC).isoformat(),
                        etag=etag,
                        last_modified=last_modified,
                        content_type=content_type,
                        from_cache=False,
                    )

        except httpx.TimeoutException as e:
            raise HTTPDownloadError(
                f"Request timeout: {e}",
                source_id=source_config.id,
                url=source_config.url,
            ) from e
        except httpx.RequestError as e:
            raise HTTPDownloadError(
                f"Request failed: {e}",
                source_id=source_config.id,
                url=source_config.url,
            ) from e

    def download_all(
        self,
        sources_config: "list[SourceConfig]",
        raw_dir: Path,
        previous_manifest: "dict[str, SourceEntry] | None" = None,
    ) -> list[DownloadResult]:
        """Download all HTTP sources from configuration.

        Args:
            sources_config: List of source configurations
            raw_dir: Directory to store downloaded files
            previous_manifest: Optional dict mapping source_id to previous SourceEntry
                              for conditional request headers and checksum validation

        Returns:
            List of DownloadResult for each HTTP source
        """
        results = []
        previous_manifest = previous_manifest or {}

        for source in sources_config:
            if source.type != SourceType.HTTP:
                logger.debug(f"Skipping non-HTTP source: {source.id}")
                continue

            # Get cached metadata if available
            previous_entry = previous_manifest.get(source.id)
            expected_checksum = previous_entry.checksum_sha256 if previous_entry else None
            cached_etag = previous_entry.etag if previous_entry else None
            cached_last_modified = previous_entry.last_modified if previous_entry else None

            try:
                result = self.download(
                    source,
                    raw_dir,
                    expected_checksum=expected_checksum,
                    cached_etag=cached_etag,
                    cached_last_modified=cached_last_modified,
                )
                results.append(result)
            except HTTPDownloadError as e:
                logger.error(f"Failed to download {source.id}: {e}")
                raise

        return results


def download_http_source(
    source_config: SourceConfig,
    raw_dir: Path,
    expected_checksum: str | None = None,
    cached_etag: str | None = None,
    cached_last_modified: str | None = None,
) -> DownloadResult:
    """Convenience function to download a single HTTP source.

    Args:
        source_config: Source configuration from sources.yaml
        raw_dir: Directory to store downloaded files
        expected_checksum: If provided, skip download if local file matches
        cached_etag: Previous ETag for conditional request
        cached_last_modified: Previous Last-Modified for conditional request

    Returns:
        DownloadResult with download metadata
    """
    downloader = HTTPDownloader()
    return downloader.download(
        source_config,
        raw_dir,
        expected_checksum=expected_checksum,
        cached_etag=cached_etag,
        cached_last_modified=cached_last_modified,
    )


def create_source_entry_from_download(
    source_config: SourceConfig,
    result: DownloadResult,
    raw_dir: Path,
) -> SourceEntry:
    """Create a SourceEntry from download result.

    Args:
        source_config: Original source configuration
        result: Download result with metadata
        raw_dir: Raw directory (for computing relative path)

    Returns:
        SourceEntry with full provenance information
    """
    # Compute relative path for manifest
    try:
        local_path = str(result.local_path.relative_to(raw_dir.parent))
    except ValueError:
        local_path = str(result.local_path)

    return create_source_entry(
        source_config,
        local_path=local_path,
        checksum=result.checksum_sha256,
        retrieved_at=result.retrieved_at,
        etag=result.etag,
        last_modified=result.last_modified,
    )
