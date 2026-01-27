"""Git downloader with commit pinning and export to raw/.

This module handles downloading Git-hosted sources declared in sources.yaml:
1. Clones/fetches Git repositories into a cache directory
2. Checks out pinned_revision (commit/tag) for reproducibility
3. Records the resolved commit hash in source_manifest.json
4. Exports relevant files (or tarball archive) into raw/

Example usage:
    downloader = GitDownloader(cache_dir=Path("./git_cache"))
    result = downloader.download(source_config, raw_dir=Path("./raw"))
    # result contains: local_path, resolved_commit, checksum, etc.
"""

import hashlib
import logging
import re
import subprocess
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from finetune.dataset_agent.schemas import SourceEntry, SourceType
from finetune.dataset_agent.sources import (
    SourceConfig,
    create_source_entry,
    get_deterministic_filename,
)

logger = logging.getLogger(__name__)


class GitDownloadError(Exception):
    """Error during Git download."""

    def __init__(
        self,
        message: str,
        source_id: str | None = None,
        url: str | None = None,
        command: str | None = None,
    ) -> None:
        self.source_id = source_id
        self.url = url
        self.command = command
        context = []
        if source_id:
            context.append(f"source_id={source_id}")
        if url:
            context.append(f"url={url}")
        if command:
            context.append(f"command={command}")
        full_message = f"{message} [{', '.join(context)}]" if context else message
        super().__init__(full_message)


@dataclass
class GitDownloadResult:
    """Result of a Git download operation.

    Contains all metadata needed to populate a SourceEntry.
    """

    local_path: Path  # Path where archive was saved in raw/
    checksum_sha256: str  # SHA256 of archive
    size_bytes: int  # Archive file size
    retrieved_at: str  # ISO 8601 timestamp
    resolved_commit: str  # Full commit hash that was checked out
    pinned_revision: str | None = None  # Original pinned revision (tag/branch/commit)
    from_cache: bool = False  # Whether repo was already in cache


def _sanitize_repo_name(url: str) -> str:
    """Convert Git URL to a safe directory name for caching.

    Args:
        url: Git repository URL

    Returns:
        Safe directory name derived from URL
    """
    # Remove protocol
    name = re.sub(r"^(https?://|git@|ssh://)", "", url)
    # Replace special characters
    name = re.sub(r"[/:@]", "_", name)
    # Remove .git suffix
    name = re.sub(r"\.git$", "", name)
    # Remove any remaining unsafe characters
    name = re.sub(r"[^\w_.-]", "_", name)
    return name


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_git_command(
    args: list[str],
    cwd: Path | None = None,
    source_id: str | None = None,
    url: str | None = None,
) -> str:
    """Run a git command and return stdout.

    Args:
        args: Git command arguments (without 'git' prefix)
        cwd: Working directory for command
        source_id: Source ID for error context
        url: URL for error context

    Returns:
        Command stdout as string

    Raises:
        GitDownloadError: If command fails
    """
    cmd = ["git", *args]
    cmd_str = " ".join(cmd)

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=False,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            raise GitDownloadError(
                f"Git command failed: {error_msg}",
                source_id=source_id,
                url=url,
                command=cmd_str,
            )

        return result.stdout.strip()

    except subprocess.TimeoutExpired as e:
        raise GitDownloadError(
            "Git command timed out after 300s",
            source_id=source_id,
            url=url,
            command=cmd_str,
        ) from e
    except FileNotFoundError as e:
        raise GitDownloadError(
            "Git executable not found. Please install git.",
            source_id=source_id,
            url=url,
            command=cmd_str,
        ) from e


class GitDownloader:
    """Downloader for Git-hosted sources with commit pinning and export.

    Features:
    - Clone/fetch repositories into persistent cache directory
    - Checkout specific commits or tags for reproducibility
    - Export files or archives into raw/ directory
    - Record resolved commit hashes in manifest
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the Git downloader.

        Args:
            cache_dir: Directory to store cloned repositories.
                      If None, uses a temp directory within each download call.
        """
        self.cache_dir = cache_dir

    def _get_repo_cache_dir(self, source_config: SourceConfig) -> Path:
        """Get the cache directory for a repository.

        Args:
            source_config: Source configuration

        Returns:
            Path to repository cache directory
        """
        if self.cache_dir is None:
            raise GitDownloadError(
                "No cache directory configured",
                source_id=source_config.id,
                url=source_config.url,
            )

        repo_name = _sanitize_repo_name(source_config.url)
        return self.cache_dir / repo_name

    def _clone_or_fetch(self, source_config: SourceConfig) -> tuple[Path, bool]:
        """Clone repository or fetch updates if already cached.

        Args:
            source_config: Source configuration

        Returns:
            Tuple of (repo_path, from_cache)

        Raises:
            GitDownloadError: If clone/fetch fails
        """
        repo_dir = self._get_repo_cache_dir(source_config)
        from_cache = False

        if repo_dir.exists() and (repo_dir / ".git").exists():
            # Repository exists, fetch updates
            logger.info(f"Fetching updates for {source_config.id} in {repo_dir}")
            run_git_command(
                ["fetch", "--all", "--tags"],
                cwd=repo_dir,
                source_id=source_config.id,
                url=source_config.url,
            )
            from_cache = True
        else:
            # Clone fresh
            logger.info(f"Cloning {source_config.id} from {source_config.url}")
            repo_dir.parent.mkdir(parents=True, exist_ok=True)

            # Clone with --no-checkout to avoid checking out default branch
            run_git_command(
                ["clone", "--no-checkout", source_config.url, str(repo_dir)],
                source_id=source_config.id,
                url=source_config.url,
            )

        return repo_dir, from_cache

    def _checkout_revision(
        self,
        repo_dir: Path,
        pinned_revision: str,
        source_config: SourceConfig,
    ) -> str:
        """Checkout a specific revision and return resolved commit hash.

        Args:
            repo_dir: Path to repository
            pinned_revision: Tag, branch, or commit to checkout
            source_config: Source configuration for error context

        Returns:
            Resolved full commit hash

        Raises:
            GitDownloadError: If checkout fails
        """
        logger.info(f"Checking out {pinned_revision} for {source_config.id}")

        # Checkout the revision
        run_git_command(
            ["checkout", pinned_revision],
            cwd=repo_dir,
            source_id=source_config.id,
            url=source_config.url,
        )

        # Get the resolved commit hash
        resolved_commit = run_git_command(
            ["rev-parse", "HEAD"],
            cwd=repo_dir,
            source_id=source_config.id,
            url=source_config.url,
        )

        logger.info(f"Resolved {pinned_revision} to {resolved_commit[:12]} for {source_config.id}")
        return resolved_commit

    def _export_to_archive(
        self,
        repo_dir: Path,
        raw_dir: Path,
        source_config: SourceConfig,
    ) -> Path:
        """Export repository to a tar.gz archive in raw/.

        Args:
            repo_dir: Path to checked-out repository
            raw_dir: Destination raw/ directory
            source_config: Source configuration

        Returns:
            Path to created archive

        Raises:
            GitDownloadError: If export fails
        """
        filename = get_deterministic_filename(source_config)
        archive_path = raw_dir / filename

        raw_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting {source_config.id} to {archive_path}")

        # If specific files are configured, only archive those
        if source_config.files:
            # Create archive with only specified files
            with tarfile.open(archive_path, "w:gz") as tar:
                for file_pattern in source_config.files:
                    # Handle glob patterns
                    if "*" in file_pattern:
                        matched_files = list(repo_dir.glob(file_pattern))
                    else:
                        file_path = repo_dir / file_pattern
                        matched_files = [file_path] if file_path.exists() else []

                    for file_path in matched_files:
                        if file_path.exists():
                            # Add with relative path
                            arcname = str(file_path.relative_to(repo_dir))
                            tar.add(file_path, arcname=arcname)
                        else:
                            logger.warning(f"File not found: {file_path}")
        else:
            # Archive entire repository (excluding .git)
            with tarfile.open(archive_path, "w:gz") as tar:
                for item in repo_dir.iterdir():
                    if item.name != ".git":
                        tar.add(item, arcname=item.name)

        return archive_path

    def download(
        self,
        source_config: SourceConfig,
        raw_dir: Path,
        expected_checksum: str | None = None,
        expected_commit: str | None = None,
    ) -> GitDownloadResult:
        """Download a Git source and export to raw/.

        Args:
            source_config: Source configuration from sources.yaml
            raw_dir: Directory to store exported archive
            expected_checksum: If provided and archive exists with matching checksum, skip
            expected_commit: If provided and matches pinned revision resolve, skip download

        Returns:
            GitDownloadResult with download metadata

        Raises:
            GitDownloadError: On download failure
        """
        if source_config.type != SourceType.GIT:
            raise GitDownloadError(
                f"Cannot download non-Git source type: {source_config.type.value}",
                source_id=source_config.id,
            )

        if not source_config.pinned_revision:
            raise GitDownloadError(
                "Git sources require pinned_revision for reproducibility",
                source_id=source_config.id,
                url=source_config.url,
            )

        # Determine expected output path
        filename = get_deterministic_filename(source_config)
        archive_path = raw_dir / filename

        # Check if we can use cached archive based on checksum
        if expected_checksum and archive_path.exists():
            actual_checksum = compute_file_sha256(archive_path)
            if actual_checksum == expected_checksum:
                logger.info(
                    f"Cache hit for {source_config.id}: checksum matches {expected_checksum[:16]}..."
                )
                return GitDownloadResult(
                    local_path=archive_path,
                    checksum_sha256=expected_checksum,
                    size_bytes=archive_path.stat().st_size,
                    retrieved_at=datetime.now(UTC).isoformat(),
                    resolved_commit=expected_commit or "",
                    pinned_revision=source_config.pinned_revision,
                    from_cache=True,
                )

        # Clone or fetch the repository
        repo_dir, from_cache = self._clone_or_fetch(source_config)

        # Checkout the pinned revision
        resolved_commit = self._checkout_revision(
            repo_dir,
            source_config.pinned_revision,
            source_config,
        )

        # Export to archive
        archive_path = self._export_to_archive(repo_dir, raw_dir, source_config)

        # Compute checksum
        checksum = compute_file_sha256(archive_path)
        size_bytes = archive_path.stat().st_size

        logger.info(
            f"Downloaded {source_config.id}: {size_bytes} bytes, "
            f"commit={resolved_commit[:12]}, checksum={checksum[:16]}..."
        )

        return GitDownloadResult(
            local_path=archive_path,
            checksum_sha256=checksum,
            size_bytes=size_bytes,
            retrieved_at=datetime.now(UTC).isoformat(),
            resolved_commit=resolved_commit,
            pinned_revision=source_config.pinned_revision,
            from_cache=from_cache,
        )

    def download_all(
        self,
        sources_config: list[SourceConfig],
        raw_dir: Path,
        previous_manifest: dict[str, SourceEntry] | None = None,
    ) -> list[GitDownloadResult]:
        """Download all Git sources from configuration.

        Args:
            sources_config: List of source configurations
            raw_dir: Directory to store exported archives
            previous_manifest: Optional dict mapping source_id to previous SourceEntry
                              for checksum validation and commit tracking

        Returns:
            List of GitDownloadResult for each Git source
        """
        results = []
        previous_manifest = previous_manifest or {}

        for source in sources_config:
            if source.type != SourceType.GIT:
                logger.debug(f"Skipping non-Git source: {source.id}")
                continue

            # Get cached metadata if available
            previous_entry = previous_manifest.get(source.id)
            expected_checksum = previous_entry.checksum_sha256 if previous_entry else None
            expected_commit = previous_entry.resolved_commit if previous_entry else None

            try:
                result = self.download(
                    source,
                    raw_dir,
                    expected_checksum=expected_checksum,
                    expected_commit=expected_commit,
                )
                results.append(result)
            except GitDownloadError as e:
                logger.error(f"Failed to download {source.id}: {e}")
                raise

        return results


def download_git_source(
    source_config: SourceConfig,
    raw_dir: Path,
    cache_dir: Path | None = None,
    expected_checksum: str | None = None,
    expected_commit: str | None = None,
) -> GitDownloadResult:
    """Convenience function to download a single Git source.

    Args:
        source_config: Source configuration from sources.yaml
        raw_dir: Directory to store exported archive
        cache_dir: Directory to cache cloned repositories
        expected_checksum: If provided and archive exists with matching checksum, skip
        expected_commit: If provided and matches, include in result

    Returns:
        GitDownloadResult with download metadata
    """
    downloader = GitDownloader(cache_dir=cache_dir)
    return downloader.download(
        source_config,
        raw_dir,
        expected_checksum=expected_checksum,
        expected_commit=expected_commit,
    )


def create_source_entry_from_git_download(
    source_config: SourceConfig,
    result: GitDownloadResult,
    raw_dir: Path,
) -> SourceEntry:
    """Create a SourceEntry from Git download result.

    Args:
        source_config: Original source configuration
        result: Git download result with metadata
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
        resolved_commit=result.resolved_commit,
    )
