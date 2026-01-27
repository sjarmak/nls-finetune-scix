"""Backend ADS query validation for dataset generation pipeline.

This module provides Tier 2 (syntax) and Tier 3 (results) validation against the
ADS API, with caching and rate limiting. It wraps the existing validate_query
function from the scix domain and adds:

- Offline/online mode switching (offline = no-op validation)
- Request caching keyed by canonical query + validator version
- Rate limiting for outgoing validation calls
- Batch validation with progress tracking

Output:
- manifests/validation_cache.jsonl: Cache of validation results
- pairs/pairs.jsonl: Updated with validation_tier 2 or 3 for validated pairs
- pairs/quarantine.jsonl: Pairs that failed backend validation
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from finetune.dataset_agent.pair_renderer import canonicalize_query
from finetune.dataset_agent.schemas import Pair, QuarantinedPair
from finetune.dataset_agent.writers import JSONLReader, JSONLWriter
from finetune.domains.scix.validate import validate_query

# Validator version for cache keying
# Bump this when validation logic changes to invalidate old cache entries
VALIDATOR_VERSION = "1.0.0"


class ValidationMode(Enum):
    """Validation mode for backend validator."""

    OFFLINE = "offline"  # No-op validation, all pairs pass
    ONLINE = "online"  # Full API validation


class ValidationTier(Enum):
    """Validation tier achieved by a pair."""

    NONE = 0  # No validation
    LOCAL = 1  # Passed local syntax validation
    BACKEND_SYNTAX = 2  # Passed ADS API syntax validation
    BACKEND_RESULTS = 3  # Passed ADS API validation and returned results


@dataclass
class ValidationCacheEntry:
    """Cache entry for a validation result.

    Attributes:
        query_hash: MD5 hash of canonical query + validator version
        canonical_query: The canonicalized query string
        validator_version: Version of validator that produced this result
        valid: Whether the query is valid
        tier: Validation tier achieved (2 or 3)
        num_results: Number of results returned (None if not checked)
        error_message: Error message if invalid
        cached_at: ISO 8601 timestamp when cached
    """

    query_hash: str
    canonical_query: str
    validator_version: str
    valid: bool
    tier: int
    num_results: int | None = None
    error_message: str | None = None
    cached_at: str | None = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        result = {
            "query_hash": self.query_hash,
            "canonical_query": self.canonical_query,
            "validator_version": self.validator_version,
            "valid": self.valid,
            "tier": self.tier,
        }
        if self.num_results is not None:
            result["num_results"] = self.num_results
        if self.error_message is not None:
            result["error_message"] = self.error_message
        if self.cached_at is not None:
            result["cached_at"] = self.cached_at
        return result

    @classmethod
    def from_dict(cls, data: dict) -> ValidationCacheEntry:
        """Create from dict."""
        return cls(
            query_hash=data["query_hash"],
            canonical_query=data["canonical_query"],
            validator_version=data["validator_version"],
            valid=data["valid"],
            tier=data["tier"],
            num_results=data.get("num_results"),
            error_message=data.get("error_message"),
            cached_at=data.get("cached_at"),
        )


@dataclass
class BackendValidatorConfig:
    """Configuration for backend query validation.

    Attributes:
        mode: Validation mode (offline or online)
        api_key: ADS API key (defaults to ADS_API_KEY env var)
        api_url: ADS API endpoint URL
        rate_limit_rps: Maximum requests per second (default 5)
        rate_limit_burst: Maximum burst requests (default 10)
        require_results: Whether to require at least 1 result (tier 3 vs tier 2)
        cache_path: Path to validation cache file
    """

    mode: ValidationMode = ValidationMode.OFFLINE
    api_key: str | None = None
    api_url: str = "https://api.adsabs.harvard.edu/v1/search/query"
    rate_limit_rps: float = 5.0
    rate_limit_burst: int = 10
    require_results: bool = False
    cache_path: Path | None = None


@dataclass
class BackendValidatorStats:
    """Statistics from backend validation.

    Attributes:
        pairs_processed: Total number of pairs processed
        pairs_valid: Number of pairs that passed validation
        pairs_invalid: Number of pairs that failed validation
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        api_calls: Number of API calls made
        rate_limit_waits: Number of times rate limiter caused a wait
        errors_by_type: Count of errors by error type
    """

    pairs_processed: int = 0
    pairs_valid: int = 0
    pairs_invalid: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0
    rate_limit_waits: int = 0
    errors_by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class BackendValidationResult:
    """Result of backend validation for a single pair.

    Attributes:
        valid: Whether the query passed validation
        tier: Validation tier achieved (0, 2, or 3)
        num_results: Number of results if validated
        error_message: Error message if invalid
        from_cache: Whether result was from cache
    """

    valid: bool
    tier: int
    num_results: int | None = None
    error_message: str | None = None
    from_cache: bool = False


def compute_query_hash(query: str, version: str = VALIDATOR_VERSION) -> str:
    """Compute cache key hash for a query.

    Args:
        query: The ADS query to hash
        version: Validator version to include in hash

    Returns:
        MD5 hash string
    """
    # Canonicalize query first for consistent hashing
    canonical = canonicalize_query(query)
    key = f"{canonical}::{version}"
    return hashlib.md5(key.encode()).hexdigest()


class RateLimiter:
    """Token bucket rate limiter.

    Provides rate limiting with burst capacity using the token bucket algorithm.
    """

    def __init__(self, rps: float = 5.0, burst: int = 10) -> None:
        """Initialize rate limiter.

        Args:
            rps: Requests per second rate
            burst: Maximum burst capacity
        """
        self.rps = rps
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()

    def acquire(self) -> float:
        """Acquire a token, waiting if necessary.

        Returns:
            Time waited in seconds (0 if no wait needed)
        """
        now = time.monotonic()

        # Add tokens based on elapsed time
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rps)
        self.last_update = now

        # If we have tokens, consume one and return immediately
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0

        # Otherwise, wait for a token to become available
        wait_time = (1.0 - self.tokens) / self.rps
        time.sleep(wait_time)
        self.tokens = 0.0
        self.last_update = time.monotonic()
        return wait_time


class ValidationCache:
    """Cache for validation results.

    Provides in-memory caching with optional persistence to JSONL file.
    """

    def __init__(self, cache_path: Path | None = None) -> None:
        """Initialize validation cache.

        Args:
            cache_path: Optional path to persist cache
        """
        self.cache_path = cache_path
        self._cache: dict[str, ValidationCacheEntry] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from file if it exists."""
        if self.cache_path and self.cache_path.exists():
            reader = JSONLReader(self.cache_path)
            for entry_dict in reader:
                entry = ValidationCacheEntry.from_dict(entry_dict)
                # Only load entries from current validator version
                if entry.validator_version == VALIDATOR_VERSION:
                    self._cache[entry.query_hash] = entry

    def get(self, query: str) -> ValidationCacheEntry | None:
        """Get cached validation result for a query.

        Args:
            query: The query to look up

        Returns:
            Cached entry if found, None otherwise
        """
        query_hash = compute_query_hash(query)
        return self._cache.get(query_hash)

    def put(self, entry: ValidationCacheEntry) -> None:
        """Store a validation result in the cache.

        Args:
            entry: The cache entry to store
        """
        self._cache[entry.query_hash] = entry

    def save(self) -> tuple[str, int] | None:
        """Save cache to file.

        Returns:
            Tuple of (checksum, count) if cache_path set, None otherwise
        """
        if not self.cache_path:
            return None

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        with JSONLWriter(self.cache_path) as writer:
            for entry in sorted(self._cache.values(), key=lambda e: e.query_hash):
                writer.write_line(entry)

        return writer.checksum, writer.line_count


class BackendValidator:
    """Backend validator for ADS query pairs.

    Validates queries against the ADS API with caching and rate limiting.
    """

    def __init__(self, config: BackendValidatorConfig | None = None) -> None:
        """Initialize the backend validator.

        Args:
            config: Validation configuration
        """
        self.config = config or BackendValidatorConfig()
        self._stats = BackendValidatorStats()
        self._cache = ValidationCache(self.config.cache_path)
        self._rate_limiter = RateLimiter(
            rps=self.config.rate_limit_rps,
            burst=self.config.rate_limit_burst,
        )

    def _get_api_key(self) -> str | None:
        """Get API key from config or environment."""
        return self.config.api_key or os.environ.get("ADS_API_KEY")

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error type from error message."""
        lower = error_message.lower()
        if "parse error" in lower:
            return "parse_error"
        if "rate limit" in lower:
            return "rate_limit"
        if "timeout" in lower:
            return "timeout"
        if "api key" in lower or "unauthorized" in lower or "401" in lower:
            return "auth_error"
        if "request error" in lower or "connection" in lower:
            return "network_error"
        return "api_error"

    def validate_query(self, query: str) -> BackendValidationResult:
        """Validate a single query against the ADS API.

        Args:
            query: The ADS query to validate

        Returns:
            BackendValidationResult with validation details
        """
        # In offline mode, all queries pass as tier 2
        if self.config.mode == ValidationMode.OFFLINE:
            return BackendValidationResult(
                valid=True,
                tier=ValidationTier.BACKEND_SYNTAX.value,
                from_cache=False,
            )

        # Check cache first
        cached = self._cache.get(query)
        if cached is not None:
            self._stats.cache_hits += 1
            return BackendValidationResult(
                valid=cached.valid,
                tier=cached.tier,
                num_results=cached.num_results,
                error_message=cached.error_message,
                from_cache=True,
            )

        self._stats.cache_misses += 1

        # Rate limit before API call
        wait_time = self._rate_limiter.acquire()
        if wait_time > 0:
            self._stats.rate_limit_waits += 1

        # Call ADS API
        self._stats.api_calls += 1
        api_key = self._get_api_key()

        result = validate_query(
            query=query,
            api_key=api_key,
            api_url=self.config.api_url,
        )

        # Parse result
        if result.valid:
            # Check for results warning
            num_results = None
            has_results = True
            for warning in result.warnings:
                if "returned 0 results" in warning:
                    num_results = 0
                    has_results = False
                    break

            if has_results and num_results is None:
                # Query returned results (we don't know how many from this response)
                num_results = 1  # Placeholder indicating "some results"

            # Determine tier based on results requirement
            if self.config.require_results and not has_results:
                tier = ValidationTier.BACKEND_SYNTAX.value
            else:
                tier = (
                    ValidationTier.BACKEND_RESULTS.value
                    if has_results
                    else ValidationTier.BACKEND_SYNTAX.value
                )

            valid = True
            error_message = None
        else:
            num_results = None
            tier = ValidationTier.LOCAL.value  # Tier 1 at best
            valid = False
            error_message = "; ".join(result.errors)

        # Cache result
        from datetime import UTC, datetime

        cache_entry = ValidationCacheEntry(
            query_hash=compute_query_hash(query),
            canonical_query=canonicalize_query(query),
            validator_version=VALIDATOR_VERSION,
            valid=valid,
            tier=tier,
            num_results=num_results,
            error_message=error_message,
            cached_at=datetime.now(UTC).isoformat(),
        )
        self._cache.put(cache_entry)

        return BackendValidationResult(
            valid=valid,
            tier=tier,
            num_results=num_results,
            error_message=error_message,
            from_cache=False,
        )

    def validate_pair(self, pair: Pair) -> tuple[Pair | None, QuarantinedPair | None]:
        """Validate a single pair against the backend.

        Args:
            pair: The pair to validate (must have passed local validation)

        Returns:
            Tuple of (valid_pair, quarantined_pair) - one will be None
        """
        self._stats.pairs_processed += 1

        result = self.validate_query(pair.ads_query)

        if result.valid:
            # Update the pair's validation tier
            validated_pair = Pair(
                pair_id=pair.pair_id,
                user_text=pair.user_text,
                ads_query=pair.ads_query,
                template_id=pair.template_id,
                filled_slots=pair.filled_slots,
                validation_tier=result.tier,
                validation_errors=[],
            )
            self._stats.pairs_valid += 1
            return validated_pair, None
        else:
            # Create quarantined pair with error details
            error_type = self._categorize_error(result.error_message or "unknown")
            quarantined = QuarantinedPair(
                pair_id=pair.pair_id,
                user_text=pair.user_text,
                ads_query=pair.ads_query,
                template_id=pair.template_id,
                filled_slots=pair.filled_slots,
                error_type=error_type,
                error_details=result.error_message or "Unknown error",
                failed_at_tier=2,  # Failed at backend validation
            )
            self._stats.pairs_invalid += 1
            self._stats.errors_by_type[error_type] = (
                self._stats.errors_by_type.get(error_type, 0) + 1
            )
            return None, quarantined

    def validate_pairs(
        self, pairs: list[Pair]
    ) -> tuple[list[Pair], list[QuarantinedPair]]:
        """Validate multiple pairs.

        Args:
            pairs: List of pairs to validate

        Returns:
            Tuple of (valid_pairs, quarantined_pairs)
        """
        valid_pairs: list[Pair] = []
        quarantined_pairs: list[QuarantinedPair] = []

        for pair in pairs:
            valid, quarantined = self.validate_pair(pair)
            if valid:
                valid_pairs.append(valid)
            if quarantined:
                quarantined_pairs.append(quarantined)

        return valid_pairs, quarantined_pairs

    def validate_to_files(
        self,
        pairs: list[Pair],
        valid_output_path: Path,
        quarantine_output_path: Path,
    ) -> tuple[str, int, str, int, BackendValidatorStats]:
        """Validate pairs and write to separate files.

        Args:
            pairs: List of pairs to validate
            valid_output_path: Path for valid pairs (pairs.jsonl)
            quarantine_output_path: Path for quarantined pairs (quarantine.jsonl)

        Returns:
            Tuple of (valid_checksum, valid_count, quarantine_checksum, quarantine_count, stats)
        """
        # Reset stats
        self._stats = BackendValidatorStats()

        valid_pairs, quarantined_pairs = self.validate_pairs(pairs)

        # Write valid pairs
        valid_checksum, valid_count = JSONLWriter(valid_output_path).write_all(
            valid_pairs
        )

        # Write quarantined pairs
        quarantine_checksum, quarantine_count = JSONLWriter(
            quarantine_output_path
        ).write_all(quarantined_pairs)

        # Save cache
        self._cache.save()

        return (
            valid_checksum,
            valid_count,
            quarantine_checksum,
            quarantine_count,
            self._stats,
        )

    def validate_from_file(
        self,
        input_path: Path,
        valid_output_path: Path,
        quarantine_output_path: Path,
    ) -> tuple[str, int, str, int, BackendValidatorStats]:
        """Load pairs from file, validate, and write to separate files.

        Args:
            input_path: Path to input pairs.jsonl
            valid_output_path: Path for valid pairs
            quarantine_output_path: Path for quarantined pairs

        Returns:
            Tuple of (valid_checksum, valid_count, quarantine_checksum, quarantine_count, stats)
        """
        # Reset stats
        self._stats = BackendValidatorStats()

        reader = JSONLReader(input_path)
        pairs = [Pair.from_dict(d) for d in reader]

        return self.validate_to_files(pairs, valid_output_path, quarantine_output_path)

    @property
    def stats(self) -> BackendValidatorStats:
        """Get validation statistics."""
        return self._stats

    @property
    def cache(self) -> ValidationCache:
        """Get the validation cache."""
        return self._cache


def validate_pairs_backend(
    pairs: list[Pair],
    valid_output_path: Path,
    quarantine_output_path: Path,
    config: BackendValidatorConfig | None = None,
) -> tuple[str, int, str, int, BackendValidatorStats]:
    """Convenience function to validate pairs and write to files.

    Args:
        pairs: List of pairs to validate
        valid_output_path: Path for valid pairs
        quarantine_output_path: Path for quarantined pairs
        config: Validation configuration

    Returns:
        Tuple of (valid_checksum, valid_count, quarantine_checksum, quarantine_count, stats)
    """
    validator = BackendValidator(config=config)
    return validator.validate_to_files(pairs, valid_output_path, quarantine_output_path)


def validate_pairs_from_file_backend(
    input_path: Path,
    valid_output_path: Path,
    quarantine_output_path: Path,
    config: BackendValidatorConfig | None = None,
) -> tuple[str, int, str, int, BackendValidatorStats]:
    """Convenience function to validate pairs from file.

    Args:
        input_path: Path to input pairs.jsonl
        valid_output_path: Path for valid pairs
        quarantine_output_path: Path for quarantined pairs
        config: Validation configuration

    Returns:
        Tuple of (valid_checksum, valid_count, quarantine_checksum, quarantine_count, stats)
    """
    validator = BackendValidator(config=config)
    return validator.validate_from_file(
        input_path, valid_output_path, quarantine_output_path
    )
