"""Local ADS query syntax validation for dataset generation pipeline.

This module provides Tier 1 local validation for ADS queries, catching malformed
queries before any backend calls. It leverages the existing lint_query and
validate_field_constraints functions from the scix domain.

Features:
- Validate rendered queries against field/operator whitelist
- Check for unbalanced parentheses, quotes, brackets
- Validate field constraint values (doctype, property, bibgroup, database)
- Quarantine invalid pairs with detailed error messages

Output:
- pairs/pairs.jsonl: Valid pairs
- pairs/quarantine.jsonl: Invalid pairs with error details
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from finetune.dataset_agent.schemas import Pair, QuarantinedPair
from finetune.dataset_agent.writers import JSONLReader, JSONLWriter
from finetune.domains.scix.validate import (
    ConstraintValidationResult,
    ValidationResult,
    lint_query,
    validate_field_constraints,
)


@dataclass
class LocalValidatorConfig:
    """Configuration for local query validation.

    Attributes:
        validate_syntax: Run basic syntax validation (quotes, parens, brackets)
        validate_fields: Check that field prefixes are valid ADS fields
        validate_constraints: Check field values against allowed enumerations
        stop_on_first_error: Stop validation on first error (faster but less info)
    """

    validate_syntax: bool = True
    validate_fields: bool = True
    validate_constraints: bool = True
    stop_on_first_error: bool = False


@dataclass
class LocalValidationResult:
    """Result of local validation for a single pair.

    Attributes:
        valid: Whether the query passed all validations
        syntax_result: Result from lint_query
        constraint_result: Result from validate_field_constraints
        all_errors: Combined list of all error messages
        error_type: Category of the first error (for quarantine)
    """

    valid: bool
    syntax_result: ValidationResult | None = None
    constraint_result: ConstraintValidationResult | None = None

    @property
    def all_errors(self) -> list[str]:
        """Get all error messages from all validations."""
        errors = []
        if self.syntax_result:
            errors.extend(self.syntax_result.errors)
        if self.constraint_result:
            errors.extend(self.constraint_result.error_messages)
        return errors

    @property
    def error_type(self) -> str:
        """Categorize the error type for quarantine."""
        if self.syntax_result and self.syntax_result.errors:
            first_error = self.syntax_result.errors[0]
            if "Unbalanced quotes" in first_error:
                return "unbalanced_quotes"
            elif "Unbalanced parentheses" in first_error:
                return "unbalanced_parentheses"
            elif "Unbalanced brackets" in first_error:
                return "unbalanced_brackets"
            elif "Unknown field" in first_error:
                return "unknown_field"
            elif "boolean operator" in first_error.lower():
                return "invalid_boolean"
            elif "cannot start with" in first_error.lower():
                return "invalid_boolean"
            elif "cannot end with" in first_error.lower():
                return "invalid_boolean"
            elif "Empty query" in first_error:
                return "empty_query"
            else:
                return "syntax_error"
        elif self.constraint_result and self.constraint_result.errors:
            return "invalid_field_value"
        return "unknown"


@dataclass
class LocalValidatorStats:
    """Statistics from local validation.

    Attributes:
        pairs_processed: Total number of pairs processed
        pairs_valid: Number of pairs that passed validation
        pairs_invalid: Number of pairs that failed validation
        errors_by_type: Count of errors by error type
    """

    pairs_processed: int = 0
    pairs_valid: int = 0
    pairs_invalid: int = 0
    errors_by_type: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.errors_by_type is None:
            self.errors_by_type = {}


def validate_query_local(
    query: str,
    config: LocalValidatorConfig | None = None,
) -> LocalValidationResult:
    """Perform local validation on an ADS query.

    Args:
        query: The ADS query string to validate
        config: Validation configuration

    Returns:
        LocalValidationResult with validation details
    """
    config = config or LocalValidatorConfig()

    syntax_result: ValidationResult | None = None
    constraint_result: ConstraintValidationResult | None = None
    valid = True

    # Syntax validation (quotes, parens, field prefixes, boolean operators)
    if config.validate_syntax or config.validate_fields:
        syntax_result = lint_query(query)
        if not syntax_result.valid:
            valid = False
            if config.stop_on_first_error:
                return LocalValidationResult(
                    valid=False,
                    syntax_result=syntax_result,
                )

    # Field constraint validation (doctype, property, bibgroup, database values)
    if config.validate_constraints:
        constraint_result = validate_field_constraints(query)
        if not constraint_result.valid:
            valid = False

    return LocalValidationResult(
        valid=valid,
        syntax_result=syntax_result,
        constraint_result=constraint_result,
    )


class LocalValidator:
    """Local validator for ADS query pairs.

    Validates queries using local rules and separates valid pairs from
    invalid pairs (quarantine).
    """

    def __init__(self, config: LocalValidatorConfig | None = None) -> None:
        """Initialize the local validator.

        Args:
            config: Validation configuration
        """
        self.config = config or LocalValidatorConfig()
        self._stats = LocalValidatorStats()

    def validate_pair(self, pair: Pair) -> tuple[Pair | None, QuarantinedPair | None]:
        """Validate a single pair.

        Args:
            pair: The pair to validate

        Returns:
            Tuple of (valid_pair, quarantined_pair) - one will be None
        """
        self._stats.pairs_processed += 1

        result = validate_query_local(pair.ads_query, self.config)

        if result.valid:
            # Update the pair's validation tier
            validated_pair = Pair(
                pair_id=pair.pair_id,
                user_text=pair.user_text,
                ads_query=pair.ads_query,
                template_id=pair.template_id,
                filled_slots=pair.filled_slots,
                validation_tier=1,  # Tier 1 = local validation passed
                validation_errors=[],
            )
            self._stats.pairs_valid += 1
            return validated_pair, None
        else:
            # Create quarantined pair with error details
            error_type = result.error_type
            quarantined = QuarantinedPair(
                pair_id=pair.pair_id,
                user_text=pair.user_text,
                ads_query=pair.ads_query,
                template_id=pair.template_id,
                filled_slots=pair.filled_slots,
                error_type=error_type,
                error_details="; ".join(result.all_errors),
                failed_at_tier=1,
            )
            self._stats.pairs_invalid += 1
            # Track error types
            if self._stats.errors_by_type is not None:
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
    ) -> tuple[str, int, str, int, LocalValidatorStats]:
        """Validate pairs and write to separate files.

        Args:
            pairs: List of pairs to validate
            valid_output_path: Path for valid pairs (pairs.jsonl)
            quarantine_output_path: Path for quarantined pairs (quarantine.jsonl)

        Returns:
            Tuple of (valid_checksum, valid_count, quarantine_checksum, quarantine_count, stats)
        """
        # Reset stats
        self._stats = LocalValidatorStats()

        valid_pairs, quarantined_pairs = self.validate_pairs(pairs)

        # Write valid pairs
        valid_checksum, valid_count = JSONLWriter(valid_output_path).write_all(
            valid_pairs
        )

        # Write quarantined pairs
        quarantine_checksum, quarantine_count = JSONLWriter(
            quarantine_output_path
        ).write_all(quarantined_pairs)

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
    ) -> tuple[str, int, str, int, LocalValidatorStats]:
        """Load pairs from file, validate, and write to separate files.

        Args:
            input_path: Path to input pairs.jsonl
            valid_output_path: Path for valid pairs
            quarantine_output_path: Path for quarantined pairs

        Returns:
            Tuple of (valid_checksum, valid_count, quarantine_checksum, quarantine_count, stats)
        """
        # Reset stats
        self._stats = LocalValidatorStats()

        reader = JSONLReader(input_path)
        pairs = [Pair.from_dict(d) for d in reader]

        return self.validate_to_files(
            pairs, valid_output_path, quarantine_output_path
        )

    @property
    def stats(self) -> LocalValidatorStats:
        """Get validation statistics."""
        return self._stats


def validate_pairs_local(
    pairs: list[Pair],
    valid_output_path: Path,
    quarantine_output_path: Path,
    config: LocalValidatorConfig | None = None,
) -> tuple[str, int, str, int, LocalValidatorStats]:
    """Convenience function to validate pairs and write to files.

    Args:
        pairs: List of pairs to validate
        valid_output_path: Path for valid pairs
        quarantine_output_path: Path for quarantined pairs
        config: Validation configuration

    Returns:
        Tuple of (valid_checksum, valid_count, quarantine_checksum, quarantine_count, stats)
    """
    validator = LocalValidator(config=config)
    return validator.validate_to_files(pairs, valid_output_path, quarantine_output_path)


def validate_pairs_from_file(
    input_path: Path,
    valid_output_path: Path,
    quarantine_output_path: Path,
    config: LocalValidatorConfig | None = None,
) -> tuple[str, int, str, int, LocalValidatorStats]:
    """Convenience function to validate pairs from file.

    Args:
        input_path: Path to input pairs.jsonl
        valid_output_path: Path for valid pairs
        quarantine_output_path: Path for quarantined pairs
        config: Validation configuration

    Returns:
        Tuple of (valid_checksum, valid_count, quarantine_checksum, quarantine_count, stats)
    """
    validator = LocalValidator(config=config)
    return validator.validate_from_file(
        input_path, valid_output_path, quarantine_output_path
    )
