"""ROR (Research Organization Registry) institution normalizer.

This module parses ROR data (from ror-data.json format) and produces
normalized EntityEntry records for the entity_catalog_institutions.jsonl output.

ROR source: https://github.com/ror-community/ror-data
Schema v2 format: JSON array of organization objects with fields:
  - id: ROR identifier (e.g., "https://ror.org/04nt9b154")
  - names: Array of name objects with types (ror_display, alias, acronym, label)
  - types: Organization classification (education, funder, etc.)
  - status: active, inactive, withdrawn
  - locations: Geographic data with geonames_details, subdivisions
  - external_ids: Other identifiers (fundref, grid, isni, wikidata)
  - relationships: Related organizations
  - links: Website URLs
  - established: Year established
  - domains: Registered domains
  - admin: Creation/modification metadata
"""

from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finetune.dataset_agent.schemas import EntityEntry

if TYPE_CHECKING:
    from collections.abc import Iterator


class RORParseError(Exception):
    """Error parsing ROR data."""

    def __init__(self, message: str, source_path: str | None = None):
        self.source_path = source_path
        super().__init__(message)


def extract_ror_id(ror_url: str) -> str:
    """Extract the ROR ID from a ROR URL.

    Args:
        ror_url: Full ROR URL (e.g., "https://ror.org/04nt9b154")

    Returns:
        ROR ID (e.g., "ror:04nt9b154")
    """
    # Handle various URL formats
    if "ror.org/" in ror_url:
        # Standard format: https://ror.org/04nt9b154
        ror_id = ror_url.split("ror.org/")[-1].strip("/")
        return f"ror:{ror_id}"
    # Fallback: use the whole URL
    return ror_url


def normalize_name(name: str) -> str:
    """Normalize a name by stripping whitespace.

    Args:
        name: Raw name string

    Returns:
        Normalized name (stripped of leading/trailing whitespace)
    """
    if not name:
        return ""
    return name.strip()


def extract_names_by_type(
    names: list[dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    """Extract display name, aliases, and acronyms from names array.

    ROR schema v2 uses a single 'names' array with types:
    - ror_display: Primary display name
    - alias: Alternative names
    - acronym: Abbreviations
    - label: Translations/localized names

    Args:
        names: Array of name objects with 'value' and 'types' fields

    Returns:
        Tuple of (display_name, aliases, acronyms)
    """
    display_name = ""
    aliases: list[str] = []
    acronyms: list[str] = []

    for name_obj in names:
        if not isinstance(name_obj, dict):
            continue

        value = name_obj.get("value", "")
        if not value:
            continue

        types = name_obj.get("types", [])
        if not isinstance(types, list):
            types = [types] if types else []

        # ror_display is the primary name
        if "ror_display" in types:
            display_name = normalize_name(value)
        # Handle other types
        elif "acronym" in types:
            acronyms.append(normalize_name(value))
        elif "alias" in types or "label" in types:
            aliases.append(normalize_name(value))
        elif not display_name:
            # Fallback: use first name as display if no ror_display
            display_name = normalize_name(value)

    return display_name, aliases, acronyms


def extract_names_v1(
    org: dict[str, Any],
) -> tuple[str, list[str], list[str]]:
    """Extract name, aliases, and acronyms from ROR schema v1 format.

    Schema v1 has separate fields:
    - name: Primary name
    - aliases: List of alternative names
    - acronyms: List of abbreviations
    - labels: Translations with iso639 code and label

    Args:
        org: Organization object in v1 format

    Returns:
        Tuple of (display_name, aliases, acronyms)
    """
    display_name = normalize_name(org.get("name", ""))

    aliases: list[str] = []
    raw_aliases = org.get("aliases") or []
    for alias in raw_aliases:
        if alias:
            aliases.append(normalize_name(alias))

    # Add labels (translations) to aliases
    labels = org.get("labels") or []
    for label_obj in labels:
        if isinstance(label_obj, dict):
            label_value = label_obj.get("label", "")
            if label_value:
                aliases.append(normalize_name(label_value))

    acronyms: list[str] = []
    raw_acronyms = org.get("acronyms") or []
    for acronym in raw_acronyms:
        if acronym:
            acronyms.append(normalize_name(acronym))

    return display_name, aliases, acronyms


def deduplicate_aliases(
    aliases: list[str],
    preferred_name: str,
    case_insensitive: bool = True,
) -> list[str]:
    """Deduplicate aliases, removing duplicates and the preferred name.

    Args:
        aliases: List of alternative names
        preferred_name: The primary name to exclude
        case_insensitive: Whether to compare case-insensitively

    Returns:
        Deduplicated list of aliases (empty strings and whitespace-only removed)
    """
    seen: set[str] = set()
    result: list[str] = []

    # Normalize the preferred name for comparison
    pref_norm = preferred_name.lower() if case_insensitive else preferred_name

    for alias in aliases:
        normalized = normalize_name(alias)
        if not normalized:  # Skip empty/whitespace-only
            continue

        # Create comparison key
        key = normalized.lower() if case_insensitive else normalized

        # Skip if duplicate or matches preferred name
        if key in seen or key == pref_norm:
            continue

        seen.add(key)
        result.append(normalized)

    return result


def extract_country(org: dict[str, Any]) -> str | None:
    """Extract country code from organization.

    Supports both v1 (country.country_code) and v2 (locations[0].geonames_details.country_code)
    formats.

    Args:
        org: Organization object

    Returns:
        ISO 3166-1 alpha-2 country code or None
    """
    # Schema v2: locations array
    locations = org.get("locations")
    if locations and isinstance(locations, list):
        for loc in locations:
            if isinstance(loc, dict):
                geonames = loc.get("geonames_details", {})
                if geonames:
                    country_code = geonames.get("country_code")
                    if country_code:
                        return country_code

    # Schema v1: country object
    country = org.get("country")
    if isinstance(country, dict):
        return country.get("country_code")

    return None


def is_schema_v2(data: list[dict[str, Any]]) -> bool:
    """Detect whether ROR data is in schema v2 format.

    Schema v2 has 'names' array with type objects.
    Schema v1 has 'name' string field.

    Args:
        data: List of organization objects

    Returns:
        True if data appears to be schema v2
    """
    if not data:
        return False

    first_org = data[0]
    # v2 has 'names' array, v1 has 'name' string
    return "names" in first_org and isinstance(first_org.get("names"), list)


def parse_ror_organization(
    org: dict[str, Any],
    source_id: str | None = None,
    is_v2: bool = True,
) -> EntityEntry | None:
    """Parse a single ROR organization into an EntityEntry.

    Args:
        org: Raw organization dictionary from ROR JSON
        source_id: Optional source identifier for provenance
        is_v2: Whether the data is in schema v2 format

    Returns:
        EntityEntry or None if organization is invalid
    """
    # Extract ROR ID
    ror_url = org.get("id")
    if not ror_url:
        return None

    entity_id = extract_ror_id(ror_url)

    # Extract names based on schema version
    if is_v2:
        names = org.get("names", [])
        display_name, aliases, acronyms = extract_names_by_type(names)
    else:
        display_name, aliases, acronyms = extract_names_v1(org)

    if not display_name:
        return None

    # Combine aliases and acronyms, deduplicate
    all_aliases = aliases + acronyms
    deduped_aliases = deduplicate_aliases(all_aliases, display_name)

    # Extract country
    country = extract_country(org)

    # Extract organization types
    org_types = org.get("types", [])
    if not isinstance(org_types, list):
        org_types = []

    # Extract status (active, inactive, withdrawn)
    status = org.get("status", "active")

    # Build metadata
    metadata: dict[str, Any] = {}
    if country:
        metadata["country"] = country
    if org_types:
        metadata["types"] = org_types
    if status and status != "active":
        metadata["status"] = status
    if acronyms:
        metadata["acronyms"] = acronyms

    return EntityEntry(
        id=entity_id,
        label=display_name,
        aliases=deduped_aliases,
        metadata=metadata,
        source_id=source_id,
        domain_tags=["multidisciplinary"],
        source_vocabulary="ror",
    )


def load_ror_json(path: Path) -> list[dict[str, Any]]:
    """Load ROR organizations from a JSON file.

    Supports plain JSON files, JSON files inside tar.gz archives, and zip archives.

    Args:
        path: Path to ror-data.json, tar.gz archive, or zip archive containing it

    Returns:
        List of organization dictionaries

    Raises:
        RORParseError: If the file cannot be parsed
    """
    try:
        if path.suffix == ".zip" or path.name.endswith(".zip"):
            # Extract from zip archive (Zenodo format)
            return _load_from_zipfile(path)
        elif path.suffix == ".gz" or path.name.endswith(".tar.gz"):
            # Extract from tar.gz archive
            return _load_from_tarball(path)
        else:
            # Plain JSON file
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise RORParseError(
                    f"Expected JSON array at root, got {type(data).__name__}",
                    source_path=str(path),
                )
            return data

    except json.JSONDecodeError as e:
        raise RORParseError(f"Invalid JSON: {e}", source_path=str(path)) from e
    except FileNotFoundError as e:
        raise RORParseError(f"File not found: {path}", source_path=str(path)) from e


def _load_from_tarball(path: Path) -> list[dict[str, Any]]:
    """Load ROR JSON from a tar.gz archive.

    Looks for ror-data.json, ror-data_schema_v2.json, or similar inside the archive.

    Args:
        path: Path to the tar.gz archive

    Returns:
        List of organization dictionaries

    Raises:
        RORParseError: If no suitable JSON file found in archive
    """
    # Files to look for, in priority order (prefer v2)
    target_patterns = [
        "ror-data_schema_v2.json",
        "ror-data.json",
        "_schema_v2.json",
    ]

    try:
        with tarfile.open(path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                name = member.name
                # Check if this file matches any target pattern
                for pattern in target_patterns:
                    if name.endswith(pattern):
                        f = tar.extractfile(member)
                        if f is not None:
                            data = json.load(f)
                            if isinstance(data, list):
                                return data

            # No suitable file found
            raise RORParseError(
                f"No ROR JSON file found in archive. Looked for patterns: {target_patterns}",
                source_path=str(path),
            )

    except tarfile.TarError as e:
        raise RORParseError(f"Invalid tar archive: {e}", source_path=str(path)) from e


def _load_from_zipfile(path: Path) -> list[dict[str, Any]]:
    """Load ROR JSON from a zip archive (Zenodo format).

    Looks for ror-data_schema_v2.json or ror-data.json inside the archive.

    Args:
        path: Path to the zip archive

    Returns:
        List of organization dictionaries

    Raises:
        RORParseError: If no suitable JSON file found in archive
    """
    # Files to look for, in priority order (prefer v2)
    target_patterns = [
        "ror-data_schema_v2.json",
        "ror-data.json",
        "_schema_v2.json",
    ]

    try:
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                # Check if this file matches any target pattern
                for pattern in target_patterns:
                    if name.endswith(pattern):
                        with zf.open(name) as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                return data

            # No suitable file found
            raise RORParseError(
                f"No ROR JSON file found in zip archive. Looked for patterns: {target_patterns}",
                source_path=str(path),
            )

    except zipfile.BadZipFile as e:
        raise RORParseError(f"Invalid zip archive: {e}", source_path=str(path)) from e


def normalize_ror(
    source_path: Path,
    source_id: str | None = "ror",
    active_only: bool = True,
) -> Iterator[EntityEntry]:
    """Normalize ROR data into EntityEntry records.

    Args:
        source_path: Path to ROR JSON file or tar.gz archive
        source_id: Source identifier for provenance tracking
        active_only: If True, skip inactive/withdrawn organizations

    Yields:
        EntityEntry records for each valid organization
    """
    organizations = load_ror_json(source_path)

    # Detect schema version
    is_v2 = is_schema_v2(organizations)

    for org in organizations:
        # Skip inactive organizations if requested
        if active_only:
            status = org.get("status", "active")
            if status in ("inactive", "withdrawn"):
                continue

        entry = parse_ror_organization(org, source_id=source_id, is_v2=is_v2)
        if entry is not None:
            yield entry


def normalize_ror_to_catalog(
    source_path: Path,
    output_path: Path,
    source_id: str | None = "ror",
    active_only: bool = True,
) -> tuple[int, str]:
    """Normalize ROR data and write to entity_catalog_institutions.jsonl.

    Args:
        source_path: Path to ROR JSON file or tar.gz archive
        output_path: Path for output JSONL file
        source_id: Source identifier for provenance tracking
        active_only: If True, skip inactive/withdrawn organizations

    Returns:
        Tuple of (entry count, SHA256 checksum)
    """
    from finetune.dataset_agent.writers import JSONLWriter

    writer = JSONLWriter(output_path)
    count = 0

    with writer:
        for entry in normalize_ror(
            source_path, source_id=source_id, active_only=active_only
        ):
            writer.write_line(entry)
            count += 1

    return count, writer.checksum
