"""USGS/IAU Planetary Nomenclature normalizer.

This module parses USGS Planetary Gazetteer shapefiles and produces
normalized EntityEntry records for the entity_catalog_planetary.jsonl output.

Source: https://planetarynames.wr.usgs.gov/
Format: ESRI Shapefiles (.shp/.dbf/.shx) containing named planetary surface
features with attributes:
  - Feature_Name / Clean_Feature_Name: Feature name
  - Feature_Type: Type of feature (crater, mons, vallis, etc.)
  - Target: Target body (Mars, Moon, etc.)
  - Diameter: Feature diameter in km
  - Center_Lat / Center_Lon: Center coordinates
  - Approval_Status: IAU approval status
  - Feature_ID: Unique feature identifier

License: Public Domain (USGS)
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import shapefile

from finetune.dataset_agent.schemas import EntityEntry

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Target bodies to include by default (Mars and Moon per requirements)
DEFAULT_TARGET_BODIES = frozenset({"Mars", "Moon"})

# Feature attribute field names in USGS shapefiles
FIELD_FEATURE_ID = "Feature_ID"
FIELD_FEATURE_NAME = "Feature_Name"
FIELD_CLEAN_NAME = "Clean_Feature_Name"
FIELD_FEATURE_TYPE = "Feature_Type"
FIELD_TARGET = "Target"
FIELD_DIAMETER = "Diameter"
FIELD_CENTER_LAT = "Center_Lat"
FIELD_CENTER_LON = "Center_Lon"
FIELD_APPROVAL_STATUS = "Approval_Status"

# Map variant field names to canonical names.
# DBF format truncates field names to 10 characters, and newer USGS
# shapefiles (2026+) use all-lowercase names. Both are mapped here.
_FIELD_ALIASES: dict[str, str] = {
    # DBF 10-char truncation aliases
    "Feature_Na": FIELD_FEATURE_NAME,
    "Clean_Feat": FIELD_CLEAN_NAME,
    "Feature_Ty": FIELD_FEATURE_TYPE,
    "Approval_S": FIELD_APPROVAL_STATUS,
    # Newer USGS lowercase field names (2026+ per-body shapefiles)
    "name": FIELD_FEATURE_NAME,
    "clean_name": FIELD_CLEAN_NAME,
    "type": FIELD_FEATURE_TYPE,
    "approval": FIELD_APPROVAL_STATUS,
    "diameter": FIELD_DIAMETER,
    "center_lat": FIELD_CENTER_LAT,
    "center_lon": FIELD_CENTER_LON,
}


class PlanetaryParseError(Exception):
    """Error parsing planetary nomenclature data."""

    def __init__(self, message: str, source_path: str | None = None):
        self.source_path = source_path
        super().__init__(message)


def _build_feature_id(feature_id: str | int, target: str) -> str:
    """Build a canonical entity ID for a planetary feature.

    Args:
        feature_id: USGS feature identifier
        target: Target body name (e.g., 'Mars', 'Moon')

    Returns:
        Entity ID (e.g., 'planetary:Mars/12345')
    """
    return f"planetary:{target}/{feature_id}"


def _normalize_feature_type(feature_type: str) -> str:
    """Normalize a feature type string to a lowercase entity_subtype.

    Args:
        feature_type: Raw feature type from shapefile (e.g., 'Crater, craters')

    Returns:
        Normalized type (e.g., 'crater')
    """
    if not feature_type:
        return ""
    # Take the first word before comma or space-separated plural info
    # e.g., "Crater, craters" -> "crater"
    base = feature_type.split(",")[0].strip()
    return base.lower()


def _safe_float(value: Any) -> float | None:
    """Safely convert a value to float, returning None on failure.

    Args:
        value: Value to convert

    Returns:
        Float value or None
    """
    if value is None:
        return None
    try:
        result = float(value)
        return result
    except (ValueError, TypeError):
        return None


def _safe_str(value: Any) -> str:
    """Safely convert a value to string, stripping whitespace.

    Args:
        value: Value to convert

    Returns:
        Stripped string, or empty string if None
    """
    if value is None:
        return ""
    return str(value).strip()


def _extract_feature_id_from_link(link: str) -> str | None:
    """Extract the numeric feature ID from a USGS link URL.

    Args:
        link: URL like 'http://planetarynames.wr.usgs.gov/Feature/15149'

    Returns:
        Feature ID string (e.g., '15149'), or None if not extractable
    """
    if not link:
        return None
    parts = link.rstrip("/").rsplit("/", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1]
    return None


def _infer_target_from_source_id(source_id: str | None) -> str | None:
    """Infer target body from source_id for per-body shapefiles.

    Args:
        source_id: Source identifier (e.g., 'planetary_mars', 'planetary_moon')

    Returns:
        Target body name (e.g., 'Mars', 'Moon') or None
    """
    if not source_id:
        return None
    lower = source_id.lower()
    if "mars" in lower:
        return "Mars"
    if "moon" in lower:
        return "Moon"
    if "mercury" in lower:
        return "Mercury"
    if "venus" in lower:
        return "Venus"
    return None


def parse_planetary_record(
    record: dict[str, Any],
    source_id: str | None = None,
    default_target: str | None = None,
) -> EntityEntry | None:
    """Parse a single shapefile record into an EntityEntry.

    Args:
        record: Dictionary of field values from a shapefile record
        source_id: Optional source identifier for provenance
        default_target: Default target body (for per-body shapefiles that lack Target field)

    Returns:
        EntityEntry or None if record is invalid
    """
    # Resolve feature ID: try Feature_ID first, then extract from link URL
    feature_id = record.get(FIELD_FEATURE_ID)
    if feature_id is None or str(feature_id).strip() == "":
        link = _safe_str(record.get("link", ""))
        feature_id = _extract_feature_id_from_link(link)
    if feature_id is None or str(feature_id).strip() == "":
        return None

    # Prefer Clean_Feature_Name, fall back to Feature_Name
    name = _safe_str(record.get(FIELD_CLEAN_NAME))
    if not name:
        name = _safe_str(record.get(FIELD_FEATURE_NAME))
    if not name:
        return None

    # Resolve target: from record, then default, then infer from source_id
    target = _safe_str(record.get(FIELD_TARGET))
    if not target:
        target = default_target or _infer_target_from_source_id(source_id) or ""
    if not target:
        return None

    entity_id = _build_feature_id(str(feature_id).strip(), target)
    feature_type = _normalize_feature_type(_safe_str(record.get(FIELD_FEATURE_TYPE)))

    # Build aliases from the other name field if different
    aliases: list[str] = []
    alt_name = _safe_str(record.get(FIELD_FEATURE_NAME))
    if alt_name and alt_name != name:
        aliases.append(alt_name)

    # Build metadata
    metadata: dict[str, Any] = {"target": target}

    diameter = _safe_float(record.get(FIELD_DIAMETER))
    if diameter is not None:
        metadata["diameter_km"] = diameter

    center_lat = _safe_float(record.get(FIELD_CENTER_LAT))
    center_lon = _safe_float(record.get(FIELD_CENTER_LON))
    if center_lat is not None:
        metadata["center_lat"] = center_lat
    if center_lon is not None:
        metadata["center_lon"] = center_lon

    approval_status = _safe_str(record.get(FIELD_APPROVAL_STATUS))
    if approval_status:
        metadata["approval_status"] = approval_status

    if feature_type:
        metadata["feature_type"] = feature_type

    return EntityEntry(
        id=entity_id,
        label=name,
        aliases=aliases,
        metadata=metadata,
        source_id=source_id,
        entity_subtype=feature_type,
        domain_tags=["planetary"],
        source_vocabulary="planetary",
    )


def _read_shapefile_records(sf: shapefile.Reader) -> Iterator[dict[str, Any]]:
    """Read all records from a shapefile Reader as dictionaries.

    Maps truncated DBF field names to canonical names using _FIELD_ALIASES.

    Args:
        sf: shapefile.Reader instance

    Yields:
        Dictionary of canonical_field_name -> value for each record
    """
    # Get field names (skip the deletion flag field at index 0)
    raw_field_names = [field[0] for field in sf.fields[1:]]
    # Map truncated names to canonical names
    field_names = [_FIELD_ALIASES.get(name, name) for name in raw_field_names]

    for shape_rec in sf.iterShapeRecords():
        record_dict: dict[str, Any] = {}
        for i, field_name in enumerate(field_names):
            if i < len(shape_rec.record):
                record_dict[field_name] = shape_rec.record[i]
        yield record_dict


def load_planetary_shapefile(path: Path) -> shapefile.Reader:
    """Load a planetary nomenclature shapefile.

    Supports:
    - Direct .shp file
    - .zip archive containing .shp/.dbf/.shx files
    - Directory containing .shp/.dbf/.shx files

    Args:
        path: Path to shapefile, zip archive, or directory

    Returns:
        shapefile.Reader instance

    Raises:
        PlanetaryParseError: If the shapefile cannot be loaded
    """
    try:
        if path.suffix == ".zip" or path.name.endswith(".zip"):
            return _load_from_zip(path)
        elif path.suffix == ".shp":
            return shapefile.Reader(str(path))
        elif path.is_dir():
            return _load_from_directory(path)
        else:
            # Try loading directly (pyshp may handle it)
            return shapefile.Reader(str(path))
    except PlanetaryParseError:
        raise
    except Exception as e:
        raise PlanetaryParseError(
            f"Failed to load shapefile: {e}", source_path=str(path)
        ) from e


def _load_from_zip(path: Path) -> shapefile.Reader:
    """Load shapefile from a zip archive.

    Reads the zip into memory and creates a shapefile.Reader from the
    .shp, .dbf, and .shx components.

    Args:
        path: Path to the zip archive

    Returns:
        shapefile.Reader instance

    Raises:
        PlanetaryParseError: If no suitable shapefile found
    """
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()

            # Find .shp file
            shp_files = [n for n in names if n.endswith(".shp")]
            if not shp_files:
                raise PlanetaryParseError(
                    "No .shp file found in zip archive", source_path=str(path)
                )

            shp_name = shp_files[0]
            base_name = shp_name[:-4]  # Remove .shp extension

            # Read required components
            shp_data = io.BytesIO(zf.read(shp_name))

            dbf_name = base_name + ".dbf"
            if dbf_name not in names:
                raise PlanetaryParseError(
                    f"Missing .dbf file for {shp_name}", source_path=str(path)
                )
            dbf_data = io.BytesIO(zf.read(dbf_name))

            # .shx is optional (pyshp can reconstruct it)
            shx_data = None
            shx_name = base_name + ".shx"
            if shx_name in names:
                shx_data = io.BytesIO(zf.read(shx_name))

            return shapefile.Reader(shp=shp_data, dbf=dbf_data, shx=shx_data)

    except zipfile.BadZipFile as e:
        raise PlanetaryParseError(
            f"Invalid zip archive: {e}", source_path=str(path)
        ) from e


def _load_from_directory(path: Path) -> shapefile.Reader:
    """Load shapefile from a directory.

    Searches for .shp files in the directory.

    Args:
        path: Directory path

    Returns:
        shapefile.Reader instance

    Raises:
        PlanetaryParseError: If no suitable shapefile found
    """
    shp_files = sorted(path.glob("*.shp"))
    if not shp_files:
        # Search recursively
        shp_files = sorted(path.rglob("*.shp"))

    if not shp_files:
        raise PlanetaryParseError(
            "No .shp files found in directory", source_path=str(path)
        )

    return shapefile.Reader(str(shp_files[0]))


def normalize_planetary(
    source_path: Path,
    source_id: str | None = "planetary",
    target_bodies: frozenset[str] | None = None,
) -> Iterator[EntityEntry]:
    """Normalize planetary nomenclature data into EntityEntry records.

    Args:
        source_path: Path to shapefile, zip archive, or directory
        source_id: Source identifier for provenance tracking
        target_bodies: Set of target body names to include (None = all bodies)

    Yields:
        EntityEntry records for each valid feature
    """
    sf = load_planetary_shapefile(source_path)

    # Infer default target from source_id for per-body shapefiles
    default_target = _infer_target_from_source_id(source_id)

    seen_ids: set[str] = set()
    for record in _read_shapefile_records(sf):
        # Filter by target body if specified
        if target_bodies is not None:
            target = _safe_str(record.get(FIELD_TARGET)) or default_target or ""
            if target not in target_bodies:
                continue

        entry = parse_planetary_record(
            record, source_id=source_id, default_target=default_target
        )
        if entry is None:
            continue
        if entry.id in seen_ids:
            continue
        seen_ids.add(entry.id)
        yield entry


def normalize_planetary_to_catalog(
    source_path: Path,
    output_path: Path,
    source_id: str | None = "planetary",
    target_bodies: frozenset[str] | None = None,
) -> tuple[int, str]:
    """Normalize planetary features and write to entity_catalog_planetary.jsonl.

    Args:
        source_path: Path to shapefile, zip archive, or directory
        output_path: Path for output JSONL file
        source_id: Source identifier for provenance
        target_bodies: Set of target body names to include (None = all bodies)

    Returns:
        Tuple of (entry count, SHA256 checksum)
    """
    from finetune.dataset_agent.writers import JSONLWriter

    writer = JSONLWriter(output_path)
    count = 0

    with writer:
        for entry in normalize_planetary(
            source_path, source_id=source_id, target_bodies=target_bodies
        ):
            writer.write_line(entry)
            count += 1

    logger.info(
        f"Planetary normalization complete: {count} features written to {output_path}"
    )
    return count, writer.checksum
