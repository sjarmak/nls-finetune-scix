"""Artifact writers for dataset generation pipeline.

This module provides writers for JSON and JSONL formats, ensuring consistent
output formatting across all pipeline artifacts.

JSONL (JSON Lines) format:
- One JSON object per line
- No trailing commas or array wrappers
- Used for streaming/large datasets: topic_catalog, entity_catalog, pairs, enrichment_labels

JSON format:
- Pretty-printed with indent=2
- Used for manifests, reports, and small config files
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class HasToDict(Protocol):
    """Protocol for objects with a to_dict method."""

    def to_dict(self) -> dict[str, Any]: ...


T = TypeVar("T", bound=HasToDict)


def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of bytes data."""
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class JSONWriter:
    """Writer for single JSON objects to files.

    Used for manifests, reports, and configuration files.
    """

    def __init__(self, path: Path, indent: int = 2) -> None:
        """Initialize writer.

        Args:
            path: Output file path
            indent: JSON indentation level
        """
        self.path = path
        self.indent = indent

    def write(self, obj: HasToDict | dict[str, Any]) -> str:
        """Write object to JSON file.

        Args:
            obj: Object with to_dict() method or plain dict

        Returns:
            SHA256 checksum of written content
        """
        if hasattr(obj, "to_dict"):
            data = obj.to_dict()
        else:
            data = obj

        content = json.dumps(data, indent=self.indent, ensure_ascii=False)
        content_bytes = content.encode("utf-8")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(content)

        return compute_sha256(content_bytes)

    def read(self) -> dict[str, Any]:
        """Read JSON file and return as dict."""
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)


class JSONLWriter:
    """Writer for JSONL (JSON Lines) format.

    Writes one JSON object per line, suitable for streaming large datasets.
    Used for catalogs, pairs, and enrichment labels.
    """

    def __init__(self, path: Path) -> None:
        """Initialize writer.

        Args:
            path: Output file path
        """
        self.path = path
        self._line_count = 0
        self._file_handle = None
        self._hasher = hashlib.sha256()

    def __enter__(self) -> "JSONLWriter":
        """Open file for writing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = open(self.path, "w", encoding="utf-8")
        self._line_count = 0
        self._hasher = hashlib.sha256()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close file handle."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def write_line(self, obj: HasToDict | dict[str, Any]) -> None:
        """Write a single object as one JSONL line.

        Args:
            obj: Object with to_dict() method or plain dict
        """
        if self._file_handle is None:
            raise RuntimeError("Writer not opened. Use 'with' statement.")

        if hasattr(obj, "to_dict"):
            data = obj.to_dict()
        else:
            data = obj

        line = json.dumps(data, ensure_ascii=False) + "\n"
        line_bytes = line.encode("utf-8")
        self._file_handle.write(line)
        self._hasher.update(line_bytes)
        self._line_count += 1

    def write_all(self, objects: list[HasToDict | dict[str, Any]]) -> tuple[str, int]:
        """Write all objects to JSONL file.

        Convenience method that handles opening/closing.

        Args:
            objects: List of objects to write

        Returns:
            Tuple of (SHA256 checksum, line count)
        """
        with self:
            for obj in objects:
                self.write_line(obj)
        return self.checksum, self.line_count

    @property
    def line_count(self) -> int:
        """Number of lines written."""
        return self._line_count

    @property
    def checksum(self) -> str:
        """SHA256 checksum of all content written."""
        return self._hasher.hexdigest()


class JSONLReader:
    """Reader for JSONL (JSON Lines) format.

    Reads one JSON object per line, supporting streaming large files.
    """

    def __init__(self, path: Path) -> None:
        """Initialize reader.

        Args:
            path: Input file path
        """
        self.path = path

    def __iter__(self):
        """Iterate over lines in the file."""
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def read_all(self) -> list[dict[str, Any]]:
        """Read all lines into a list."""
        return list(self)

    def count(self) -> int:
        """Count number of lines without loading into memory."""
        count = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


def write_json(path: Path, obj: HasToDict | dict[str, Any], indent: int = 2) -> str:
    """Convenience function to write a single JSON object.

    Args:
        path: Output file path
        obj: Object with to_dict() method or plain dict
        indent: JSON indentation level

    Returns:
        SHA256 checksum of written content
    """
    writer = JSONWriter(path, indent=indent)
    return writer.write(obj)


def write_jsonl(path: Path, objects: list[HasToDict | dict[str, Any]]) -> tuple[str, int]:
    """Convenience function to write JSONL file.

    Args:
        path: Output file path
        objects: List of objects to write

    Returns:
        Tuple of (SHA256 checksum, line count)
    """
    writer = JSONLWriter(path)
    return writer.write_all(objects)


def read_json(path: Path) -> dict[str, Any]:
    """Convenience function to read a JSON file.

    Args:
        path: Input file path

    Returns:
        Parsed JSON as dict
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Convenience function to read all lines from JSONL file.

    Args:
        path: Input file path

    Returns:
        List of parsed JSON objects
    """
    return JSONLReader(path).read_all()
