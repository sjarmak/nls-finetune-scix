"""SWEET (Semantic Web for Earth and Environmental Terminology) ontology normalizer.

This module parses SWEET ontology data (Turtle/OWL files) and produces
normalized TopicEntry records for the topic_catalog_sweet.jsonl output.

SWEET source: https://github.com/ESIPFed/sweet
Format: RDF Turtle files (~224 .ttl files) with:
  - Concept URIs (OWL classes)
  - rdfs:label - preferred labels
  - skos:altLabel - alternative labels
  - rdfs:subClassOf / skos:broader - parent concepts
  - skos:narrower - child concepts (inferred from broader)

License: CC0-1.0
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from finetune.dataset_agent.schemas import TopicEntry

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

SWEET_NS = Namespace("http://sweetontology.net/")


class SWEETParseError(Exception):
    """Error parsing SWEET ontology data."""

    def __init__(self, message: str, source_path: str | None = None):
        self.source_path = source_path
        super().__init__(message)


def extract_concept_id(uri: str) -> str:
    """Extract a concept ID from a SWEET URI.

    Converts full URIs like
    'http://sweetontology.net/phenAtmo/Precipitation'
    to 'sweet:phenAtmo/Precipitation'.

    Args:
        uri: Full concept URI

    Returns:
        Concept ID (e.g., 'sweet:phenAtmo/Precipitation')
    """
    uri_str = str(uri)
    if "sweetontology.net/" in uri_str:
        local = uri_str.split("sweetontology.net/", 1)[1]
        return f"sweet:{local}"
    # Fallback: use fragment or last path segment
    if "#" in uri_str:
        return f"sweet:{uri_str.split('#')[-1]}"
    return f"sweet:{uri_str.rsplit('/', 1)[-1]}"


def _extract_labels(graph: Graph, subject: URIRef) -> tuple[str, list[str]]:
    """Extract preferred label and alternative labels for a concept.

    Checks rdfs:label first, then falls back to the local name from the URI.

    Args:
        graph: RDF graph
        subject: Concept URI

    Returns:
        Tuple of (preferred_label, alt_labels)
    """
    preferred = ""
    alt_labels: list[str] = []

    # Try rdfs:label
    for obj in graph.objects(subject, RDFS.label):
        label = str(obj).strip()
        if label and not preferred:
            preferred = label
        elif label:
            alt_labels.append(label)

    # Try skos:prefLabel
    if not preferred:
        for obj in graph.objects(subject, SKOS.prefLabel):
            label = str(obj).strip()
            if label:
                preferred = label
                break

    # Fallback: extract from URI fragment or local name
    if not preferred:
        uri_str = str(subject)
        if "#" in uri_str:
            preferred = uri_str.split("#")[-1]
        elif "/" in uri_str:
            preferred = uri_str.rsplit("/", 1)[-1]

    # Collect skos:altLabel values
    for obj in graph.objects(subject, SKOS.altLabel):
        label = str(obj).strip()
        if label and label != preferred:
            alt_labels.append(label)

    # Deduplicate alt_labels
    seen: set[str] = set()
    deduped: list[str] = []
    pref_lower = preferred.lower() if preferred else ""
    for label in alt_labels:
        key = label.lower()
        if key not in seen and key != pref_lower:
            seen.add(key)
            deduped.append(label)

    return preferred, deduped


def _extract_parents(graph: Graph, subject: URIRef) -> list[str]:
    """Extract parent concept IDs from broader/subClassOf relationships.

    Args:
        graph: RDF graph
        subject: Concept URI

    Returns:
        List of parent concept IDs
    """
    parents: list[str] = []
    seen: set[str] = set()

    # Check rdfs:subClassOf
    for obj in graph.objects(subject, RDFS.subClassOf):
        if isinstance(obj, URIRef) and "sweetontology.net/" in str(obj):
            parent_id = extract_concept_id(str(obj))
            if parent_id not in seen:
                seen.add(parent_id)
                parents.append(parent_id)

    # Check skos:broader
    for obj in graph.objects(subject, SKOS.broader):
        if isinstance(obj, URIRef) and "sweetontology.net/" in str(obj):
            parent_id = extract_concept_id(str(obj))
            if parent_id not in seen:
                seen.add(parent_id)
                parents.append(parent_id)

    return parents


def _extract_children(graph: Graph, subject: URIRef) -> list[str]:
    """Extract child concept IDs from narrower/subClassOf-inverse relationships.

    Args:
        graph: RDF graph
        subject: Concept URI

    Returns:
        List of child concept IDs
    """
    children: list[str] = []
    seen: set[str] = set()

    # Check skos:narrower
    for obj in graph.objects(subject, SKOS.narrower):
        if isinstance(obj, URIRef) and "sweetontology.net/" in str(obj):
            child_id = extract_concept_id(str(obj))
            if child_id not in seen:
                seen.add(child_id)
                children.append(child_id)

    # Infer from rdfs:subClassOf (inverse: subject is parent)
    for subj in graph.subjects(RDFS.subClassOf, subject):
        if isinstance(subj, URIRef) and "sweetontology.net/" in str(subj):
            child_id = extract_concept_id(str(subj))
            if child_id not in seen:
                seen.add(child_id)
                children.append(child_id)

    return children


def _is_sweet_concept(graph: Graph, subject: URIRef) -> bool:
    """Check if a URI is a SWEET ontology concept (OWL class).

    Args:
        graph: RDF graph
        subject: URI to check

    Returns:
        True if the URI is a SWEET concept
    """
    uri_str = str(subject)
    if "sweetontology.net/" not in uri_str:
        return False

    # Check if it's an OWL class
    if (subject, RDF.type, OWL.Class) in graph:
        return True

    # Check if it's a SKOS concept
    if (subject, RDF.type, SKOS.Concept) in graph:
        return True

    # Check if it has rdfs:label (many SWEET concepts are typed this way)
    if any(graph.objects(subject, RDFS.label)):
        return True

    return False


def parse_sweet_concept(
    graph: Graph,
    subject: URIRef,
    source_id: str | None = None,
) -> TopicEntry | None:
    """Parse a single SWEET concept into a TopicEntry.

    Args:
        graph: RDF graph containing the concept
        subject: Concept URI
        source_id: Optional source identifier for provenance

    Returns:
        TopicEntry or None if concept is invalid
    """
    topic_id = extract_concept_id(str(subject))
    preferred_label, alt_labels = _extract_labels(graph, subject)

    if not preferred_label:
        return None

    parents = _extract_parents(graph, subject)
    children = _extract_children(graph, subject)

    return TopicEntry(
        id=topic_id,
        label=preferred_label,
        aliases=alt_labels,
        parents=parents,
        children=children,
        source_id=source_id,
        domain_tags=["earthscience"],
        source_vocabulary="sweet",
    )


def load_sweet_graph(path: Path) -> Graph:
    """Load SWEET ontology from a file or tar.gz archive.

    Supports:
    - Single .ttl file (sweetAll.ttl)
    - Directory of .ttl files
    - tar.gz archive containing .ttl files

    Args:
        path: Path to a .ttl file, directory, or tar.gz archive

    Returns:
        Loaded RDF graph

    Raises:
        SWEETParseError: If the file cannot be parsed
    """
    graph = Graph()

    try:
        if path.suffix == ".gz" or path.name.endswith(".tar.gz"):
            return _load_from_tarball(path)
        elif path.is_dir():
            return _load_from_directory(path)
        else:
            # Single file (e.g., sweetAll.ttl)
            fmt = _guess_format(path)
            graph.parse(str(path), format=fmt)
            return graph
    except SWEETParseError:
        raise
    except Exception as e:
        raise SWEETParseError(
            f"Failed to parse SWEET ontology: {e}", source_path=str(path)
        ) from e


def _guess_format(path: Path) -> str:
    """Guess RDF serialization format from file extension.

    Args:
        path: File path

    Returns:
        Format string for rdflib ('turtle', 'xml', 'n3', etc.)
    """
    suffix = path.suffix.lower()
    if suffix in (".ttl", ".turtle"):
        return "turtle"
    elif suffix in (".owl", ".rdf", ".xml"):
        return "xml"
    elif suffix == ".n3":
        return "n3"
    elif suffix == ".nt":
        return "nt"
    elif suffix in (".jsonld", ".json"):
        return "json-ld"
    return "turtle"  # Default for SWEET


def _load_from_tarball(path: Path) -> Graph:
    """Load SWEET ontology from a tar.gz archive.

    Parses all .ttl files in the archive into a single merged graph.
    The ESIPFed/sweet repo stores concepts across ~226 individual .ttl files;
    sweetAll.ttl is just an OWL imports stub, not a merged ontology.

    Args:
        path: Path to the tar.gz archive

    Returns:
        Loaded RDF graph

    Raises:
        SWEETParseError: If no parseable files found
    """
    graph = Graph()
    files_parsed = 0

    try:
        with tarfile.open(path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".ttl") and member.isfile():
                    f = tar.extractfile(member)
                    if f is not None:
                        try:
                            data = f.read()
                            graph.parse(data=data, format="turtle")
                            files_parsed += 1
                        except Exception as e:
                            logger.warning(f"Skipping {member.name}: {e}")

    except tarfile.TarError as e:
        raise SWEETParseError(
            f"Invalid tar archive: {e}", source_path=str(path)
        ) from e

    if files_parsed == 0:
        raise SWEETParseError(
            "No Turtle files found in archive", source_path=str(path)
        )

    logger.info(f"Parsed {files_parsed} Turtle files from archive")
    return graph


def _load_from_directory(path: Path) -> Graph:
    """Load SWEET ontology from a directory of .ttl files.

    Args:
        path: Path to directory containing .ttl files

    Returns:
        Loaded RDF graph

    Raises:
        SWEETParseError: If no parseable files found
    """
    graph = Graph()
    files_parsed = 0

    # Look for sweetAll.ttl first
    sweet_all = path / "sweetAll.ttl"
    if sweet_all.exists():
        graph.parse(str(sweet_all), format="turtle")
        return graph

    # Fallback: parse individual .ttl files
    for ttl_file in sorted(path.rglob("*.ttl")):
        try:
            graph.parse(str(ttl_file), format="turtle")
            files_parsed += 1
        except Exception as e:
            logger.warning(f"Skipping {ttl_file.name}: {e}")

    if files_parsed == 0:
        raise SWEETParseError(
            "No Turtle files found in directory", source_path=str(path)
        )

    logger.info(f"Parsed {files_parsed} Turtle files from directory")
    return graph


def normalize_sweet(
    source_path: Path,
    source_id: str | None = "sweet",
) -> Iterator[TopicEntry]:
    """Normalize SWEET ontology into TopicEntry records.

    Args:
        source_path: Path to SWEET data (file, directory, or tar.gz)
        source_id: Source identifier for provenance tracking

    Yields:
        TopicEntry records for each valid concept
    """
    graph = load_sweet_graph(source_path)

    # Find all SWEET concepts
    seen_uris: set[str] = set()
    for subject in graph.subjects():
        if not isinstance(subject, URIRef):
            continue
        uri_str = str(subject)
        if uri_str in seen_uris:
            continue
        if not _is_sweet_concept(graph, subject):
            continue

        seen_uris.add(uri_str)
        entry = parse_sweet_concept(graph, subject, source_id=source_id)
        if entry is not None:
            yield entry


def normalize_sweet_to_catalog(
    source_path: Path,
    output_path: Path,
    source_id: str | None = "sweet",
) -> tuple[int, str]:
    """Normalize SWEET ontology and write to topic_catalog_sweet.jsonl.

    Args:
        source_path: Path to SWEET data (file, directory, or tar.gz)
        output_path: Path for output JSONL file
        source_id: Source identifier for provenance

    Returns:
        Tuple of (entry count, SHA256 checksum)
    """
    from finetune.dataset_agent.writers import JSONLWriter

    writer = JSONLWriter(output_path)
    count = 0

    with writer:
        for entry in normalize_sweet(source_path, source_id=source_id):
            writer.write_line(entry)
            count += 1

    logger.info(f"SWEET normalization complete: {count} concepts written to {output_path}")
    return count, writer.checksum
