"""IntentSpec dataclass for structured intent extraction from natural language.

This module defines the typed intent specification that the NER extractor produces
and the assembler consumes. It represents the structured understanding of a user's
search intent before query generation.

The IntentSpec is the contract between:
- NER extraction (produces IntentSpec)
- Few-shot retrieval (uses IntentSpec for similarity)
- Query assembly (consumes IntentSpec to build ADS query)
"""

from dataclasses import asdict, dataclass, field
from json import dumps as json_dumps
from json import loads as json_loads

# Valid ADS operators that wrap queries
# Note: Do NOT add new operators here without adding corresponding gating rules in ner.py
OPERATORS: frozenset[str] = frozenset(
    {
        "citations",  # Find papers that cite the search results
        "references",  # Find papers referenced by the search results
        "trending",  # Find currently popular papers
        "useful",  # Find high-utility papers
        "similar",  # Find textually similar papers
        "reviews",  # Find review articles
    }
)


@dataclass
class IntentSpec:
    """Structured representation of user's search intent.

    This dataclass is the intermediate representation between natural language
    input and ADS query syntax. All fields are extracted by NER and validated
    against FIELD_ENUMS before use.

    Attributes:
        free_text_terms: Topic phrases to search in abs:/title: fields (AND'd together)
        or_terms: Topic phrases that should be OR'd together (e.g., "rocks or volcanoes")
        authors: Author names (will be formatted as "Last, F")
        affiliations: Institutional affiliations for aff: field
        objects: Astronomical objects for object: field
        year_from: Start year for pubdate range (inclusive)
        year_to: End year for pubdate range (inclusive)
        doctype: Document types (must be in DOCTYPES enum)
        property: Record properties (must be in PROPERTIES enum)
        collection: Collection/discipline filter (must be in COLLECTIONS enum)
        bibgroup: Bibliographic groups (must be in BIBGROUPS enum)
        esources: Electronic source types (must be in ESOURCES enum)
        data: Data archive sources (must be in DATA_SOURCES enum)
        operator: Optional wrapper operator (must be in OPERATORS set)
        operator_target: Optional target for operator (e.g., bibcode)
        raw_user_text: Original user input (preserved for debugging)
        confidence: Confidence scores for each extracted field
    """

    # Free text fields
    free_text_terms: list[str] = field(default_factory=list)
    or_terms: list[str] = field(default_factory=list)  # Topics to combine with OR
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)

    # Year range
    year_from: int | None = None
    year_to: int | None = None

    # Constrained enum fields (must be validated against FIELD_ENUMS)
    doctype: set[str] = field(default_factory=set)
    property: set[str] = field(default_factory=set)
    collection: set[str] = field(default_factory=set)
    bibgroup: set[str] = field(default_factory=set)
    esources: set[str] = field(default_factory=set)
    data: set[str] = field(default_factory=set)

    # Operator fields
    operator: str | None = None
    operator_target: str | None = None

    # Metadata
    raw_user_text: str = ""
    confidence: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate operator is in OPERATORS set if provided."""
        if self.operator is not None and self.operator not in OPERATORS:
            raise ValueError(
                f"Invalid operator '{self.operator}'. Must be one of: {sorted(OPERATORS)}"
            )

    def has_constraints(self) -> bool:
        """Check if any constrained fields are set."""
        return bool(
            self.doctype
            or self.property
            or self.collection
            or self.bibgroup
            or self.esources
            or self.data
        )

    def has_content(self) -> bool:
        """Check if the intent has any searchable content."""
        return bool(
            self.free_text_terms
            or self.or_terms
            or self.authors
            or self.affiliations
            or self.objects
            or self.year_from
            or self.year_to
            or self.has_constraints()
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Note: Sets are converted to sorted lists for deterministic output.
        """
        d = asdict(self)
        # Convert sets to sorted lists for JSON serialization
        for key in ("doctype", "property", "collection", "bibgroup", "esources", "data"):
            d[key] = sorted(d[key])
        return d

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json_dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "IntentSpec":
        """Create IntentSpec from dictionary.

        Handles conversion of lists back to sets for enum fields.
        """
        # Convert lists to sets for enum fields
        for key in ("doctype", "property", "collection", "bibgroup", "esources", "data"):
            if key in d and isinstance(d[key], list):
                d[key] = set(d[key])
        return cls(**d)

    @classmethod
    def from_json(cls, json_str: str) -> "IntentSpec":
        """Deserialize from JSON string."""
        return cls.from_dict(json_loads(json_str))

    def __repr__(self) -> str:
        """Compact representation for debugging."""
        parts = []
        if self.free_text_terms:
            parts.append(f"topics={self.free_text_terms}")
        if self.or_terms:
            parts.append(f"or_topics={self.or_terms}")
        if self.authors:
            parts.append(f"authors={self.authors}")
        if self.year_from or self.year_to:
            parts.append(f"years={self.year_from}-{self.year_to}")
        if self.operator:
            parts.append(f"op={self.operator}")
        if self.has_constraints():
            constraints = []
            if self.doctype:
                constraints.append(f"doctype={sorted(self.doctype)}")
            if self.property:
                constraints.append(f"property={sorted(self.property)}")
            if self.bibgroup:
                constraints.append(f"bibgroup={sorted(self.bibgroup)}")
            parts.extend(constraints)
        return f"IntentSpec({', '.join(parts)})"
