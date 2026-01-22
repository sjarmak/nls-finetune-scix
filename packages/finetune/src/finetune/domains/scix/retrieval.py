"""Few-shot retrieval module for gold examples.

This module provides lightweight similarity scoring for retrieving
similar gold examples to guide query assembly. Uses token overlap
with feature-based boosting for fast retrieval.

Performance target: <20ms for k=5 on 4000 examples.
"""

import math
import os
import re
from dataclasses import dataclass, field
from json import load as json_load
from pathlib import Path

from .intent_spec import OPERATORS, IntentSpec
from .pipeline import GoldExample

# Default path to gold examples file (relative to package root)
DEFAULT_GOLD_EXAMPLES_PATH = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "data" / "datasets" / "raw" / "gold_examples.json"


@dataclass
class IndexedExample:
    """Pre-processed gold example with tokenized fields for fast retrieval.

    Attributes:
        nl_query: Original natural language query
        ads_query: Corresponding ADS query
        category: Category from gold_examples.json
        nl_tokens: Tokenized NL query (lowercased, stopwords removed)
        has_author: Whether query contains author: field
        has_year: Whether query contains year/pubdate field
        has_operator: Which operator if any (e.g., 'citations')
        operators: Set of operators present in query
        doctypes: Set of doctypes in query
        properties: Set of properties in query
        bibgroups: Set of bibgroups in query
        databases: Set of databases in query
        idf_weights: Pre-computed IDF weights for tokens
    """

    nl_query: str
    ads_query: str
    category: str
    nl_tokens: set[str] = field(default_factory=set)
    has_author: bool = False
    has_year: bool = False
    has_operator: str | None = None
    operators: set[str] = field(default_factory=set)
    doctypes: set[str] = field(default_factory=set)
    properties: set[str] = field(default_factory=set)
    bibgroups: set[str] = field(default_factory=set)
    databases: set[str] = field(default_factory=set)


# Stopwords to filter from NL queries for better matching
STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "about", "above", "after", "before", "between", "into", "through",
    "during", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few",
    "more", "most", "other", "some", "such", "no", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "also",
    "now", "papers", "paper", "articles", "article", "studies", "study",
    "research", "work", "works", "published", "find", "search", "looking",
    "want", "need", "show", "me", "get", "list", "i", "my", "we", "our",
    "you", "your", "it", "its", "this", "that", "these", "those", "which",
    "what", "who", "whose", "whom", "any", "every", "many"
})


def tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase words, removing stopwords.

    Args:
        text: Input text to tokenize

    Returns:
        Set of lowercase tokens with stopwords removed
    """
    # Simple word tokenization: split on non-alphanumeric
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def extract_features_from_ads_query(ads_query: str) -> dict:
    """Extract features from an ADS query string.

    Args:
        ads_query: ADS query string

    Returns:
        Dict with extracted features (operators, doctypes, etc.)
    """
    features: dict = {
        "has_author": False,
        "has_year": False,
        "operator": None,
        "operators": set(),
        "doctypes": set(),
        "properties": set(),
        "bibgroups": set(),
        "databases": set(),
    }

    query_lower = ads_query.lower()

    # Check for author
    if "author:" in query_lower:
        features["has_author"] = True

    # Check for year/pubdate
    if "year:" in query_lower or "pubdate:" in query_lower:
        features["has_year"] = True

    # Check for operators
    for op in OPERATORS:
        if f"{op}(" in query_lower:
            features["operators"].add(op)
            if features["operator"] is None:
                features["operator"] = op

    # Extract doctype values
    for match in re.finditer(r"doctype:([a-z_]+)", query_lower):
        features["doctypes"].add(match.group(1))

    # Extract property values
    for match in re.finditer(r"property:([a-z_]+)", query_lower):
        features["properties"].add(match.group(1))

    # Extract bibgroup values
    for match in re.finditer(r"bibgroup:([a-zA-Z0-9_/-]+)", ads_query):
        features["bibgroups"].add(match.group(1))

    # Extract database values
    for match in re.finditer(r"database:([a-z]+)", query_lower):
        features["databases"].add(match.group(1))

    return features


class GoldExampleIndex:
    """In-memory index for gold examples with precomputed features.

    This class loads gold examples at startup and precomputes tokens
    and features for fast retrieval.
    """

    def __init__(self, examples: list[dict] | None = None, filepath: str | Path | None = None):
        """Initialize the index from examples or file.

        Args:
            examples: List of example dicts (if None, loads from filepath)
            filepath: Path to gold_examples.json (uses default if None)
        """
        self._examples: list[IndexedExample] = []
        self._document_frequencies: dict[str, int] = {}
        self._total_docs: int = 0
        self._avg_doc_length: float = 0.0

        if examples is not None:
            self._load_examples(examples)
        else:
            self._load_from_file(filepath or DEFAULT_GOLD_EXAMPLES_PATH)

    def _load_from_file(self, filepath: str | Path) -> None:
        """Load examples from JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            # Try alternative path from environment
            alt_path = os.environ.get("GOLD_EXAMPLES_PATH")
            if alt_path and Path(alt_path).exists():
                filepath = Path(alt_path)
            else:
                raise FileNotFoundError(f"Gold examples file not found: {filepath}")

        with open(filepath) as f:
            examples = json_load(f)
        self._load_examples(examples)

    def _load_examples(self, examples: list[dict]) -> None:
        """Process and index examples."""
        self._examples = []
        token_doc_count: dict[str, int] = {}
        total_tokens = 0

        for ex in examples:
            nl = ex.get("natural_language", "")
            ads = ex.get("ads_query", "")
            category = ex.get("category", "")

            tokens = tokenize(nl)
            features = extract_features_from_ads_query(ads)

            indexed = IndexedExample(
                nl_query=nl,
                ads_query=ads,
                category=category,
                nl_tokens=tokens,
                has_author=features["has_author"],
                has_year=features["has_year"],
                has_operator=features["operator"],
                operators=features["operators"],
                doctypes=features["doctypes"],
                properties=features["properties"],
                bibgroups=features["bibgroups"],
                databases=features["databases"],
            )
            self._examples.append(indexed)

            # Count document frequencies for IDF
            for token in tokens:
                if token not in token_doc_count:
                    token_doc_count[token] = 0
                token_doc_count[token] += 1

            total_tokens += len(tokens)

        self._document_frequencies = token_doc_count
        self._total_docs = len(self._examples)
        self._avg_doc_length = total_tokens / max(self._total_docs, 1)

    def _compute_idf(self, token: str) -> float:
        """Compute IDF score for a token using BM25 formula.

        Args:
            token: Token to compute IDF for

        Returns:
            IDF weight for the token
        """
        df = self._document_frequencies.get(token, 0)
        if df == 0:
            return 0.0
        # BM25 IDF formula
        return math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1)

    def _compute_similarity(self, intent: IntentSpec, example: IndexedExample) -> float:
        """Compute similarity score between intent and indexed example.

        Uses BM25-like scoring with feature boosting.

        Args:
            intent: User intent specification
            example: Indexed gold example

        Returns:
            Similarity score (higher is more similar)
        """
        score = 0.0

        # BM25 parameters
        k1 = 1.5
        b = 0.75

        # Get query tokens from intent
        query_tokens = set()
        for term in intent.free_text_terms:
            query_tokens.update(tokenize(term))
        for author in intent.authors:
            query_tokens.update(tokenize(author))

        # BM25-like token overlap scoring
        doc_len = len(example.nl_tokens)
        for token in query_tokens:
            if token in example.nl_tokens:
                idf = self._compute_idf(token)
                # Simplified BM25 term frequency (1 if present)
                tf = 1.0
                # Document length normalization
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / max(self._avg_doc_length, 1)))
                score += idf * (numerator / denominator)

        # Feature-based boosting

        # Boost for matching operator (highest priority)
        if intent.operator and intent.operator == example.has_operator:
            score += 5.0
        elif intent.operator and intent.operator in example.operators:
            score += 3.0

        # Boost for matching author presence
        if intent.authors and example.has_author:
            score += 2.0

        # Boost for matching year presence
        if (intent.year_from or intent.year_to) and example.has_year:
            score += 1.5

        # Boost for matching doctype
        if intent.doctype:
            overlap = intent.doctype & example.doctypes
            score += len(overlap) * 1.5

        # Boost for matching property
        if intent.property:
            overlap = intent.property & example.properties
            score += len(overlap) * 1.5

        # Boost for matching bibgroup
        if intent.bibgroup:
            overlap = intent.bibgroup & example.bibgroups
            score += len(overlap) * 1.5

        # Boost for matching database
        if intent.database:
            overlap = intent.database & example.databases
            score += len(overlap) * 1.5

        # Boost for category match (if we can infer from intent)
        if intent.operator and example.category in ("operator", "citations", "second_order"):
            score += 1.0
        if intent.authors and example.category in ("author", "first_author"):
            score += 1.0
        if intent.property and example.category in ("property", "properties", "conversational"):
            score += 1.0

        return score

    def retrieve(self, intent: IntentSpec, k: int = 5) -> list[GoldExample]:
        """Retrieve top-k similar gold examples for an intent.

        Args:
            intent: User intent specification
            k: Number of examples to retrieve

        Returns:
            List of GoldExample objects sorted by similarity (highest first)
        """
        if not self._examples:
            return []

        # Score all examples
        scored = []
        for example in self._examples:
            score = self._compute_similarity(intent, example)
            if score > 0:  # Only include examples with positive score
                scored.append((score, example))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top-k and convert to GoldExample
        results = []
        for score, example in scored[:k]:
            features = {
                "category": example.category,
                "has_author": example.has_author,
                "has_year": example.has_year,
                "operator": example.has_operator,
                "doctypes": sorted(example.doctypes),
                "properties": sorted(example.properties),
            }
            results.append(GoldExample(
                nl_query=example.nl_query,
                ads_query=example.ads_query,
                features=features,
                score=score,
            ))

        return results

    def __len__(self) -> int:
        """Return the number of indexed examples."""
        return len(self._examples)


# Global singleton index (lazy loaded)
_global_index: GoldExampleIndex | None = None


def get_index(filepath: str | Path | None = None) -> GoldExampleIndex:
    """Get or create the global gold example index.

    This provides a singleton pattern for the index to avoid
    reloading on every request.

    Args:
        filepath: Optional custom path to gold_examples.json

    Returns:
        GoldExampleIndex instance
    """
    global _global_index
    if _global_index is None:
        _global_index = GoldExampleIndex(filepath=filepath)
    return _global_index


def reset_index() -> None:
    """Reset the global index (for testing)."""
    global _global_index
    _global_index = None


def retrieve_similar(intent: IntentSpec, k: int = 5) -> list[GoldExample]:
    """Retrieve similar gold examples for an intent.

    This is the main entry point for retrieval.

    Args:
        intent: User intent specification
        k: Number of examples to retrieve (default: 5)

    Returns:
        List of GoldExample objects sorted by similarity
    """
    index = get_index()
    return index.retrieve(intent, k)
