"""Unit tests for IntentSpec dataclass and pipeline skeleton.

Tests for US-001: Define IntentSpec dataclass and pipeline skeleton.
"""

import pytest
import json

from finetune.domains.scix.intent_spec import IntentSpec, OPERATORS
from finetune.domains.scix.pipeline import (
    process_query,
    PipelineResult,
    GoldExample,
    DebugInfo,
    is_ads_query,
)


class TestIntentSpec:
    """Tests for IntentSpec dataclass."""
    
    def test_instantiate_default(self):
        """IntentSpec can be instantiated with defaults."""
        spec = IntentSpec()
        assert spec.free_text_terms == []
        assert spec.authors == []
        assert spec.year_from is None
        assert spec.year_to is None
        assert spec.operator is None
        assert spec.doctype == set()
        assert spec.property == set()
    
    def test_instantiate_with_values(self):
        """IntentSpec can be instantiated with specific values."""
        spec = IntentSpec(
            free_text_terms=["exoplanets", "habitable zones"],
            authors=["Hawking, S"],
            year_from=2020,
            year_to=2024,
            doctype={"article", "eprint"},
            property={"refereed"},
            operator="citations",
            raw_user_text="papers about exoplanets",
        )
        
        assert spec.free_text_terms == ["exoplanets", "habitable zones"]
        assert spec.authors == ["Hawking, S"]
        assert spec.year_from == 2020
        assert spec.year_to == 2024
        assert spec.doctype == {"article", "eprint"}
        assert spec.property == {"refereed"}
        assert spec.operator == "citations"
    
    def test_valid_operators(self):
        """All valid operators are accepted."""
        for op in OPERATORS:
            spec = IntentSpec(operator=op)
            assert spec.operator == op
    
    def test_invalid_operator_raises(self):
        """Invalid operator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid operator"):
            IntentSpec(operator="invalid_operator")
    
    def test_has_constraints_empty(self):
        """has_constraints returns False when no constraints set."""
        spec = IntentSpec()
        assert spec.has_constraints() is False
    
    def test_has_constraints_with_doctype(self):
        """has_constraints returns True when doctype set."""
        spec = IntentSpec(doctype={"article"})
        assert spec.has_constraints() is True
    
    def test_has_constraints_with_property(self):
        """has_constraints returns True when property set."""
        spec = IntentSpec(property={"refereed"})
        assert spec.has_constraints() is True
    
    def test_has_content_empty(self):
        """has_content returns False for empty spec."""
        spec = IntentSpec()
        assert spec.has_content() is False
    
    def test_has_content_with_terms(self):
        """has_content returns True with free text terms."""
        spec = IntentSpec(free_text_terms=["exoplanets"])
        assert spec.has_content() is True
    
    def test_has_content_with_authors(self):
        """has_content returns True with authors."""
        spec = IntentSpec(authors=["Hawking, S"])
        assert spec.has_content() is True
    
    def test_serialize_to_dict(self):
        """IntentSpec serializes to dict correctly."""
        spec = IntentSpec(
            free_text_terms=["black holes"],
            doctype={"article"},
            property={"refereed", "openaccess"},
            operator="citations",
        )
        d = spec.to_dict()
        
        assert d["free_text_terms"] == ["black holes"]
        assert d["doctype"] == ["article"]  # sorted list
        assert d["property"] == ["openaccess", "refereed"]  # sorted list
        assert d["operator"] == "citations"
    
    def test_serialize_to_json(self):
        """IntentSpec serializes to valid JSON."""
        spec = IntentSpec(
            free_text_terms=["quasars"],
            year_from=2015,
        )
        json_str = spec.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["free_text_terms"] == ["quasars"]
        assert parsed["year_from"] == 2015
    
    def test_deserialize_from_dict(self):
        """IntentSpec deserializes from dict correctly."""
        d = {
            "free_text_terms": ["neutron stars"],
            "authors": [],
            "affiliations": [],
            "objects": [],
            "year_from": None,
            "year_to": None,
            "doctype": ["eprint"],
            "property": ["refereed"],
            "database": [],
            "bibgroup": [],
            "esources": [],
            "data": [],
            "operator": None,
            "operator_target": None,
            "raw_user_text": "test",
            "confidence": {},
        }
        spec = IntentSpec.from_dict(d)
        
        assert spec.free_text_terms == ["neutron stars"]
        assert spec.doctype == {"eprint"}
        assert spec.property == {"refereed"}
    
    def test_deserialize_from_json(self):
        """IntentSpec deserializes from JSON correctly."""
        json_str = '{"free_text_terms": ["pulsars"], "authors": [], "affiliations": [], "objects": [], "year_from": null, "year_to": null, "doctype": [], "property": [], "database": [], "bibgroup": [], "esources": [], "data": [], "operator": null, "operator_target": null, "raw_user_text": "", "confidence": {}}'
        spec = IntentSpec.from_json(json_str)
        
        assert spec.free_text_terms == ["pulsars"]
    
    def test_roundtrip_serialization(self):
        """IntentSpec survives JSON roundtrip."""
        original = IntentSpec(
            free_text_terms=["gravitational waves"],
            authors=["Einstein, A"],
            year_from=2015,
            year_to=2023,
            doctype={"article"},
            property={"refereed", "openaccess"},
            bibgroup={"LIGO"},
            operator="citations",
            raw_user_text="gravitational waves papers",
            confidence={"operator": 0.9},
        )
        
        json_str = original.to_json()
        restored = IntentSpec.from_json(json_str)
        
        assert restored.free_text_terms == original.free_text_terms
        assert restored.authors == original.authors
        assert restored.year_from == original.year_from
        assert restored.year_to == original.year_to
        assert restored.doctype == original.doctype
        assert restored.property == original.property
        assert restored.bibgroup == original.bibgroup
        assert restored.operator == original.operator
        assert restored.raw_user_text == original.raw_user_text
        assert restored.confidence == original.confidence
    
    def test_repr(self):
        """IntentSpec has compact repr."""
        spec = IntentSpec(
            free_text_terms=["exoplanets"],
            authors=["Kepler, J"],
            operator="citations",
        )
        r = repr(spec)
        assert "IntentSpec" in r
        assert "exoplanets" in r
        assert "citations" in r


class TestPipeline:
    """Tests for pipeline.process_query function."""
    
    def test_process_query_returns_result(self):
        """process_query returns a PipelineResult."""
        result = process_query("papers about dark matter")
        
        assert isinstance(result, PipelineResult)
        assert result.success is True
        assert result.error is None
    
    def test_result_has_required_keys(self):
        """PipelineResult has all required fields."""
        result = process_query("exoplanet atmospheres")
        
        assert hasattr(result, "intent")
        assert hasattr(result, "retrieved_examples")
        assert hasattr(result, "final_query")
        assert hasattr(result, "debug_info")
        assert hasattr(result, "success")
    
    def test_result_intent_is_intentspec(self):
        """PipelineResult.intent is an IntentSpec."""
        result = process_query("neutron stars")
        
        assert isinstance(result.intent, IntentSpec)
        assert result.intent.raw_user_text == "neutron stars"
    
    def test_result_retrieved_examples_is_list(self):
        """PipelineResult.retrieved_examples is a list."""
        result = process_query("quasars")
        
        assert isinstance(result.retrieved_examples, list)
    
    def test_result_final_query_is_string(self):
        """PipelineResult.final_query is a string."""
        result = process_query("black holes")
        
        assert isinstance(result.final_query, str)
        assert len(result.final_query) > 0
    
    def test_result_debug_info_has_timing(self):
        """DebugInfo has timing information."""
        result = process_query("gamma ray bursts")
        
        assert isinstance(result.debug_info, DebugInfo)
        assert result.debug_info.ner_time_ms >= 0
        assert result.debug_info.retrieval_time_ms >= 0
        assert result.debug_info.assembly_time_ms >= 0
        assert result.debug_info.total_time_ms >= 0
    
    def test_result_serializes_to_dict(self):
        """PipelineResult serializes to dict."""
        result = process_query("stellar evolution")
        d = result.to_dict()
        
        assert "intent" in d
        assert "retrieved_examples" in d
        assert "final_query" in d
        assert "debug_info" in d
        assert "success" in d
    
    def test_result_serializes_to_json(self):
        """PipelineResult serializes to valid JSON."""
        result = process_query("supermassive black holes")
        json_str = result.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["success"] is True
        assert "final_query" in parsed
    
    def test_stable_keys_for_same_input(self):
        """Pipeline returns stable keys for identical inputs."""
        result1 = process_query("JWST observations")
        result2 = process_query("JWST observations")
        
        d1 = result1.to_dict()
        d2 = result2.to_dict()
        
        assert d1.keys() == d2.keys()
        assert d1["intent"].keys() == d2["intent"].keys()
        assert d1["debug_info"].keys() == d2["debug_info"].keys()
    
    def test_different_inputs_produce_different_queries(self):
        """Pipeline produces different queries for different inputs."""
        result1 = process_query("exoplanets")
        result2 = process_query("black holes")
        
        # Skeleton just uses input as topic, so queries should differ
        assert result1.final_query != result2.final_query


class TestIsAdsQuery:
    """Tests for is_ads_query helper function."""
    
    def test_natural_language_not_ads(self):
        """Natural language is not detected as ADS query."""
        assert is_ads_query("papers about exoplanets") is False
        assert is_ads_query("show me recent astronomy papers") is False
    
    def test_author_field_is_ads(self):
        """author: field is detected as ADS query."""
        assert is_ads_query('author:"Hawking, S"') is True
    
    def test_abs_field_is_ads(self):
        """abs: field is detected as ADS query."""
        assert is_ads_query('abs:"black holes"') is True
    
    def test_pubdate_field_is_ads(self):
        """pubdate: field is detected as ADS query."""
        assert is_ads_query("pubdate:[2020 TO 2024]") is True
    
    def test_citations_operator_is_ads(self):
        """citations() operator is detected as ADS query."""
        assert is_ads_query('citations(abs:"cosmology")') is True
    
    def test_mixed_content(self):
        """Mixed content with ADS syntax is detected."""
        assert is_ads_query("papers author:Einstein") is True


class TestGoldExample:
    """Tests for GoldExample dataclass."""
    
    def test_instantiate(self):
        """GoldExample can be instantiated."""
        ex = GoldExample(
            nl_query="papers about exoplanets",
            ads_query='abs:"exoplanets"',
            features={"has_topic": True},
            score=0.85,
        )
        
        assert ex.nl_query == "papers about exoplanets"
        assert ex.ads_query == 'abs:"exoplanets"'
        assert ex.features == {"has_topic": True}
        assert ex.score == 0.85
    
    def test_to_dict(self):
        """GoldExample serializes to dict."""
        ex = GoldExample(nl_query="test", ads_query="abs:test")
        d = ex.to_dict()
        
        assert d["nl_query"] == "test"
        assert d["ads_query"] == "abs:test"


class TestDebugInfo:
    """Tests for DebugInfo dataclass."""
    
    def test_instantiate_default(self):
        """DebugInfo can be instantiated with defaults."""
        info = DebugInfo()
        
        assert info.ner_time_ms == 0.0
        assert info.retrieval_time_ms == 0.0
        assert info.assembly_time_ms == 0.0
        assert info.total_time_ms == 0.0
        assert info.constraint_corrections == []
        assert info.fallback_reason is None
    
    def test_to_dict(self):
        """DebugInfo serializes to dict."""
        info = DebugInfo(ner_time_ms=5.0, total_time_ms=10.0)
        d = info.to_dict()
        
        assert d["ner_time_ms"] == 5.0
        assert d["total_time_ms"] == 10.0
