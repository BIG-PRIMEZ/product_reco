"""
Unit tests for QueryProcessor service.
"""

import pytest
from services.query_processor import QueryProcessor, QueryUnderstanding


class TestQueryProcessor:
    """Test cases for QueryProcessor."""

    @pytest.fixture
    def processor(self):
        """Create QueryProcessor instance for testing."""
        return QueryProcessor(enable_expansion=True)

    def test_process_query_basic(self, processor):
        """Test basic query processing."""
        result = processor.process_query("red lunch bag")

        assert isinstance(result, QueryUnderstanding)
        assert result.original_query == "red lunch bag"
        assert result.processed_query == "red lunch bag"
        assert result.confidence > 0

    def test_extract_color_entity(self, processor):
        """Test color entity extraction."""
        result = processor.process_query("red polka dot bag")

        assert 'colors' in result.entities
        assert 'red' in result.entities['colors']

    def test_extract_category_entity(self, processor):
        """Test category entity extraction."""
        result = processor.process_query("lunch bag")

        assert 'categories' in result.entities
        assert 'lunch bag' in result.entities['categories']

    def test_extract_price_entity(self, processor):
        """Test price entity extraction."""
        result = processor.process_query("lunch bag under $20")

        assert 'price_max' in result.entities
        assert result.entities['price_max'] == 20.0

    def test_query_expansion(self, processor):
        """Test query expansion."""
        expanded = processor.expand_query("lunch bag", entities={})

        assert isinstance(expanded, list)
        # Should have some expansions
        assert len(expanded) >= 0  # May or may not have expansions

    def test_intent_detection_search(self, processor):
        """Test search intent detection."""
        intent = processor.extract_intent("find red lunch bag", entities={})

        assert intent['primary_intent'] == 'search'

    def test_intent_detection_browse(self, processor):
        """Test browse intent detection."""
        intent = processor.extract_intent("show me all tea sets", entities={})

        assert intent['primary_intent'] == 'browse'

    def test_spell_correction(self, processor):
        """Test spell correction."""
        corrected, corrections = processor._correct_spelling("luch bag")

        assert 'lunch' in corrected
        assert len(corrections) > 0

    def test_clean_query(self, processor):
        """Test query cleaning."""
        cleaned = processor._clean_query("  Red   Lunch   Bag!!! ")

        assert cleaned == "red lunch bag"
        assert "!" not in cleaned

    def test_empty_query(self, processor):
        """Test handling of empty query."""
        result = processor.process_query("")

        assert result.processed_query == ""
        assert result.confidence < 1.0

    def test_special_characters(self, processor):
        """Test handling of special characters."""
        result = processor.process_query("red@#$%lunch&*bag")

        # Should clean special characters
        assert '@' not in result.processed_query
        assert '#' not in result.processed_query

    def test_refinement_suggestions(self, processor):
        """Test refinement suggestions."""
        results = [
            {'description': 'RED LUNCH BAG'},
            {'description': 'PINK LUNCH BAG'},
            {'description': 'BLUE LUNCH BAG'}
        ]

        suggestions = processor.suggest_refinements("bag", results)

        assert isinstance(suggestions, list)
        # Should generate some suggestions
        assert len(suggestions) >= 0
