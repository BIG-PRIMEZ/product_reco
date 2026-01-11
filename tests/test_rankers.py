"""
Unit tests for ranking strategies.
"""

import pytest
from services.rankers import (
    SimilarityRanker,
    PopularityRanker,
    HybridRanker,
    get_ranker
)


class TestSimilarityRanker:
    """Test cases for SimilarityRanker."""

    @pytest.fixture
    def ranker(self):
        """Create SimilarityRanker instance."""
        return SimilarityRanker()

    @pytest.fixture
    def candidates(self):
        """Sample candidates for ranking."""
        return [
            {'stock_code': 'A', 'description': 'Product A', 'similarity_score': 0.9},
            {'stock_code': 'B', 'description': 'Product B', 'similarity_score': 0.7},
            {'stock_code': 'C', 'description': 'Product C', 'similarity_score': 0.85}
        ]

    def test_rank_by_similarity(self, ranker, candidates):
        """Test ranking by similarity score."""
        ranked = ranker.rank(candidates, {})

        # Should be sorted by similarity descending
        assert ranked[0]['similarity_score'] == 0.9
        assert ranked[1]['similarity_score'] == 0.85
        assert ranked[2]['similarity_score'] == 0.7

    def test_handle_missing_scores(self, ranker):
        """Test handling candidates with missing similarity scores."""
        candidates = [
            {'stock_code': 'A', 'description': 'Product A'},
            {'stock_code': 'B', 'description': 'Product B', 'similarity_score': 0.8}
        ]

        ranked = ranker.rank(candidates, {})

        # Should only include candidates with scores
        assert len(ranked) == 1


class TestPopularityRanker:
    """Test cases for PopularityRanker."""

    @pytest.fixture
    def ranker(self):
        """Create PopularityRanker instance."""
        return PopularityRanker()

    @pytest.fixture
    def candidates(self):
        """Sample candidates."""
        return [
            {'stock_code': '20726', 'description': 'Product A'},
            {'stock_code': '22112', 'description': 'Product B'},  # Highest popularity
            {'stock_code': 'UNKNOWN', 'description': 'Product C'}
        ]

    def test_rank_by_popularity(self, ranker, candidates):
        """Test ranking by popularity."""
        ranked = ranker.rank(candidates, {})

        # Should add popularity scores
        assert all('popularity_score' in c for c in ranked)

        # Should be sorted by popularity descending
        assert ranked[0]['stock_code'] == '22112'  # Hot water bottle (highest popularity)

    def test_unknown_products_get_zero_score(self, ranker):
        """Test unknown products get zero popularity score."""
        candidates = [{'stock_code': 'UNKNOWN', 'description': 'Unknown Product'}]

        ranked = ranker.rank(candidates, {})

        assert ranked[0]['popularity_score'] == 0.0


class TestHybridRanker:
    """Test cases for HybridRanker."""

    @pytest.fixture
    def ranker(self):
        """Create HybridRanker instance."""
        return HybridRanker()

    @pytest.fixture
    def candidates(self):
        """Sample candidates."""
        return [
            {
                'stock_code': '20726',
                'description': 'RED LUNCH BAG',
                'price': 15.0,
                'similarity_score': 0.8
            },
            {
                'stock_code': '22112',
                'description': 'CHOCOLATE HOT WATER BOTTLE',
                'price': 25.0,
                'similarity_score': 0.6
            }
        ]

    @pytest.fixture
    def processed_query(self):
        """Sample processed query."""
        return {
            'original_query': 'red lunch bag under $20',
            'processed_query': 'red lunch bag under 20',
            'entities': {
                'colors': ['red'],
                'categories': ['lunch bag'],
                'price_max': 20.0
            },
            'intent': 'search'
        }

    def test_hybrid_ranking(self, ranker, candidates, processed_query):
        """Test hybrid ranking combines multiple signals."""
        ranked = ranker.rank(candidates, processed_query)

        # Should add composite scores
        assert all('composite_score' in c for c in ranked)
        assert all('score_breakdown' in c for c in ranked)

        # First product should rank higher (better entity match and price)
        assert ranked[0]['description'] == 'RED LUNCH BAG'

    def test_entity_matching(self, ranker):
        """Test entity matching score calculation."""
        candidate = {
            'description': 'RED LUNCH BAG POLKA DOT',
            'price': 15.0
        }

        processed_query = {
            'entities': {
                'colors': ['red'],
                'categories': ['lunch bag']
            }
        }

        score = ranker._compute_entity_match(candidate, processed_query)

        # Should have high entity match (red + lunch bag both match)
        assert score > 0.5

    def test_price_score_under_budget(self, ranker):
        """Test price scoring for budget constraints."""
        candidate = {'price': 15.0}

        processed_query = {
            'entities': {'price_max': 20.0}
        }

        score = ranker._compute_price_score(candidate, processed_query)

        # Should have good price score (under budget)
        assert score > 0.5

    def test_price_score_over_budget(self, ranker):
        """Test price scoring when over budget."""
        candidate = {'price': 30.0}

        processed_query = {
            'entities': {'price_max': 20.0}
        }

        score = ranker._compute_price_score(candidate, processed_query)

        # Should have lower price score (over budget)
        assert score < 0.5


class TestRankerFactory:
    """Test ranker factory function."""

    def test_get_similarity_ranker(self):
        """Test getting similarity ranker."""
        ranker = get_ranker('similarity')

        assert isinstance(ranker, SimilarityRanker)

    def test_get_popularity_ranker(self):
        """Test getting popularity ranker."""
        ranker = get_ranker('popularity')

        assert isinstance(ranker, PopularityRanker)

    def test_get_hybrid_ranker(self):
        """Test getting hybrid ranker."""
        ranker = get_ranker('hybrid')

        assert isinstance(ranker, HybridRanker)

    def test_get_unknown_ranker_defaults_to_hybrid(self):
        """Test unknown strategy defaults to hybrid."""
        ranker = get_ranker('unknown_strategy')

        assert isinstance(ranker, HybridRanker)
