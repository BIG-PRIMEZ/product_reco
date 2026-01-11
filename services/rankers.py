"""
Ranking strategies for product recommendations.

Provides multiple ranking algorithms:
- SimilarityRanker: Pure vector similarity
- PopularityRanker: Product popularity/sales
- HybridRanker: Weighted combination of multiple signals
"""

from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import numpy as np
from utils.logger import get_logger
from config import get_ranking_config

logger = get_logger(__name__)


class BaseRanker(ABC):
    """Base class for ranking strategies."""

    @abstractmethod
    def rank(self, candidates: List[Dict], processed_query: Dict) -> List[Dict]:
        """
        Rank candidates and return sorted list.

        Args:
            candidates: List of product candidates with metadata
            processed_query: Processed query with entities and intent

        Returns:
            Sorted list of candidates (highest ranked first)
        """
        pass


class SimilarityRanker(BaseRanker):
    """Rank purely by vector similarity score."""

    def __init__(self):
        """Initialize similarity ranker."""
        logger.info("SimilarityRanker initialized")

    def rank(self, candidates: List[Dict], processed_query: Dict) -> List[Dict]:
        """
        Rank by similarity score (cosine similarity).

        Args:
            candidates: List of candidates with 'similarity_score' field
            processed_query: Processed query (not used in this ranker)

        Returns:
            Candidates sorted by similarity score (descending)
        """
        try:
            logger.debug(f"Ranking {len(candidates)} candidates by similarity")

            # Ensure all candidates have similarity scores
            scored_candidates = [
                c for c in candidates
                if 'similarity_score' in c
            ]

            if len(scored_candidates) < len(candidates):
                logger.warning(
                    f"{len(candidates) - len(scored_candidates)} candidates missing similarity scores"
                )

            # Sort by similarity score
            ranked = sorted(
                scored_candidates,
                key=lambda x: x.get('similarity_score', 0),
                reverse=True
            )

            logger.debug(
                f"Ranked {len(ranked)} candidates",
                extra={'top_score': ranked[0]['similarity_score'] if ranked else 0}
            )

            return ranked

        except Exception as e:
            logger.error(f"Error in SimilarityRanker: {str(e)}", exc_info=True)
            return candidates  # Return unranked on error


class PopularityRanker(BaseRanker):
    """Rank by product popularity (sales, views, etc.)."""

    def __init__(self, popularity_weights: Optional[Dict[str, float]] = None):
        """
        Initialize popularity ranker.

        Args:
            popularity_weights: Dictionary mapping stock_code to popularity score
                               If None, uses mock data based on stock codes
        """
        self.weights = popularity_weights or self._load_mock_popularity()
        logger.info(
            "PopularityRanker initialized",
            extra={'num_products': len(self.weights)}
        )

    def rank(self, candidates: List[Dict], processed_query: Dict) -> List[Dict]:
        """
        Rank by popularity score.

        Args:
            candidates: List of candidates
            processed_query: Processed query (not used)

        Returns:
            Candidates sorted by popularity score (descending)
        """
        try:
            logger.debug(f"Ranking {len(candidates)} candidates by popularity")

            # Add popularity scores
            for candidate in candidates:
                stock_code = candidate.get('stock_code', '')
                candidate['popularity_score'] = self.weights.get(stock_code, 0.0)

            # Sort by popularity
            ranked = sorted(
                candidates,
                key=lambda x: x.get('popularity_score', 0),
                reverse=True
            )

            logger.debug(
                f"Ranked {len(ranked)} candidates by popularity",
                extra={'top_score': ranked[0]['popularity_score'] if ranked else 0}
            )

            return ranked

        except Exception as e:
            logger.error(f"Error in PopularityRanker: {str(e)}", exc_info=True)
            return candidates

    def _load_mock_popularity(self) -> Dict[str, float]:
        """
        Load mock popularity data.

        In production, this would load from database/analytics.
        """
        # Mock popularity scores (normalized 0-1)
        # Higher scores = more popular products
        return {
            '20726': 0.85,  # LUNCH BAG WOODLAND
            '21034': 0.75,  # REX CASH+CARRY JUMBO SHOPPER
            '21931': 0.65,  # JUMBO STORAGE BAG SUKI
            '22077': 0.55,  # 6 RIBBONS RUSTIC CHARM
            '22112': 0.90,  # CHOCOLATE HOT WATER BOTTLE (very popular)
            '22139': 0.80,  # RETROSPOT TEA SET CERAMIC 11 PC
            '22384': 0.70,  # LUNCH BAG PINK POLKADOT
            '22423': 0.60,  # REGENCY CAKESTAND 3 TIER
            '22727': 0.75,  # ALARM CLOCK BAKELIKE RED
            '23298': 0.50,  # SPOTTY BUNTING
        }


class HybridRanker(BaseRanker):
    """
    Combine multiple ranking signals.

    Composite score = α·similarity + β·popularity + γ·entity_match + δ·price_preference
    """

    def __init__(
        self,
        similarity_weight: Optional[float] = None,
        popularity_weight: Optional[float] = None,
        entity_match_weight: Optional[float] = None,
        price_weight: Optional[float] = None,
        popularity_data: Optional[Dict] = None
    ):
        """
        Initialize hybrid ranker.

        Args:
            similarity_weight: Weight for similarity score (default from config)
            popularity_weight: Weight for popularity score (default from config)
            entity_match_weight: Weight for entity matching (default from config)
            price_weight: Weight for price preference (default from config)
            popularity_data: Popularity weights dictionary
        """
        config = get_ranking_config()

        self.weights = {
            'similarity': similarity_weight or config.similarity_weight,
            'popularity': popularity_weight or config.popularity_weight,
            'entity_match': entity_match_weight or config.entity_match_weight,
            'price': price_weight or config.price_weight
        }

        # Validate weights sum to ~1.0
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):
            logger.warning(f"Ranking weights sum to {total}, normalizing to 1.0")
            # Normalize
            for key in self.weights:
                self.weights[key] /= total

        # Load popularity data
        self.popularity_ranker = PopularityRanker(popularity_data)

        logger.info(
            "HybridRanker initialized",
            extra={'weights': self.weights}
        )

    def rank(self, candidates: List[Dict], processed_query: Dict) -> List[Dict]:
        """
        Rank using hybrid scoring.

        Args:
            candidates: List of candidates
            processed_query: Processed query with entities and intent

        Returns:
            Candidates sorted by composite score (descending)
        """
        try:
            logger.debug(
                f"Ranking {len(candidates)} candidates with hybrid strategy",
                extra={'weights': self.weights}
            )

            # Add popularity scores
            for candidate in candidates:
                stock_code = candidate.get('stock_code', '')
                candidate['popularity_score'] = self.popularity_ranker.weights.get(stock_code, 0.0)

            # Compute composite scores
            for candidate in candidates:
                scores = {
                    'similarity': self._normalize(candidate.get('similarity_score', 0)),
                    'popularity': self._normalize(candidate.get('popularity_score', 0)),
                    'entity_match': self._compute_entity_match(candidate, processed_query),
                    'price': self._compute_price_score(candidate, processed_query)
                }

                # Weighted sum
                candidate['composite_score'] = sum(
                    scores[signal] * self.weights[signal]
                    for signal in scores
                )

                # Store individual scores for debugging
                candidate['score_breakdown'] = scores

            # Sort by composite score
            ranked = sorted(
                candidates,
                key=lambda x: x.get('composite_score', 0),
                reverse=True
            )

            if ranked:
                logger.debug(
                    f"Top candidate score: {ranked[0]['composite_score']:.4f}",
                    extra={
                        'top_product': ranked[0].get('description', ''),
                        'score_breakdown': ranked[0].get('score_breakdown', {})
                    }
                )

            return ranked

        except Exception as e:
            logger.error(f"Error in HybridRanker: {str(e)}", exc_info=True)
            # Fall back to similarity ranking
            return sorted(
                candidates,
                key=lambda x: x.get('similarity_score', 0),
                reverse=True
            )

    def _normalize(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize score to [0, 1] range."""
        if score < min_val:
            return 0.0
        if score > max_val:
            return 1.0
        return (score - min_val) / (max_val - min_val) if max_val > min_val else score

    def _compute_entity_match(self, candidate: Dict, processed_query: Dict) -> float:
        """
        Score based on entity matching.

        Example:
            Query entities: {colors: ['red'], categories: ['lunch bag']}
            Product: "RED LUNCH BAG POLKADOT"
            → High entity match score

        Args:
            candidate: Product candidate
            processed_query: Processed query with entities

        Returns:
            Entity match score (0-1)
        """
        try:
            entities = processed_query.get('entities', {})
            if not entities:
                return 0.5  # Neutral score if no entities

            description = candidate.get('description', '').lower()
            matches = 0
            total = 0

            # Check color matches
            for color in entities.get('colors', []):
                total += 1
                if color.lower() in description:
                    matches += 1

            # Check category matches
            for category in entities.get('categories', []):
                total += 1
                # More lenient matching for categories
                category_words = category.lower().split()
                if all(word in description for word in category_words):
                    matches += 2  # Category matches are more important
                    total += 1  # But weight is still 1

            # Check pattern matches
            for pattern in entities.get('patterns', []):
                total += 1
                if pattern.lower() in description:
                    matches += 1

            # Check material matches
            for material in entities.get('materials', []):
                total += 1
                if material.lower() in description:
                    matches += 1

            # Calculate match ratio
            if total == 0:
                return 0.5  # Neutral if no entities to match

            match_ratio = matches / total
            return min(match_ratio, 1.0)

        except Exception as e:
            logger.error(f"Error computing entity match: {str(e)}", exc_info=True)
            return 0.5

    def _compute_price_score(self, candidate: Dict, processed_query: Dict) -> float:
        """
        Score based on price preferences.

        If query mentions "cheap" or "affordable", boost low-priced items.
        If query mentions "premium" or "quality", boost higher-priced items.
        If query specifies price range, score based on proximity.

        Args:
            candidate: Product candidate with 'price' field
            processed_query: Processed query with entities and intent

        Returns:
            Price preference score (0-1)
        """
        try:
            price = candidate.get('price', 0)
            if price <= 0:
                return 0.0  # Invalid price

            entities = processed_query.get('entities', {})

            # Check for explicit price constraints
            if 'price_max' in entities:
                max_price = entities['price_max']
                if price <= max_price:
                    # Prefer items well under budget
                    return 1.0 - (price / max_price) * 0.5
                else:
                    # Penalize items over budget more heavily
                    over_budget_ratio = (price - max_price) / max_price
                    return max(0.0, 0.5 - over_budget_ratio)

            if 'price_min' in entities:
                min_price = entities['price_min']
                if price >= min_price:
                    # Items above minimum are acceptable
                    return 0.7
                else:
                    # Penalize items below minimum
                    return 0.3

            if 'price_target' in entities:
                target = entities['price_target']
                # Score based on proximity to target
                diff = abs(price - target)
                return max(0.0, 1.0 - (diff / target))

            # Check for price intent from modifiers
            intent = processed_query.get('intent', 'search')
            modifiers = []  # Would come from intent detection

            if 'budget' in modifiers or 'cheap' in processed_query.get('original_query', '').lower():
                # Prefer lower prices (inverse relationship)
                return 1.0 / (1 + price / 10)

            if 'premium' in modifiers or 'quality' in processed_query.get('original_query', '').lower():
                # Prefer moderate-to-high prices
                return min(price / 50, 1.0)

            # Neutral - slight preference for moderate prices
            if 5 <= price <= 50:
                return 0.7
            else:
                return 0.5

        except Exception as e:
            logger.error(f"Error computing price score: {str(e)}", exc_info=True)
            return 0.5


class MMRRanker(BaseRanker):
    """
    Maximal Marginal Relevance (MMR) ranker for result diversification.

    Balances relevance and diversity to avoid redundant results.
    """

    def __init__(self, lambda_param: float = 0.5):
        """
        Initialize MMR ranker.

        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
                         Default 0.5 for balanced approach
        """
        self.lambda_param = lambda_param
        logger.info(
            "MMRRanker initialized",
            extra={'lambda': lambda_param}
        )

    def rank(self, candidates: List[Dict], processed_query: Dict) -> List[Dict]:
        """
        Rank using MMR for diversification.

        Args:
            candidates: List of candidates
            processed_query: Processed query

        Returns:
            Diversified ranking
        """
        try:
            logger.debug(f"Applying MMR diversification to {len(candidates)} candidates")

            if len(candidates) <= 1:
                return candidates

            # Start with highest similarity candidate
            ranked = sorted(candidates, key=lambda x: x.get('similarity_score', 0), reverse=True)
            selected = [ranked[0]]
            remaining = ranked[1:]

            # Iteratively select candidates balancing relevance and diversity
            while remaining:
                mmr_scores = []

                for candidate in remaining:
                    relevance = candidate.get('similarity_score', 0)

                    # Calculate max similarity to already selected items
                    max_sim = max(
                        self._compute_similarity(candidate, selected_item)
                        for selected_item in selected
                    )

                    # MMR score: balance relevance and novelty
                    mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                    mmr_scores.append((candidate, mmr_score))

                # Select candidate with highest MMR score
                if mmr_scores:
                    best_candidate, _ = max(mmr_scores, key=lambda x: x[1])
                    selected.append(best_candidate)
                    remaining.remove(best_candidate)
                else:
                    break

            logger.debug(f"MMR diversification complete, selected {len(selected)} items")

            return selected

        except Exception as e:
            logger.error(f"Error in MMR ranker: {str(e)}", exc_info=True)
            return candidates

    def _compute_similarity(self, item1: Dict, item2: Dict) -> float:
        """
        Compute similarity between two items.

        Uses description overlap as a simple similarity measure.
        In production, could use embedding similarity.
        """
        try:
            desc1 = set(item1.get('description', '').lower().split())
            desc2 = set(item2.get('description', '').lower().split())

            if not desc1 or not desc2:
                return 0.0

            # Jaccard similarity
            intersection = len(desc1 & desc2)
            union = len(desc1 | desc2)

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}", exc_info=True)
            return 0.0


def get_ranker(strategy: str = 'hybrid', **kwargs) -> BaseRanker:
    """
    Factory function to get a ranker instance.

    Args:
        strategy: Ranking strategy ('similarity', 'popularity', 'hybrid', 'mmr')
        **kwargs: Additional arguments for ranker initialization

    Returns:
        Ranker instance

    Example:
        >>> ranker = get_ranker('hybrid', similarity_weight=0.6)
        >>> ranked = ranker.rank(candidates, processed_query)
    """
    rankers = {
        'similarity': SimilarityRanker,
        'popularity': PopularityRanker,
        'hybrid': HybridRanker,
        'mmr': MMRRanker
    }

    if strategy not in rankers:
        logger.warning(
            f"Unknown ranking strategy '{strategy}', defaulting to 'hybrid'",
            extra={'requested_strategy': strategy}
        )
        strategy = 'hybrid'

    logger.info(f"Creating {strategy} ranker", extra={'strategy': strategy})

    return rankers[strategy](**kwargs)
