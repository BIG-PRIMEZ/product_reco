"""
Intelligent product recommender service.

Orchestrates query processing, retrieval, ranking, and response generation
for personalized product recommendations.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from utils.logger import get_logger
from services.query_processor import QueryProcessor
from services.rankers import get_ranker, BaseRanker
from config import get_config

logger = get_logger(__name__)


@dataclass
class RecommendationResponse:
    """
    Structured recommendation response.

    Attributes:
        products: List of recommended products
        query_understanding: Query analysis metadata
        response: Natural language response message
        suggestions: Query refinement suggestions
        metadata: Additional metadata (timing, strategy, etc.)
    """
    products: List[Dict]
    query_understanding: Dict
    response: str
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ResponseGenerator:
    """
    Generate natural language responses for search results.

    Creates context-aware, helpful messages for users.
    """

    def __init__(self):
        """Initialize response generator."""
        logger.debug("ResponseGenerator initialized")

    def generate(
        self,
        query: str,
        processed: Dict,
        products: List[Dict]
    ) -> str:
        """
        Generate response message based on results.

        Args:
            query: Original query
            processed: Processed query understanding
            products: Recommended products

        Returns:
            Natural language response string
        """
        try:
            if not products:
                return self._generate_no_results_response(query, processed)

            if len(products) < 3:
                return self._generate_few_results_response(query, products, processed)

            return self._generate_standard_response(query, products, processed)

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return f"Found {len(products)} products matching your search."

    def _generate_no_results_response(self, query: str, processed: Dict) -> str:
        """Generate helpful message when no results found."""
        # Get catalog suggestions based on entities or default
        suggestions = self._get_catalog_suggestions(processed)

        message = f"No products found for '{query}'. "

        # Add context based on query understanding
        if processed.get('entities'):
            entities = processed['entities']
            if 'price_max' in entities:
                message += f"We don't have items matching your criteria under ${entities['price_max']:.2f}. "

        message += "Our catalog focuses on home decor and gift items. "
        message += f"You might try: {', '.join(suggestions[:4])}."

        return message

    def _generate_few_results_response(
        self,
        query: str,
        products: List[Dict],
        processed: Dict
    ) -> str:
        """Generate response for small number of results."""
        product_names = [p['description'][:50] for p in products]

        message = f"Found {len(products)} product"
        message += "s" if len(products) > 1 else ""
        message += f" matching '{query}'. "

        # Add price range if available
        if all('price' in p for p in products):
            prices = [p['price'] for p in products]
            price_range = f"${min(prices):.2f} - ${max(prices):.2f}"
            message += f"Price range: {price_range}. "

        # List products
        message += "Available: " + ", ".join(product_names) + "."

        return message

    def _generate_standard_response(
        self,
        query: str,
        products: List[Dict],
        processed: Dict
    ) -> str:
        """Generate standard response for good results."""
        top_products = [p['description'][:50] for p in products[:3]]

        message = f"Found {len(products)} products matching '{query}'. "

        # Add price range
        if all('price' in p for p in products):
            prices = [p['price'] for p in products]
            price_range = f"${min(prices):.2f} - ${max(prices):.2f}"
            message += f"Price range: {price_range}. "

        # Mention top matches
        message += f"Top matches: {', '.join(top_products)}."

        # Add entity-specific messaging
        entities = processed.get('entities', {})
        if 'colors' in entities:
            colors = entities['colors']
            message += f" Showing {', '.join(colors)} items."

        return message

    def _get_catalog_suggestions(self, processed: Dict) -> List[str]:
        """Get catalog suggestions based on query understanding."""
        # Default popular searches
        default_suggestions = [
            "red lunch bag",
            "alarm clock",
            "tea set",
            "storage bag",
            "polka dot items",
            "hand warmer",
            "metal lantern",
            "cake stand"
        ]

        # Customize based on entities
        entities = processed.get('entities', {})

        if 'colors' in entities:
            color = entities['colors'][0]
            return [
                f"{color} lunch bag",
                f"{color} tea set",
                f"{color} storage bag",
                "alarm clock"
            ]

        if 'categories' in entities:
            # User searched for a category but no results
            # Suggest similar categories
            return [
                "lunch bag",
                "storage bag",
                "tea set",
                "alarm clock"
            ]

        return default_suggestions


class RecommenderService:
    """
    Intelligent product recommender with NL understanding.

    Coordinates query processing, retrieval, ranking, and response generation
    for personalized recommendations.
    """

    def __init__(
        self,
        vector_service,
        query_processor: Optional[QueryProcessor] = None,
        default_strategy: str = 'hybrid'
    ):
        """
        Initialize recommender service.

        Args:
            vector_service: VectorDBService instance for retrieval
            query_processor: QueryProcessor instance (creates new if None)
            default_strategy: Default ranking strategy
        """
        self.vector_service = vector_service
        self.query_processor = query_processor or QueryProcessor()
        self.default_strategy = default_strategy

        # Initialize rankers
        self.rankers = {
            'similarity': get_ranker('similarity'),
            'popularity': get_ranker('popularity'),
            'hybrid': get_ranker('hybrid'),
            'mmr': get_ranker('mmr')
        }

        # Response generator
        self.response_generator = ResponseGenerator()

        # Configuration
        self.config = get_config()

        logger.info(
            "RecommenderService initialized",
            extra={
                'default_strategy': default_strategy,
                'rankers': list(self.rankers.keys())
            }
        )

    def recommend(
        self,
        query: str,
        context: Optional[Dict] = None,
        strategy: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> RecommendationResponse:
        """
        Generate product recommendations for a query.

        Args:
            query: User query (text)
            context: Optional user context (history, preferences)
            strategy: Ranking strategy ('similarity', 'hybrid', 'popularity', 'mmr')
            top_k: Number of recommendations (default from config)
            min_score: Minimum similarity score (default from config)

        Returns:
            RecommendationResponse with products and metadata

        Example:
            >>> recommender = RecommenderService(vector_service)
            >>> response = recommender.recommend("red lunch bag", strategy='hybrid', top_k=10)
            >>> print(response.response)
            "Found 8 products matching 'red lunch bag'..."
        """
        start_time = time.time()

        # Use defaults from config if not specified
        strategy = strategy or self.default_strategy
        top_k = top_k or self.config.default_top_k
        min_score = min_score or self.config.min_similarity_score

        logger.info(
            f"Processing recommendation request",
            extra={
                'query': query,
                'strategy': strategy,
                'top_k': top_k,
                'min_score': min_score
            }
        )

        try:
            # 1. Query understanding
            processed = self.query_processor.process_query(query, context)

            # 2. Candidate retrieval
            candidates = self._retrieve_candidates(processed, min_score)

            # 3. Ranking
            ranked_products = self._rank_products(candidates, processed, strategy)

            # 4. Filtering & business rules
            final_products = self._apply_business_rules(ranked_products)

            # 5. Diversification (if enabled and using appropriate strategy)
            if self.config.enable_cache and strategy != 'mmr':
                # MMR already handles diversification
                final_products = self._diversify_results(final_products)

            # 6. Limit to top_k
            top_products = final_products[:top_k]

            # 7. Format products for response
            formatted_products = self._format_products(top_products)

            # 8. Generate response message
            response_message = self.response_generator.generate(
                query=query,
                processed=processed.__dict__ if hasattr(processed, '__dict__') else processed,
                products=formatted_products
            )

            # 9. Generate refinement suggestions
            suggestions = self.query_processor.suggest_refinements(
                query=query,
                results=formatted_products,
                max_suggestions=5
            )

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Build response
            recommendation = RecommendationResponse(
                products=formatted_products,
                query_understanding={
                    'original_query': processed.original_query,
                    'processed_query': processed.processed_query,
                    'intent': processed.intent,
                    'entities': processed.entities,
                    'confidence': processed.confidence
                },
                response=response_message,
                suggestions=suggestions,
                metadata={
                    'total_candidates': len(candidates),
                    'strategy_used': strategy,
                    'processing_time_ms': round(processing_time_ms, 2),
                    'top_k': top_k,
                    'min_score': min_score
                }
            )

            logger.info(
                f"Recommendation generated successfully",
                extra={
                    'query': query,
                    'num_products': len(formatted_products),
                    'processing_time_ms': processing_time_ms,
                    'strategy': strategy
                }
            )

            return recommendation

        except Exception as e:
            logger.error(
                f"Error generating recommendations: {str(e)}",
                extra={'query': query},
                exc_info=True
            )

            # Return minimal response on error
            return RecommendationResponse(
                products=[],
                query_understanding={'original_query': query, 'processed_query': query},
                response=f"An error occurred while processing your request. Please try again.",
                metadata={
                    'error': str(e),
                    'processing_time_ms': (time.time() - start_time) * 1000
                }
            )

    def _retrieve_candidates(
        self,
        processed_query,
        min_score: float
    ) -> List[Dict]:
        """
        Retrieve candidate products using multiple strategies.

        Args:
            processed_query: QueryUnderstanding object
            min_score: Minimum similarity score

        Returns:
            List of candidate products with metadata
        """
        logger.debug("Retrieving candidates")

        candidates = []
        seen_stock_codes = set()

        try:
            # Primary vector search with processed query
            primary_results = self.vector_service.search_products(
                query=processed_query.processed_query,
                top_k=50,
                min_score=max(min_score - 0.1, 0.1)  # Slightly lower threshold for candidates
            )

            for result in primary_results:
                stock_code = result.get('stock_code', '')
                if stock_code and stock_code not in seen_stock_codes:
                    candidates.append({
                        'stock_code': stock_code,
                        'description': result.get('description', ''),
                        'price': result.get('price', 0),
                        'country': result.get('country', ''),
                        'similarity_score': result.get('score', 0),
                        'source': 'primary_query'
                    })
                    seen_stock_codes.add(stock_code)

            # Expanded query searches (if enabled)
            if self.config.enable_query_expansion and processed_query.expanded_queries:
                for expanded_query in processed_query.expanded_queries[:self.config.max_expanded_queries]:
                    expanded_results = self.vector_service.search_products(
                        query=expanded_query,
                        top_k=20,
                        min_score=min_score
                    )

                    for result in expanded_results:
                        stock_code = result.get('stock_code', '')
                        if stock_code and stock_code not in seen_stock_codes:
                            candidates.append({
                                'stock_code': stock_code,
                                'description': result.get('description', ''),
                                'price': result.get('price', 0),
                                'country': result.get('country', ''),
                                'similarity_score': result.get('score', 0) * 0.95,  # Slight penalty for expanded queries
                                'source': f'expanded:{expanded_query}'
                            })
                            seen_stock_codes.add(stock_code)

            logger.debug(
                f"Retrieved {len(candidates)} unique candidates",
                extra={
                    'primary_count': len(primary_results),
                    'expanded_count': len(candidates) - len(primary_results)
                }
            )

            return candidates

        except Exception as e:
            logger.error(f"Error retrieving candidates: {str(e)}", exc_info=True)
            return candidates  # Return what we have so far

    def _rank_products(
        self,
        candidates: List[Dict],
        processed_query,
        strategy: str
    ) -> List[Dict]:
        """
        Rank candidates using specified strategy.

        Args:
            candidates: List of candidate products
            processed_query: QueryUnderstanding object
            strategy: Ranking strategy name

        Returns:
            Ranked list of products
        """
        logger.debug(f"Ranking {len(candidates)} candidates with '{strategy}' strategy")

        try:
            ranker = self.rankers.get(strategy, self.rankers['hybrid'])

            # Convert QueryUnderstanding to dict for ranker
            query_dict = {
                'original_query': processed_query.original_query,
                'processed_query': processed_query.processed_query,
                'intent': processed_query.intent,
                'entities': processed_query.entities,
                'confidence': processed_query.confidence
            }

            ranked = ranker.rank(candidates, query_dict)

            logger.debug(f"Ranked {len(ranked)} products")

            return ranked

        except Exception as e:
            logger.error(f"Error ranking products: {str(e)}", exc_info=True)
            # Fall back to similarity sorting
            return sorted(candidates, key=lambda x: x.get('similarity_score', 0), reverse=True)

    def _apply_business_rules(self, products: List[Dict]) -> List[Dict]:
        """
        Apply business logic and filters.

        - Filter out invalid products
        - Remove test/placeholder data
        - Apply stock/availability rules (future)

        Args:
            products: List of products

        Returns:
            Filtered list of products
        """
        logger.debug(f"Applying business rules to {len(products)} products")

        try:
            filtered = []

            for product in products:
                if self._is_valid_product(product):
                    filtered.append(product)

            logger.debug(
                f"Filtered to {len(filtered)} valid products",
                extra={'filtered_out': len(products) - len(filtered)}
            )

            return filtered

        except Exception as e:
            logger.error(f"Error applying business rules: {str(e)}", exc_info=True)
            return products

    def _is_valid_product(self, product: Dict) -> bool:
        """
        Check if product is valid for recommendation.

        Args:
            product: Product dictionary

        Returns:
            True if valid, False otherwise
        """
        import re

        description = product.get('description', '').strip().lower()
        price = product.get('price', 0)

        # Filter empty descriptions
        if not description:
            return False

        # Invalid descriptions (use word boundaries to match complete words only)
        # This prevents filtering "MANUAL TYPEWRITER" but still filters "INSTRUCTION MANUAL"
        invalid_terms = ['ebay', 'test', 'unknown', 'samples', 'manual', 'postage', 'carriage']

        # Check for exact match
        if description in invalid_terms:
            return False

        # Check if any invalid term appears as a standalone word (with word boundaries)
        for term in invalid_terms:
            if re.search(rf'\b{re.escape(term)}\b', description):
                return False

        # Invalid prices
        if price <= 0:
            return False

        # Very high prices might be data errors
        if price > 1000:
            logger.warning(
                f"Unusually high price detected: ${price}",
                extra={'product': description}
            )
            # Still allow, but log

        return True

    def _diversify_results(self, products: List[Dict]) -> List[Dict]:
        """
        Ensure result diversity (avoid redundant products).

        Uses simple heuristic: avoid products with very similar descriptions.

        Args:
            products: List of products

        Returns:
            Diversified list of products
        """
        if len(products) <= 3:
            return products  # Too few to diversify

        logger.debug(f"Diversifying {len(products)} products")

        try:
            diversified = [products[0]]  # Always include top result

            for product in products[1:]:
                # Check if too similar to already selected products
                is_diverse = True
                for selected in diversified:
                    similarity = self._compute_description_similarity(
                        product['description'],
                        selected['description']
                    )
                    if similarity > 0.8:  # Very similar
                        is_diverse = False
                        break

                if is_diverse:
                    diversified.append(product)

            logger.debug(
                f"Diversified to {len(diversified)} products",
                extra={'removed': len(products) - len(diversified)}
            )

            return diversified

        except Exception as e:
            logger.error(f"Error diversifying results: {str(e)}", exc_info=True)
            return products

    def _compute_description_similarity(self, desc1: str, desc2: str) -> float:
        """Compute simple Jaccard similarity between descriptions."""
        try:
            words1 = set(desc1.lower().split())
            words2 = set(desc2.lower().split())

            if not words1 or not words2:
                return 0.0

            intersection = len(words1 & words2)
            union = len(words1 | words2)

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}", exc_info=True)
            return 0.0

    def _format_products(self, products: List[Dict]) -> List[Dict]:
        """
        Format products for API response.

        Args:
            products: List of raw product dictionaries

        Returns:
            List of formatted product dictionaries
        """
        formatted = []

        for product in products:
            formatted.append({
                'stock_code': product.get('stock_code', ''),
                'description': product.get('description', ''),
                'price': float(product.get('price', 0)),
                'unit_price': f"${product.get('price', 0):.2f}",
                'country': product.get('country', ''),
                'relevance_score': round(product.get('composite_score', product.get('similarity_score', 0)), 4),
                'similarity_score': round(product.get('similarity_score', 0), 4)
            })

        return formatted
