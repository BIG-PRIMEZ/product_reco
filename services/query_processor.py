"""
Query processing service with NL understanding.

Provides advanced query processing including:
- Query expansion (synonyms, related terms)
- Entity extraction (colors, categories, attributes)
- Intent detection
- Spell correction
- Query refinement suggestions
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryUnderstanding:
    """
    Structured representation of query understanding.

    Attributes:
        original_query: The raw user query
        processed_query: Cleaned and normalized query
        intent: Detected intent ('search', 'browse', 'compare', 'info')
        entities: Extracted entities (colors, categories, etc.)
        expanded_queries: Alternative search terms
        confidence: Confidence score (0-1)
        corrections: Spelling corrections applied
    """
    original_query: str
    processed_query: str
    intent: str = 'search'
    entities: Dict[str, List[str]] = field(default_factory=dict)
    expanded_queries: List[str] = field(default_factory=list)
    confidence: float = 1.0
    corrections: List[Tuple[str, str]] = field(default_factory=list)  # (original, corrected)


class QueryProcessor:
    """
    Advanced natural language query processor for e-commerce search.

    Uses rule-based processing and optional LLM integration for:
    - Semantic expansion
    - Intent detection
    - Entity extraction
    - Contextual refinement
    """

    def __init__(self, llm_client=None, enable_expansion: bool = True):
        """
        Initialize query processor.

        Args:
            llm_client: Optional LLM client for advanced understanding
            enable_expansion: Enable query expansion
        """
        self.llm_client = llm_client
        self.enable_expansion = enable_expansion

        # Load synonym database
        self.synonyms = self._load_synonyms()

        # Entity patterns
        self.color_patterns = self._load_color_patterns()
        self.category_patterns = self._load_category_patterns()
        self.price_patterns = self._compile_price_patterns()

        logger.info("QueryProcessor initialized", extra={
            'llm_enabled': llm_client is not None,
            'expansion_enabled': enable_expansion
        })

    def process_query(self, query: str, context: Optional[Dict] = None) -> QueryUnderstanding:
        """
        Process and enhance a user query.

        Args:
            query: Raw user query string
            context: Optional context (user history, session data)

        Returns:
            QueryUnderstanding object with processed query and metadata

        Example:
            >>> processor = QueryProcessor()
            >>> result = processor.process_query("red lunch bag under $20")
            >>> print(result.entities)
            {'colors': ['red'], 'categories': ['lunch bag'], 'price_max': 20.0}
        """
        logger.debug(f"Processing query: {query}", extra={'query': query})

        try:
            # 1. Clean and normalize query
            cleaned_query = self._clean_query(query)

            # 2. Spell correction (if enabled)
            corrected_query, corrections = self._correct_spelling(cleaned_query)

            # 3. Extract entities
            entities = self.extract_entities(corrected_query)

            # 4. Detect intent
            intent = self.extract_intent(corrected_query, entities)

            # 5. Expand query (if enabled)
            expanded_queries = []
            if self.enable_expansion:
                expanded_queries = self.expand_query(corrected_query, entities)

            # 6. Calculate confidence
            confidence = self._calculate_confidence(corrected_query, entities, corrections)

            result = QueryUnderstanding(
                original_query=query,
                processed_query=corrected_query,
                intent=intent['primary_intent'],
                entities=entities,
                expanded_queries=expanded_queries,
                confidence=confidence,
                corrections=corrections
            )

            logger.info(
                f"Query processed successfully",
                extra={
                    'query': query,
                    'intent': intent['primary_intent'],
                    'entities_count': len(entities),
                    'expanded_count': len(expanded_queries),
                    'confidence': confidence
                }
            )

            return result

        except Exception as e:
            logger.error(
                f"Error processing query: {str(e)}",
                extra={'query': query},
                exc_info=True
            )
            # Return basic understanding on error
            return QueryUnderstanding(
                original_query=query,
                processed_query=query.strip().lower(),
                confidence=0.5
            )

    def expand_query(self, query: str, entities: Optional[Dict] = None) -> List[str]:
        """
        Generate semantically similar query variations.

        Args:
            query: Processed query string
            entities: Extracted entities (optional)

        Returns:
            List of expanded query strings

        Example:
            >>> processor.expand_query("lunch bag")
            ['lunch box', 'food container bag', 'lunch tote', 'insulated lunch bag']
        """
        logger.debug(f"Expanding query: {query}")

        expanded = []

        try:
            # Synonym-based expansion
            words = query.split()
            for i, word in enumerate(words):
                if word in self.synonyms:
                    for synonym in self.synonyms[word][:2]:  # Max 2 synonyms per word
                        expanded_words = words.copy()
                        expanded_words[i] = synonym
                        expanded.append(' '.join(expanded_words))

            # Entity-based expansion (add attributes if missing)
            if entities:
                # If no color specified, try common colors for this category
                if 'categories' in entities and not entities.get('colors'):
                    category = entities['categories'][0] if entities['categories'] else None
                    if category:
                        common_colors = self._get_common_colors_for_category(category)
                        for color in common_colors[:2]:
                            expanded.append(f"{color} {query}")

            # Remove duplicates and original query
            expanded = [q for q in set(expanded) if q != query]

            logger.debug(f"Query expanded to {len(expanded)} variations", extra={
                'original': query,
                'expanded_count': len(expanded)
            })

            return expanded[:3]  # Limit to top 3 expansions

        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}", exc_info=True)
            return []

    def extract_entities(self, query: str) -> Dict[str, List]:
        """
        Extract structured data from query.

        Args:
            query: Processed query string

        Returns:
            Dictionary of extracted entities

        Example:
            >>> processor.extract_entities("red polka dot lunch bag under $20")
            {
                'colors': ['red'],
                'patterns': ['polka dot'],
                'categories': ['lunch bag'],
                'price_max': 20.0
            }
        """
        logger.debug(f"Extracting entities from: {query}")

        entities = {}

        try:
            # Extract colors
            colors = []
            for color in self.color_patterns:
                if re.search(rf'\b{color}\b', query, re.IGNORECASE):
                    colors.append(color)
            if colors:
                entities['colors'] = colors

            # Extract categories/product types
            categories = []
            for category, patterns in self.category_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        categories.append(category)
                        break
            if categories:
                entities['categories'] = list(set(categories))  # Remove duplicates

            # Extract patterns
            patterns = []
            pattern_keywords = ['polka dot', 'striped', 'checkered', 'spotted', 'floral', 'plain']
            for pattern_kw in pattern_keywords:
                if pattern_kw in query.lower():
                    patterns.append(pattern_kw)
            if patterns:
                entities['patterns'] = patterns

            # Extract price information
            price_match = re.search(self.price_patterns, query)
            if price_match:
                price_str = price_match.group(1)
                try:
                    price = float(price_str)
                    if 'under' in query or 'below' in query or 'less than' in query:
                        entities['price_max'] = price
                    elif 'over' in query or 'above' in query or 'more than' in query:
                        entities['price_min'] = price
                    else:
                        entities['price_target'] = price
                except ValueError:
                    pass

            # Extract materials
            materials = []
            material_keywords = ['metal', 'plastic', 'wood', 'ceramic', 'glass', 'fabric', 'cotton', 'leather']
            for material in material_keywords:
                if re.search(rf'\b{material}\b', query, re.IGNORECASE):
                    materials.append(material)
            if materials:
                entities['materials'] = materials

            logger.debug(f"Extracted {len(entities)} entity types", extra={
                'query': query,
                'entities': entities
            })

            return entities

        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
            return {}

    def extract_intent(self, query: str, entities: Optional[Dict] = None) -> Dict:
        """
        Detect user intent from query.

        Args:
            query: Processed query string
            entities: Extracted entities (optional)

        Returns:
            Dictionary with intent information

        Example:
            >>> processor.extract_intent("show me cheap lunch bags")
            {
                'primary_intent': 'search',
                'modifiers': ['cheap'],
                'specificity': 0.7
            }
        """
        logger.debug(f"Detecting intent for: {query}")

        try:
            intent = {
                'primary_intent': 'search',
                'modifiers': [],
                'specificity': 0.5
            }

            # Browse intent
            browse_keywords = ['show', 'browse', 'view', 'see', 'list']
            if any(keyword in query.lower() for keyword in browse_keywords):
                intent['primary_intent'] = 'browse'

            # Compare intent
            compare_keywords = ['compare', 'difference', 'vs', 'versus', 'better']
            if any(keyword in query.lower() for keyword in compare_keywords):
                intent['primary_intent'] = 'compare'

            # Info intent
            info_keywords = ['what is', 'tell me about', 'information', 'details']
            if any(keyword in query.lower() for keyword in info_keywords):
                intent['primary_intent'] = 'info'

            # Extract modifiers
            modifier_keywords = {
                'cheap': 'budget',
                'affordable': 'budget',
                'expensive': 'premium',
                'quality': 'premium',
                'best': 'top_rated',
                'popular': 'trending',
                'new': 'latest'
            }

            for keyword, modifier in modifier_keywords.items():
                if keyword in query.lower():
                    intent['modifiers'].append(modifier)

            # Calculate specificity
            specificity = 0.3  # Base specificity
            if entities:
                # More entities = more specific
                specificity += len(entities) * 0.1
            if len(query.split()) > 3:
                # Longer queries tend to be more specific
                specificity += 0.2

            intent['specificity'] = min(specificity, 1.0)

            logger.debug(f"Intent detected", extra={'intent': intent, 'query': query})

            return intent

        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}", exc_info=True)
            return {'primary_intent': 'search', 'modifiers': [], 'specificity': 0.5}

    def suggest_refinements(
        self,
        query: str,
        results: List[Dict],
        max_suggestions: int = 5
    ) -> List[str]:
        """
        Suggest query refinements based on results.

        Args:
            query: Original query
            results: Search results
            max_suggestions: Maximum number of suggestions

        Returns:
            List of refinement suggestions

        Example:
            >>> results = [{'description': 'LUNCH BAG RED'}, {'description': 'LUNCH BAG PINK'}]
            >>> processor.suggest_refinements("bag", results)
            ['lunch bag', 'red lunch bag', 'pink lunch bag']
        """
        logger.debug(f"Generating refinement suggestions for: {query}")

        try:
            suggestions = []

            if not results:
                # No results - suggest popular categories
                suggestions = ['lunch bag', 'alarm clock', 'tea set', 'storage bag', 'polka dot items']
                return suggestions[:max_suggestions]

            # Analyze result descriptions to find common attributes
            descriptions = [r.get('description', '').lower() for r in results[:20]]

            # Extract common words (potential refinements)
            word_counts = {}
            for desc in descriptions:
                words = desc.split()
                for word in words:
                    if len(word) > 3 and word not in query.lower():  # Meaningful words not in query
                        word_counts[word] = word_counts.get(word, 0) + 1

            # Sort by frequency
            common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

            # Build refinement suggestions
            for word, count in common_words[:max_suggestions]:
                if count >= 2:  # Appears in at least 2 results
                    suggestions.append(f"{query} {word}".strip())

            logger.debug(f"Generated {len(suggestions)} refinement suggestions")

            return suggestions[:max_suggestions]

        except Exception as e:
            logger.error(f"Error generating refinements: {str(e)}", exc_info=True)
            return []

    # Private helper methods

    def _clean_query(self, query: str) -> str:
        """Clean and normalize query string."""
        # Remove extra whitespace
        cleaned = ' '.join(query.split())

        # Convert to lowercase
        cleaned = cleaned.lower()

        # Remove special characters (except $ for prices)
        cleaned = re.sub(r'[^\w\s$.-]', '', cleaned)

        return cleaned.strip()

    def _correct_spelling(self, query: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Perform spell correction on query.

        Returns:
            (corrected_query, list of (original, correction) tuples)
        """
        # Simple correction dictionary
        corrections_dict = {
            'luch': 'lunch',
            'lunsh': 'lunch',
            'luntch': 'lunch',
            'alram': 'alarm',
            'clok': 'clock',
            'clokc': 'clock',
            'bagg': 'bag',
            'storag': 'storage',
            'poket': 'pocket',
            'cushin': 'cushion'
        }

        corrected = query
        corrections = []

        for wrong, right in corrections_dict.items():
            if wrong in corrected:
                corrected = corrected.replace(wrong, right)
                corrections.append((wrong, right))

        return corrected, corrections

    def _calculate_confidence(
        self,
        query: str,
        entities: Dict,
        corrections: List
    ) -> float:
        """Calculate confidence score for query understanding."""
        confidence = 1.0

        # Reduce confidence for very short queries
        if len(query.split()) < 2:
            confidence -= 0.2

        # Reduce confidence if spelling corrections were applied
        if corrections:
            confidence -= 0.1 * len(corrections)

        # Increase confidence if entities were found
        if entities:
            confidence += 0.1

        return max(0.1, min(confidence, 1.0))

    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load synonym database."""
        return {
            'bag': ['tote', 'carrier', 'sack', 'pouch'],
            'lunch': ['food', 'meal'],
            'storage': ['container', 'organizer', 'holder'],
            'clock': ['timepiece', 'timer'],
            'alarm': ['wake-up', 'alert'],
            'tea': ['beverage', 'drink'],
            'set': ['collection', 'kit', 'bundle'],
            'polka': ['spotted', 'dotted'],
            'red': ['crimson', 'scarlet', 'cherry'],
            'blue': ['navy', 'azure', 'cobalt'],
            'green': ['lime', 'olive', 'emerald'],
            'yellow': ['golden', 'lemon'],
            'pink': ['rose', 'coral', 'blush'],
            'cheap': ['affordable', 'budget', 'economical'],
            'expensive': ['premium', 'luxury', 'high-end']
        }

    def _load_color_patterns(self) -> List[str]:
        """Load color patterns for entity extraction."""
        return [
            'red', 'blue', 'green', 'yellow', 'pink', 'purple', 'orange', 'black',
            'white', 'brown', 'gray', 'grey', 'gold', 'silver', 'navy', 'beige',
            'turquoise', 'lime', 'coral', 'cream', 'ivory', 'mint'
        ]

    def _load_category_patterns(self) -> Dict[str, List[str]]:
        """Load category patterns for entity extraction."""
        return {
            'lunch bag': [r'\blunch\s*bag\b', r'\blunch\s*box\b', r'\bfood\s*bag\b'],
            'storage bag': [r'\bstorage\s*bag\b', r'\borganizer\s*bag\b'],
            'shopping bag': [r'\bshopping\s*bag\b', r'\btote\s*bag\b', r'\bshopper\b'],
            'alarm clock': [r'\balarm\s*clock\b', r'\bclock\b'],
            'tea set': [r'\btea\s*set\b', r'\bteapot\b', r'\btea\s*service\b'],
            'bunting': [r'\bbunting\b', r'\bgarland\b', r'\bbanner\b'],
            'hot water bottle': [r'\bhot\s*water\s*bottle\b', r'\bwarmer\b'],
            'cake stand': [r'\bcake\s*stand\b', r'\bcakestand\b', r'\btiered\s*stand\b'],
            'holder': [r'\bholder\b', r'\bstand\b'],
            'lantern': [r'\blantern\b', r'\blight\b', r'\blamp\b']
        }

    def _compile_price_patterns(self) -> re.Pattern:
        """Compile regex pattern for price extraction."""
        return re.compile(r'\$?\s*(\d+(?:\.\d{2})?)')

    def _get_common_colors_for_category(self, category: str) -> List[str]:
        """Get common colors for a product category."""
        category_colors = {
            'lunch bag': ['red', 'pink', 'blue', 'green'],
            'alarm clock': ['red', 'black', 'white', 'silver'],
            'tea set': ['white', 'blue', 'floral', 'vintage'],
            'bunting': ['multi-color', 'red', 'blue', 'vintage'],
            'storage bag': ['clear', 'blue', 'gray', 'black']
        }

        return category_colors.get(category, ['red', 'blue', 'black', 'white'])
