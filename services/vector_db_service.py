"""
Vector Database Service - Pinecone Integration
Handles product vector search using Pinecone vector database.
"""

import os
import logging
from typing import List, Dict, Any
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class VectorDBService:
    """
    Service for interacting with Pinecone vector database.
    Provides product search functionality using semantic similarity.
    """

    def __init__(self):
        """Initialize Pinecone client and embedding model."""
        try:
            logger.info("Initializing VectorDBService...")

            # Load environment variables
            load_dotenv()

            # Initialize Pinecone
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                logger.error("PINECONE_API_KEY not found in environment variables")
                raise ValueError("PINECONE_API_KEY not found in environment variables")

            logger.info("Connecting to Pinecone...")
            self.pc = Pinecone(api_key=api_key)
            self.index_name = "ecommerce-products"
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Successfully connected to Pinecone index: {self.index_name}")

            # Load embedding model
            logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("VectorDBService initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize VectorDBService: {str(e)}", exc_info=True)
            raise

    def search_products(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for products using natural language query.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)

        Returns:
            List of product dictionaries with metadata and scores
        """
        try:
            logger.info(f"Searching products for query: '{query}' (top_k={top_k}, min_score={min_score})")

            # Generate query embedding
            logger.debug("Generating query embedding...")
            query_vector = self.model.encode([query], normalize_embeddings=True)[0].tolist()

            # Search Pinecone
            logger.debug(f"Querying Pinecone index: {self.index_name}")
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )

            # Format results
            products = []
            for match in results['matches']:
                if match['score'] >= min_score:
                    products.append({
                        'stock_code': match['metadata']['stock_code'],
                        'description': match['metadata']['description'],
                        'price': match['metadata']['price'],
                        'country': match['metadata']['country'],
                        'similarity_score': match['score']
                    })

            logger.info(f"Found {len(products)} products matching query (filtered by min_score={min_score})")
            if products:
                logger.debug(f"Top result: {products[0]['description']} (score: {products[0]['similarity_score']:.4f})")

            return products

        except Exception as e:
            logger.error(f"Error searching products for query '{query}': {str(e)}", exc_info=True)
            raise

    def get_similar_products(
        self,
        stock_code: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar products based on a given product's stock code.

        Args:
            stock_code: Stock code of the reference product
            top_k: Number of similar products to return

        Returns:
            List of similar product dictionaries
        """
        try:
            logger.info(f"Finding similar products for stock_code: {stock_code} (top_k={top_k})")

            # Fetch the product vector
            logger.debug(f"Fetching vector for stock_code: {stock_code}")
            fetch_result = self.index.fetch(ids=[str(stock_code)])

            if not fetch_result['vectors']:
                logger.warning(f"No vector found for stock_code: {stock_code}")
                return []

            # Get the vector
            product_vector = fetch_result['vectors'][str(stock_code)]['values']
            logger.debug(f"Retrieved vector for stock_code: {stock_code}")

            # Search for similar products
            logger.debug(f"Querying for similar products...")
            results = self.index.query(
                vector=product_vector,
                top_k=top_k + 1,  # +1 because the product itself will be returned
                include_metadata=True
            )

            # Format results (exclude the product itself)
            products = []
            for match in results['matches']:
                if match['metadata']['stock_code'] != stock_code:
                    products.append({
                        'stock_code': match['metadata']['stock_code'],
                        'description': match['metadata']['description'],
                        'price': match['metadata']['price'],
                        'country': match['metadata']['country'],
                        'similarity_score': match['score']
                    })

            final_products = products[:top_k]
            logger.info(f"Found {len(final_products)} similar products for stock_code: {stock_code}")

            return final_products

        except Exception as e:
            logger.error(f"Error finding similar products for stock_code '{stock_code}': {str(e)}", exc_info=True)
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.

        Returns:
            Dictionary containing index statistics
        """
        try:
            logger.info(f"Retrieving index stats for: {self.index_name}")
            stats = self.index.describe_index_stats()

            index_info = {
                'total_vectors': stats['total_vector_count'],
                'dimension': stats['dimension'],
                'index_fullness': stats.get('index_fullness', 0)
            }

            logger.info(f"Index stats: {index_info}")
            return index_info

        except Exception as e:
            logger.error(f"Error retrieving index stats: {str(e)}", exc_info=True)
            raise

    def validate_query(self, query: str) -> tuple[bool, str]:
        """
        Validate user query for safety and quality.

        Args:
            query: User's search query

        Returns:
            Tuple of (is_valid, error_message)
        """
        logger.debug(f"Validating query: '{query}'")

        # Check if query is empty
        if not query or not query.strip():
            logger.warning("Validation failed: Query is empty")
            return False, "Query cannot be empty"

        # Check query length
        if len(query) > 500:
            logger.warning(f"Validation failed: Query too long ({len(query)} chars)")
            return False, "Query too long (max 500 characters)"

        if len(query) < 2:
            logger.warning(f"Validation failed: Query too short ({len(query)} chars)")
            return False, "Query too short (min 2 characters)"

        # Basic injection check (simple version)
        suspicious_patterns = ['<script', 'javascript:', 'onerror=', 'onclick=']
        query_lower = query.lower()
        for pattern in suspicious_patterns:
            if pattern in query_lower:
                logger.warning(f"Validation failed: Suspicious pattern detected: {pattern}")
                return False, "Invalid characters detected"

        logger.debug("Query validation passed")
        return True, ""
