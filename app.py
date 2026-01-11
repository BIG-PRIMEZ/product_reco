"""
Flask application for product recommendation system.

Refactored to use intelligent RecommenderService for all endpoints.
"""

from flask import Flask, request, jsonify, render_template
from services.vector_db_service import VectorDBService
from services.ocr_service import OCRService
from services.cnn_service import CNNService
from services.recommender_service import RecommenderService
from utils.logger import init_app_logging, get_logger
from config import get_config

# Initialize logging
config = get_config()
init_app_logging(
    level=config.log_level,
    log_dir=config.log_dir if not config.debug else None,
    json_format=config.json_logs
)

logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize services
logger.info("Initializing services...")
vector_service = VectorDBService()
ocr_service = OCRService()
cnn_service = CNNService()

# Initialize recommender service (new!)
recommender_service = RecommenderService(
    vector_service=vector_service,
    default_strategy=config.default_ranking_strategy
)

logger.info("All services initialized successfully")


@app.route('/product-recommendation', methods=['POST'])
def product_recommendation():
    """
    Endpoint for product recommendations based on natural language queries.
    Input: Form data containing 'query' (string).
    Output: JSON with products, query understanding, and suggestions.
    """
    query = request.form.get('query', '')

    logger.info(f"Text query request", extra={'query': query, 'endpoint': '/product-recommendation'})

    try:
        # Validate query
        is_valid, error_msg = vector_service.validate_query(query)
        if not is_valid:
            logger.warning(f"Invalid query: {error_msg}", extra={'query': query})
            return jsonify({
                "products": [],
                "response": f"Invalid query: {error_msg}"
            }), 400

        # Use new recommender service
        if config.use_new_recommender:
            recommendation = recommender_service.recommend(
                query=query,
                strategy=config.default_ranking_strategy,
                top_k=config.default_top_k,
                min_score=config.min_similarity_score
            )

            # Format response for backward compatibility
            formatted_products = [
                {
                    "StockCode": p['stock_code'],
                    "Description": p['description'],
                    "UnitPrice": p['unit_price'],
                    "Country": p['country'],
                    "RelevanceScore": p.get('relevance_score')
                }
                for p in recommendation.products
            ]

            response_data = {
                "products": formatted_products,
                "response": recommendation.response,
                "query_understanding": recommendation.query_understanding,
                "suggestions": recommendation.suggestions,
                "metadata": recommendation.metadata
            }

            logger.info(
                f"Recommendation generated",
                extra={
                    'query': query,
                    'num_products': len(formatted_products),
                    'processing_time_ms': recommendation.metadata.get('processing_time_ms')
                }
            )

            return jsonify(response_data)

        else:
            # Legacy logic (fallback)
            logger.info("Using legacy recommender logic")
            results = vector_service.search_products(query, top_k=10, min_score=0.3)

            products = []
            for result in results:
                description = result['description'].strip().lower()
                if description in ['ebay', 'test', 'unknown', ''] or result['price'] == 0:
                    continue

                products.append({
                    "StockCode": result['stock_code'],
                    "Description": result['description'],
                    "UnitPrice": f"${result['price']:.2f}",
                    "Country": result['country']
                })

            if products:
                top_products = [p['Description'] for p in products[:3]]
                response = f"Found {len(products)} products matching '{query}'. Top matches: {', '.join(top_products)}."
            else:
                suggestions = ["red lunch bag", "alarm clock", "hand warmer", "polka dot"]
                response = (f"No products found matching '{query}'. "
                           f"Our catalog focuses on home decor and gift items. "
                           f"Try searching for: {', '.join(suggestions[:4])}.")

            return jsonify({"products": products, "response": response})

    except Exception as e:
        logger.error(
            f"Error processing text query: {str(e)}",
            extra={'query': query},
            exc_info=True
        )
        return jsonify({
            "products": [],
            "response": "An error occurred while processing your request. Please try again."
        }), 500


@app.route('/ocr-query', methods=['POST'])
def ocr_query():
    """
    Endpoint to process handwritten queries extracted from uploaded images.
    Input: Form data containing 'image_data' (file).
    Output: JSON with products and response.
    """
    image_file = request.files.get('image_data')

    logger.info("OCR query request", extra={'endpoint': '/ocr-query'})

    if not image_file:
        logger.warning("No image file provided")
        return jsonify({
            "products": [],
            "response": "No image file provided"
        }), 400

    try:
        # Extract text from image
        image_bytes = image_file.read()
        extracted_text, confidence = ocr_service.extract_text_from_bytes(image_bytes)

        logger.info(
            f"OCR extraction complete",
            extra={'extracted_text': extracted_text, 'confidence': confidence}
        )

        if not extracted_text:
            logger.warning("Could not extract text from image")
            return jsonify({
                "products": [],
                "response": "Could not extract text from image. Please ensure the image contains clear, readable text."
            }), 400

        # Validate extracted text
        is_valid, error_msg = vector_service.validate_query(extracted_text)
        if not is_valid:
            logger.warning(f"Invalid extracted text: {error_msg}", extra={'text': extracted_text})
            return jsonify({
                "products": [],
                "response": f"Extracted text '{extracted_text}' is invalid: {error_msg}"
            }), 400

        # Use new recommender service
        if config.use_new_recommender:
            recommendation = recommender_service.recommend(
                query=extracted_text,
                strategy=config.default_ranking_strategy,
                top_k=config.default_top_k,
                min_score=config.min_similarity_score
            )

            # Format response
            formatted_products = [
                {
                    "StockCode": p['stock_code'],
                    "Description": p['description'],
                    "UnitPrice": p['unit_price'],
                    "Country": p['country']
                }
                for p in recommendation.products
            ]

            # Add OCR info to response
            response = f"Extracted: '{extracted_text}' (confidence: {confidence:.0%}). "
            response += recommendation.response

            return jsonify({
                "products": formatted_products,
                "response": response,
                "extracted_text": extracted_text,
                "ocr_confidence": confidence,
                "query_understanding": recommendation.query_understanding,
                "suggestions": recommendation.suggestions
            })

        else:
            # Legacy logic
            results = vector_service.search_products(extracted_text, top_k=10, min_score=0.3)

            products = []
            for result in results:
                description = result['description'].strip().lower()
                if description in ['ebay', 'test', 'unknown', ''] or result['price'] == 0:
                    continue

                products.append({
                    "StockCode": result['stock_code'],
                    "Description": result['description'],
                    "UnitPrice": f"${result['price']:.2f}",
                    "Country": result['country']
                })

            if products:
                top_products = [p['Description'] for p in products[:3]]
                response = (
                    f"Extracted query: '{extracted_text}' (confidence: {confidence:.0%}). "
                    f"Found {len(products)} products. "
                    f"Top matches: {', '.join(top_products)}."
                )
            else:
                suggestions = ["red lunch bag", "alarm clock", "hand warmer", "polka dot"]
                response = (f"Extracted query: '{extracted_text}' (confidence: {confidence:.0%}). "
                           f"No products found. Try: {', '.join(suggestions)}.")

            return jsonify({"products": products, "response": response})

    except Exception as e:
        logger.error(f"Error processing OCR query: {str(e)}", exc_info=True)
        return jsonify({
            "products": [],
            "response": f"Error processing image: {str(e)}"
        }), 500


@app.route('/image-product-search', methods=['POST'])
def image_product_search():
    """
    Endpoint to identify and suggest products from uploaded product images.
    Input: Form data containing 'product_image' (file).
    Output: JSON with products and CNN prediction info.
    """
    product_image = request.files.get('product_image')

    logger.info("Image product search request", extra={'endpoint': '/image-product-search'})

    if not product_image:
        logger.warning("No product image provided")
        return jsonify({
            "products": [],
            "response": "No product image provided"
        }), 400

    try:
        # Read image bytes
        image_bytes = product_image.read()

        # Use CNN to predict product class
        prediction = cnn_service.predict_from_bytes(image_bytes, return_top_k=3)

        logger.info(
            f"CNN prediction complete",
            extra={
                'predicted_product': prediction.get('predicted_product'),
                'confidence': prediction.get('confidence')
            }
        )

        # Handle prediction errors
        if 'error' in prediction:
            logger.error(f"CNN prediction error: {prediction['error']}")
            return jsonify({
                "products": [],
                "response": f"Error processing image: {prediction['error']}"
            }), 500

        # Get CNN prediction details
        predicted_product = prediction['predicted_product']
        confidence = prediction['confidence']

        # Handle low confidence predictions
        if confidence < 0.3:
            logger.warning(f"Low confidence prediction: {confidence:.2%}")
            return jsonify({
                "products": [],
                "response": (
                    f"Unable to identify product with confidence. "
                    f"Best guess: {predicted_product} ({confidence:.0%} confidence). "
                    f"Please try a clearer image."
                )
            }), 200

        # Use new recommender service with predicted product name
        if config.use_new_recommender:
            recommendation = recommender_service.recommend(
                query=predicted_product,
                strategy=config.default_ranking_strategy,
                top_k=config.default_top_k,
                min_score=config.min_similarity_score
            )

            # Format response
            formatted_products = [
                {
                    "StockCode": p['stock_code'],
                    "Description": p['description'],
                    "UnitPrice": p['unit_price'],
                    "Country": p['country']
                }
                for p in recommendation.products
            ]

            # Build response message
            response = f"Detected: {predicted_product} ({confidence:.0%} confidence). "
            if recommendation.products:
                response += f"Found {len(recommendation.products)} similar products."
            else:
                response += "No similar products found in database."

            return jsonify({
                "products": formatted_products,
                "response": response,
                "cnn_prediction": {
                    "predicted_class": predicted_product,
                    "stock_code": prediction.get('predicted_stock_code'),
                    "confidence": f"{confidence:.2%}",
                    "top_predictions": [
                        {
                            "product": p['product_name'],
                            "confidence": f"{p['confidence']:.2%}"
                        }
                        for p in prediction.get('top_predictions', [])
                    ]
                },
                "query_understanding": recommendation.query_understanding,
                "suggestions": recommendation.suggestions
            })

        else:
            # Legacy logic
            results = vector_service.search_products(predicted_product, top_k=10, min_score=0.3)

            products = []
            for result in results:
                description = result['description'].strip().lower()
                if description in ['ebay', 'test', 'unknown', ''] or result['price'] == 0:
                    continue

                products.append({
                    "StockCode": result['stock_code'],
                    "Description": result['description'],
                    "UnitPrice": f"${result['price']:.2f}",
                    "Country": result['country']
                })

            if products:
                top_products = [p['Description'] for p in products[:3]]
                response = (
                    f"Detected product: {predicted_product} ({confidence:.0%} confidence). "
                    f"Found {len(products)} similar products. "
                    f"Top matches: {', '.join(top_products)}."
                )
            else:
                response = (
                    f"Detected product: {predicted_product} ({confidence:.0%} confidence). "
                    f"No similar products found in database."
                )

            return jsonify({
                "products": products,
                "response": response,
                "cnn_prediction": {
                    "predicted_class": predicted_product,
                    "confidence": f"{confidence:.2%}"
                }
            })

    except Exception as e:
        logger.error(f"Error processing product image: {str(e)}", exc_info=True)
        return jsonify({
            "products": [],
            "response": f"Error processing product image: {str(e)}"
        }), 500


# Frontend routes
@app.route('/', methods=['GET'])
def index():
    """Homepage with navigation to all search interfaces."""
    return render_template('index.html')


@app.route('/text-query', methods=['GET'])
def text_query_page():
    """Text query search interface."""
    return render_template('text_query.html')


@app.route('/image-query', methods=['GET'])
def image_query_page():
    """Handwritten query image upload interface."""
    return render_template('image_query.html')


@app.route('/product-upload', methods=['GET'])
def product_upload_page():
    """Product image upload interface."""
    return render_template('product_upload.html')


@app.route('/sample_response', methods=['GET'])
def sample_response():
    """Sample JSON response page."""
    return render_template('sample_response.html')


# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "services": {
            "vector_db": "active",
            "ocr": "active",
            "cnn": "active",
            "recommender": "active" if config.use_new_recommender else "disabled"
        }
    })


if __name__ == '__main__':
    logger.info(
        f"Starting Flask app",
        extra={
            'port': config.flask_port,
            'debug': config.debug,
            'recommender_enabled': config.use_new_recommender
        }
    )

    app.run(
        debug=config.debug,
        host=config.flask_host,
        port=config.flask_port
    )
