from flask import Flask, request, jsonify, render_template
from services.vector_db_service import VectorDBService
from services.ocr_service import OCRService
from services.cnn_service import CNNService

app = Flask(__name__)

# Initialize services
vector_service = VectorDBService()
ocr_service = OCRService()
cnn_service = CNNService()

@app.route('/product-recommendation', methods=['POST'])
def product_recommendation():
    """
    Endpoint for product recommendations based on natural language queries.
    Input: Form data containing 'query' (string).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    query = request.form.get('query', '')

    # Validate query
    is_valid, error_msg = vector_service.validate_query(query)
    if not is_valid:
        return jsonify({
            "products": [],
            "response": f"Invalid query: {error_msg}"
        }), 400

    # Search products
    results = vector_service.search_products(query, top_k=10, min_score=0.3)

    # Format products and filter out invalid entries
    products = []
    for result in results:
        # Skip invalid entries (ebay, test data, etc.)
        description = result['description'].strip().lower()
        if description in ['ebay', 'test', 'unknown', ''] or result['price'] == 0:
            continue

        products.append({
            "StockCode": result['stock_code'],
            "Description": result['description'],
            "UnitPrice": f"${result['price']:.2f}",
            "Country": result['country']
        })

    # Generate response
    if products:
        top_products = [p['Description'] for p in products[:3]]
        response = f"Found {len(products)} products matching '{query}'. Top matches: {', '.join(top_products)}."
    else:
        # Provide helpful suggestions
        suggestions = [
            "red lunch bag", "alarm clock", "hand warmer", "polka dot",
            "heart holder", "metal lantern", "building blocks", "coat rack"
        ]
        response = (f"No products found matching '{query}'. "
                   f"Our catalog focuses on home decor and gift items. "
                   f"Try searching for: {', '.join(suggestions[:4])}.")

    return jsonify({"products": products, "response": response})

@app.route('/ocr-query', methods=['POST'])
def ocr_query():
    """
    Endpoint to process handwritten queries extracted from uploaded images.
    Input: Form data containing 'image_data' (file, base64-encoded image or direct file upload).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    image_file = request.files.get('image_data')

    if not image_file:
        return jsonify({
            "products": [],
            "response": "No image file provided"
        }), 400

    try:
        # Extract text from image
        image_bytes = image_file.read()
        extracted_text, confidence = ocr_service.extract_text_from_bytes(image_bytes)

        if not extracted_text:
            return jsonify({
                "products": [],
                "response": "Could not extract text from image. Please ensure the image contains clear, readable text."
            }), 400

        # Validate extracted text
        is_valid, error_msg = vector_service.validate_query(extracted_text)
        if not is_valid:
            return jsonify({
                "products": [],
                "response": f"Extracted text '{extracted_text}' is invalid: {error_msg}"
            }), 400

        # Search products using extracted text
        results = vector_service.search_products(extracted_text, top_k=10, min_score=0.3)

        # Format products and filter out invalid entries
        products = []
        for result in results:
            # Skip invalid entries (ebay, test data, etc.)
            description = result['description'].strip().lower()
            if description in ['ebay', 'test', 'unknown', ''] or result['price'] == 0:
                continue

            products.append({
                "StockCode": result['stock_code'],
                "Description": result['description'],
                "UnitPrice": f"${result['price']:.2f}",
                "Country": result['country']
            })

        # Generate response
        if products:
            top_products = [p['Description'] for p in products[:3]]
            response = (
                f"Extracted query: '{extracted_text}' (confidence: {confidence:.0%}). "
                f"Found {len(products)} products. "
                f"Top matches: {', '.join(top_products)}."
            )
        else:
            # Provide helpful suggestions
            suggestions = ["red lunch bag", "alarm clock", "hand warmer", "polka dot"]
            response = (f"Extracted query: '{extracted_text}' (confidence: {confidence:.0%}). "
                       f"No products found. Our catalog focuses on home decor and gift items. "
                       f"Try searching for: {', '.join(suggestions)}.")

        return jsonify({"products": products, "response": response})

    except Exception as e:
        return jsonify({
            "products": [],
            "response": f"Error processing image: {str(e)}"
        }), 500

@app.route('/image-product-search', methods=['POST'])
def image_product_search():
    """
    Endpoint to identify and suggest products from uploaded product images.
    Input: Form data containing 'product_image' (file, base64-encoded image or direct file upload).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    product_image = request.files.get('product_image')

    if not product_image:
        return jsonify({
            "products": [],
            "response": "No product image provided"
        }), 400

    try:
        # Read image bytes
        image_bytes = product_image.read()

        # Use CNN to predict product class
        prediction = cnn_service.predict_from_bytes(image_bytes, return_top_k=3)

        # Handle prediction errors
        if 'error' in prediction:
            return jsonify({
                "products": [],
                "response": f"Error processing image: {prediction['error']}"
            }), 500

        # Get CNN prediction details
        predicted_stock_code = prediction['predicted_stock_code']
        predicted_product = prediction['predicted_product']
        confidence = prediction['confidence']
        top_predictions = prediction['top_predictions']

        # Handle low confidence predictions
        if confidence < 0.3:
            return jsonify({
                "products": [],
                "response": (
                    f"Unable to identify product with confidence. "
                    f"Best guess: {predicted_product} ({confidence:.0%} confidence). "
                    f"Please try a clearer image."
                )
            }), 200

        # Use the predicted product name to search for similar products in vector DB
        search_query = predicted_product
        results = vector_service.search_products(search_query, top_k=10, min_score=0.3)

        # Format products and filter out invalid entries
        products = []
        for result in results:
            # Skip invalid entries (ebay, test data, etc.)
            description = result['description'].strip().lower()
            if description in ['ebay', 'test', 'unknown', ''] or result['price'] == 0:
                continue

            products.append({
                "StockCode": result['stock_code'],
                "Description": result['description'],
                "UnitPrice": f"${result['price']:.2f}",
                "Country": result['country']
            })

        # Generate natural language response
        if products:
            # Build response with CNN prediction info
            response = f"Detected product: {predicted_product} ({confidence:.0%} confidence). "

            # Add alternative predictions if confidence is moderate
            if confidence < 0.7 and len(top_predictions) > 1:
                alternatives = [p['product_name'] for p in top_predictions[1:]]
                response += f"Other possibilities: {', '.join(alternatives)}. "

            # Add search results info
            top_products = [p['Description'] for p in products[:3]]
            response += (
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
                "stock_code": predicted_stock_code,
                "confidence": f"{confidence:.2%}",
                "top_predictions": [
                    {
                        "product": p['product_name'],
                        "confidence": f"{p['confidence']:.2%}"
                    }
                    for p in top_predictions
                ]
            }
        })

    except Exception as e:
        return jsonify({
            "products": [],
            "response": f"Error processing product image: {str(e)}"
        }), 500

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
    """
    Endpoint to return a sample JSON response for the API.
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    return render_template('sample_response.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
