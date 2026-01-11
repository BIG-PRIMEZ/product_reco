# System Architecture

## Overview

The Intelligent Product Recommender is a multi-modal search system that combines natural language processing, computer vision, and vector similarity search to provide personalized product recommendations.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                           │
│  (HTML/CSS/JS - text_query.html, image_query.html, etc.)       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                        Flask API Layer                           │
│  - /product-recommendation (text queries)                        │
│  - /ocr-query (handwritten text)                                 │
│  - /image-product-search (product images)                        │
└────────────┬────────────┬────────────┬─────────────────────────┘
             │            │            │
    ┌────────▼──┐  ┌─────▼─────┐  ┌──▼──────────┐
    │ OCRService│  │CNNService │  │QueryProcessor│
    └────────┬──┘  └─────┬─────┘  └──┬──────────┘
             │            │            │
             └────────────┴────────────┘
                         │
          ┌──────────────▼──────────────────┐
          │   RecommenderService             │
          │  (orchestration & ranking)       │
          └──────────────┬──────────────────┘
                         │
          ┌──────────────▼──────────────────┐
          │   VectorDBService                │
          │  (Pinecone similarity search)    │
          └──────────────────────────────────┘
```

## Component Architecture

### 1. API Layer (`app.py`)

**Purpose**: REST API endpoints for client requests

**Key Routes**:
- `POST /product-recommendation` - Text-based product search
- `POST /ocr-query` - Handwritten text recognition + search
- `POST /image-product-search` - Image-based product identification + search
- `GET /text-query, /image-query, /product-upload` - Frontend pages

**Responsibilities**:
- Request validation and sanitization
- Input preprocessing (file uploads, text cleaning)
- Service orchestration
- Response formatting
- Error handling and logging

**Feature Flags**:
- `USE_NEW_RECOMMENDER` - Enable/disable new recommendation engine
- Allows A/B testing and gradual rollout

### 2. Service Layer

#### 2.1 RecommenderService (`services/recommender_service.py`)

**Purpose**: Central orchestrator for intelligent recommendations

**Workflow**:
```
Query → Understanding → Retrieval → Ranking → Filtering → Diversification → Response
```

**Key Methods**:
- `recommend(query, context, strategy, top_k, min_score)` - Main entry point
- `_retrieve_candidates()` - Multi-stage candidate retrieval
- `_rank_products()` - Apply ranking strategy
- `_apply_business_rules()` - Filter invalid products
- `_diversify_results()` - Ensure result variety

**Configuration**:
- Default ranking strategy: `hybrid`
- Top-K recommendations: 10
- Minimum similarity score: 0.3
- Query expansion: Enabled (max 3 expansions)

#### 2.2 QueryProcessor (`services/query_processor.py`)

**Purpose**: Advanced natural language query understanding

**Capabilities**:
- **Entity Extraction**: Colors, categories, prices, materials, patterns
- **Intent Detection**: search, browse, compare, info
- **Query Expansion**: Synonym-based expansion
- **Spell Correction**: Basic typo fixing
- **Price Parsing**: "under $20", "between $10-$30", "around $15"

**Entity Types**:
```python
entities = {
    'colors': ['red', 'blue', 'green', ...],
    'categories': ['lunch bag', 'tea set', ...],
    'patterns': ['polka dot', 'striped', ...],
    'materials': ['ceramic', 'metal', ...],
    'price_max': 20.0,
    'price_min': 10.0,
    'price_target': 15.0
}
```

**Example Processing**:
```
Input: "red lunch bag under $20"
Output:
  - processed_query: "red lunch bag 20"
  - entities: {colors: ['red'], categories: ['lunch bag'], price_max: 20.0}
  - intent: "search"
  - expanded_queries: ["crimson lunch bag", "scarlet tote bag", ...]
  - confidence: 0.85
```

#### 2.3 Ranking Strategies (`services/rankers.py`)

**Purpose**: Multiple ranking algorithms for different use cases

**Available Rankers**:

1. **SimilarityRanker** - Pure vector similarity
   - Use case: Simple semantic search
   - Score: Cosine similarity (0-1)

2. **PopularityRanker** - Product popularity/sales
   - Use case: Trending items, popular products
   - Score: Normalized popularity (0-1)

3. **HybridRanker** - Weighted combination (DEFAULT)
   - Formula: `score = 0.5·similarity + 0.2·popularity + 0.2·entity_match + 0.1·price`
   - Use case: Balanced relevance and business objectives
   - Configurable weights via `config.py`

4. **MMRRanker** - Maximal Marginal Relevance
   - Use case: Result diversification
   - Balances relevance and diversity (λ parameter)

**Hybrid Scoring Details**:
```python
composite_score = (
    0.5 * similarity_score +      # Vector similarity to query
    0.2 * popularity_score +      # Product sales/views
    0.2 * entity_match_score +    # Color/category/pattern match
    0.1 * price_preference_score  # Budget alignment
)
```

**Entity Matching Logic**:
- Exact color match: +1 point
- Category match (all words): +2 points (more important)
- Pattern/material match: +1 point
- Score = matches / total_entities

**Price Scoring Logic**:
- Under budget: High score (1.0 - 0.5 × ratio)
- Over budget: Penalty (0.5 - over_ratio)
- No constraint: Neutral (0.5-0.7)

#### 2.4 VectorDBService (`services/vector_db_service.py`)

**Purpose**: Semantic search using Pinecone vector database

**Configuration**:
- Index: `ecommerce-products`
- Cloud: AWS (us-east-1)
- Dimension: 384 (sentence-transformers)
- Metric: Cosine similarity
- Serverless tier

**Embedding Model**:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensions: 384
- Speed: ~1000 tokens/sec
- Quality: Good for short text (product descriptions)

**Methods**:
- `search_products(query, top_k, min_score)` - Vector similarity search
- `validate_query(query)` - Input validation
- Returns: Products with similarity scores

#### 2.5 OCRService (`services/ocr_service.py`)

**Purpose**: Handwriting and text recognition from images

**Technology**: EasyOCR (not Tesseract)
- Language: English
- GPU: Disabled (CPU-only for compatibility)
- First load: Downloads models (~100MB)

**Methods**:
- `extract_text(image_path)` - Extract from file path
- `extract_text_from_bytes(image_bytes)` - Extract from uploaded image
- Returns: `(text, confidence_score)`

**Performance**:
- Good on handwritten text
- Confidence scoring included
- No minimum threshold (reported to user)

#### 2.6 CNNService (`services/cnn_service.py`)

**Purpose**: Product classification from images

**Model Details**:
- Architecture: Custom CNN (trained from scratch)
- Classes: 10 product categories
- Input: 128×128×3 RGB images
- Accuracy: 74.52% on test set

**Architecture**:
```
Input (128x128x3)
  ↓
Conv2D(32) + MaxPool + Dropout(0.3)
  ↓
Conv2D(64) + MaxPool + Dropout(0.3)
  ↓
Conv2D(128) + MaxPool + Dropout(0.4)
  ↓
Conv2D(128) + MaxPool + Dropout(0.4)
  ↓
Flatten → Dense(128) + Dropout(0.5)
  ↓
Dense(10, softmax)
```

**Product Classes** (Stock Code):
- 0: LUNCH BAG WOODLAND (20726)
- 1: REX CASH+CARRY JUMBO SHOPPER (21034)
- 2: JUMBO STORAGE BAG SUKI (21931)
- 3: 6 RIBBONS RUSTIC CHARM (22077)
- 4: CHOCOLATE HOT WATER BOTTLE (22112)
- 5: RETROSPOT TEA SET CERAMIC 11 PC (22139)
- 6: LUNCH BAG PINK POLKADOT (22384)
- 7: REGENCY CAKESTAND 3 TIER (22423)
- 8: ALARM CLOCK BAKELIKE RED (22727)
- 9: SPOTTY BUNTING (23298)

**Confidence Handling**:
- Minimum threshold: 30%
- Rejects low-confidence predictions
- Returns top-3 predictions with probabilities

### 3. Infrastructure Layer

#### 3.1 Logging (`utils/logger.py`)

**Purpose**: Centralized structured logging

**Features**:
- JSON formatting for production
- File + console handlers
- Request ID tracking
- Performance metrics
- Error tracing

**Log Levels**:
- DEBUG: Detailed diagnostics
- INFO: Normal operations
- WARNING: Potential issues
- ERROR: Errors with stack traces

**Configuration**:
```python
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
LOG_JSON_FORMAT=true
```

#### 3.2 Configuration (`config.py`)

**Purpose**: Environment-based configuration management

**Config Classes**:
1. **AppConfig** - Application settings
2. **RankingConfig** - Ranking weights
3. **PipelineConfig** - ML pipeline settings

**Key Settings**:
```python
# Feature flags
USE_NEW_RECOMMENDER=true
RANKING_STRATEGY=hybrid

# Search parameters
DEFAULT_TOP_K=10
MIN_SIMILARITY_SCORE=0.3

# Query processing
ENABLE_QUERY_EXPANSION=true
MAX_EXPANDED_QUERIES=3
ENABLE_SPELL_CORRECTION=true

# Ranking weights
SIMILARITY_WEIGHT=0.5
POPULARITY_WEIGHT=0.2
ENTITY_MATCH_WEIGHT=0.2
PRICE_WEIGHT=0.1

# Caching
ENABLE_CACHE=false
```

## Data Flow

### Text Query Flow
```
1. User enters "red lunch bag under $20"
2. Flask validates input
3. QueryProcessor:
   - Extracts entities: {colors: ['red'], categories: ['lunch bag'], price_max: 20}
   - Detects intent: "search"
   - Expands query: ["crimson lunch bag", "scarlet tote"]
4. RecommenderService:
   - Retrieves 50 candidates from Pinecone (primary query)
   - Retrieves 20 candidates per expanded query
   - Deduplicates by stock_code
5. HybridRanker:
   - Scores each candidate (similarity + popularity + entity + price)
   - Sorts by composite score
6. Business rules:
   - Filters invalid products (test data, zero prices)
   - Applies price constraints
7. Diversification:
   - Removes very similar products
8. Format top 10 results
9. ResponseGenerator creates natural language message
10. Return JSON with products + metadata
```

### OCR Query Flow
```
1. User uploads handwritten image
2. Flask validates file (size, format)
3. OCRService extracts text with confidence
4. Extracted text → QueryProcessor (same as text query)
5. Continue with normal search flow...
```

### Image Query Flow
```
1. User uploads product image
2. Flask validates file
3. CNNService:
   - Preprocesses image (resize to 128×128, normalize)
   - Predicts product class
   - Returns top-3 predictions with confidence
4. If confidence < 30%: reject prediction
5. Use predicted product name as search query
6. VectorDBService searches for similar products
7. Return products + CNN prediction metadata
```

## Business Logic

### Product Filtering Rules

Products are filtered out if they have:
- Empty descriptions
- Invalid terms (as standalone words): "ebay", "test", "unknown", "samples", "manual", "postage", "carriage"
- Zero or negative prices
- Very high prices (>$1000) - logged but allowed

**Word Boundary Matching**:
- ✅ "MANUAL TYPEWRITER ALARM CLOCK" - Allowed (manual is a feature)
- ❌ "INSTRUCTION MANUAL" - Filtered (manual is the product)

### Price Handling

**Extraction Patterns**:
- "under $20" → price_max = 20
- "over $50" → price_min = 50
- "around $30" → price_target = 30
- "between $10 and $20" → price_min = 10, price_max = 20

**Scoring**:
- Items under budget: Preferred (higher scores)
- Items over budget: Penalized heavily
- No budget: Neutral scoring

### Result Diversification

**Simple Heuristic**:
- Always include top result
- For remaining results, check Jaccard similarity
- If similarity > 0.8 to any selected product, skip
- Prevents showing 10 variations of "RED LUNCH BAG"

## Scalability Considerations

### Current Bottlenecks
1. **EasyOCR initialization** - 1-2 minute first load
2. **CNN inference** - ~200ms per image
3. **Pinecone queries** - 50-100ms per request
4. **No caching** - Every query hits DB

### Scaling Strategy

**Horizontal Scaling**:
- Flask app is stateless
- Can run multiple instances behind load balancer
- Services can be extracted to separate microservices

**Optimization Opportunities**:
- Cache frequent queries (Redis)
- Pre-compute product embeddings (already done)
- Batch CNN predictions
- Async Pinecone queries
- Model quantization (CNN)

**Production Deployment**:
```
┌─────────────┐
│Load Balancer│
└──────┬──────┘
       │
   ┌───┴────┬────────┬────────┐
   │        │        │        │
┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌───▼──┐
│App 1│ │App 2│ │App 3│ │App N │
└──┬──┘ └──┬──┘ └──┬──┘ └───┬──┘
   │       │       │        │
   └───────┴───────┴────────┘
           │
    ┌──────┴──────┐
    │   Pinecone   │
    │  (Serverless)│
    └─────────────┘
```

## Security

### Input Validation
- Text queries: Length limits (2-500 chars), XSS pattern detection
- File uploads: Size limits, format validation, MIME type checking
- SQL injection: Not applicable (using Pinecone, not SQL)

### Data Sanitization
- All inputs cleaned before processing
- HTML/script tags stripped
- File paths validated

### Rate Limiting
- Not implemented (should add in production)
- Recommendation: 100 requests/minute per IP

## Monitoring & Observability

### Logging Metrics
- Request latency (processing_time_ms)
- Candidate counts (total_candidates)
- Ranking strategy used
- Error rates and types

### Key Metrics to Track
- Query success rate
- Average response time
- Top-K accuracy (if feedback available)
- Popular queries
- Failed queries (no results)

### Recommended Tools
- **Application**: Datadog, New Relic
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Tracing**: Jaeger, Zipkin
- **Metrics**: Prometheus + Grafana

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | HTML/CSS/JS | User interface |
| **API** | Flask (Python) | REST endpoints |
| **NLP** | sentence-transformers | Text embeddings |
| **Vector DB** | Pinecone | Similarity search |
| **OCR** | EasyOCR | Handwriting recognition |
| **CV** | TensorFlow/Keras | Image classification |
| **Config** | python-dotenv | Environment management |
| **Logging** | Python logging | Structured logs |
| **Testing** | pytest | Unit tests |

## Design Decisions

### Why Pinecone over PostgreSQL with pgvector?
- **Serverless**: No infrastructure management
- **Performance**: Sub-100ms queries at scale
- **Cost**: Pay-per-use, no idle costs
- **Simplicity**: Managed service, automatic scaling

### Why sentence-transformers over OpenAI embeddings?
- **Cost**: Free, run locally
- **Latency**: No API calls
- **Privacy**: Data stays local
- **Control**: Fine-tuning possible

### Why custom CNN over pre-trained models?
- **Educational**: Learning experience
- **Control**: Full architecture control
- **Size**: Small model (< 50MB)
- **Specialization**: Trained on exact product classes

### Why Flask over FastAPI?
- **Simplicity**: Fewer dependencies
- **Familiarity**: More documentation/examples
- **Maturity**: Stable, well-tested
- **Trade-off**: FastAPI would be faster (async support)

## Future Architecture Evolution

### Phase 1: Current State ✅
- Multi-modal search
- Hybrid ranking
- Basic query understanding

### Phase 2: Enhanced Intelligence
- User personalization (history, preferences)
- Collaborative filtering
- A/B testing framework
- Real-time analytics

### Phase 3: Microservices
- Separate services for CNN, OCR, Recommender
- Message queue (RabbitMQ/Kafka)
- API gateway
- Service mesh

### Phase 4: ML Ops
- Model versioning (MLflow)
- Experiment tracking
- Automated retraining
- Model monitoring

## References

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
