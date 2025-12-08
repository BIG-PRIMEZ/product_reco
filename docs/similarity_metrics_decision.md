# Similarity Metrics Selection

## Decision: Cosine Similarity ✓

We chose **cosine similarity** for the vector database.

## Why Cosine?

**1. Normalized Scores**
- Returns values between 0 and 1
- Easy to interpret (higher = more similar)
- Consistent across all queries

**2. Semantic Similarity**
- Measures angle between vectors, not magnitude
- Perfect for text embeddings
- Focuses on meaning, not length

**3. Standard for Text**
- Recommended by sentence-transformers
- Used by most NLP applications
- Proven for e-commerce search

**4. Our Model Uses It**
- all-MiniLM-L6-v2 is trained with cosine
- Embeddings are normalized for cosine
- Best performance with this metric

## Test Results

Query: "pink lunch bag"

| Metric | Top Result | Score | Correct? |
|--------|-----------|-------|----------|
| **Cosine** | LUNCH BAG PINK POLKADOT | 0.8432 | ✓ |
| Dotproduct | LUNCH BAG PINK POLKADOT | 0.8432 | ✓ |
| Euclidean | LUNCH BAG PINK POLKADOT | 0.3135 | ✓ |

All metrics found the correct product, but cosine provides the clearest scores.

## Alternatives Considered

**Dotproduct**
- Same results as cosine (when vectors normalized)
- Less intuitive scores
- Not rejected: Works fine, but cosine is clearer

**Euclidean Distance**
- Measures geometric distance
- Lower score = more similar (confusing)
- Good for spatial data, not text
- Not rejected: Works, but backward scoring

## Conclusion

Cosine similarity is the right choice because it's standard for text embeddings, gives clear 0-1 scores, and works perfectly with our sentence-transformer model.

**Final Configuration**: Pinecone index using cosine metric with 384-dimensional vectors.
