# Experimentation Notebooks

This folder contains **5 comprehensive Jupyter notebooks** that demonstrate the complete experimentation process for this project.

## Purpose

These notebooks show **how the work was done**, not just the results. They include:
- ✓ Interactive visualizations (charts, confusion matrices, t-SNE plots)
- ✓ Step-by-step explanations with markdown
- ✓ Data exploration with statistical analysis
- ✓ Model training progress with live plots
- ✓ Error analysis with sample images
- ✓ Decision-making process documented

## Notebooks

### 01_data_exploration_cleaning.ipynb
**Based on**: `01_data_exploration_and_cleaning.py` + `02_data_cleaning.py`

**Shows**:
- Raw dataset exploration with visualizations
- Data quality issues identification (special characters, missing values)
- Before/after comparisons with charts
- Statistical summaries and distributions
- Country/product analysis with bar charts
- Complete cleaning workflow

### 02_embedding_generation_analysis.ipynb
**Based on**: `05_generate_embeddings.py` + `06_upload_to_pinecone.py`

**Shows**:
- Sentence-transformers embedding generation
- Similarity metrics comparison (cosine, dot product, euclidean)
- t-SNE visualization of embedding space
- Sample query testing with top results
- Pinecone index setup and vector upload
- Embedding statistics and distributions

### 03_image_scraping_augmentation.ipynb
**Based on**: `14_scrape_products.py` + `17_augment_images.py` + `18_prepare_cnn_data.py`

**Shows**:
- Google Images scraping with Selenium
- Downloaded vs valid images comparison
- Sample scraped images visualization
- Image augmentation process
- Train/val/test split (70/15/15) with charts
- Class distribution analysis

### 04_cnn_training.ipynb
**Based on**: `19_train_cnn.py`

**Shows**:
- CNN architecture design decisions
- Model summary with parameter counts
- Sample training images visualization
- Live training progress (accuracy/loss curves)
- Epoch-by-epoch metrics
- Overfitting analysis
- Before/after comparisons

### 05_model_evaluation.ipynb
**Based on**: `20_evaluate_cnn.py`

**Shows**:
- Overall test set performance
- Confusion matrix heatmap
- Per-class accuracy with color-coding
- Confidence score distributions
- Correct vs incorrect prediction samples with images
- Error analysis (most common mistakes)
- Performance recommendations

## How to Use

1. **Install Jupyter**:
   ```bash
   pip install jupyter notebook
   ```

2. **Launch Jupyter**:
   ```bash
   cd notebook
   jupyter notebook
   ```

3. **Run notebooks in order** (01 → 05)

## Key Differences from .py Scripts

| Python Scripts (.py) | Jupyter Notebooks (.ipynb) |
|---------------------|---------------------------|
| Execute and save results | Show interactive process |
| Output to JSON/PNG files | Visualizations inline |
| No explanations | Markdown explanations |
| Hard to review | Easy to review and understand |
| Results only | Process + Results |

## For Reviewers

These notebooks demonstrate:
- ✓ **Experimentation process**: Not just "here's the result"
- ✓ **Decision-making**: Why cosine similarity? Why 4 conv blocks?
- ✓ **Data understanding**: Distributions, quality issues, cleaning impact
- ✓ **Model insights**: Training curves, confusion patterns, error analysis
- ✓ **Reproducibility**: Step-by-step code with outputs

## Files Generated

Running these notebooks will create:
- `../data/dataset_cleaned.csv` - Cleaned dataset
- `../data/product_embeddings.npy` - Product vectors
- `../data/cnn_train.csv`, `cnn_val.csv`, `cnn_test.csv` - CNN splits
- `../models/cnn_product_classifier.keras` - Trained model
- `../models/training_plots.png` - Training curves
- `../models/confusion_matrix.png` - Confusion matrix visualization
- `../models/test_evaluation.json` - Evaluation metrics

## Notes

- Original Python scripts (01-23) are preserved and still functional
- These notebooks are for **demonstration and documentation**
- The .py scripts are for **production/automation**
- Both serve different purposes and are valuable
