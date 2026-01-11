# Model Pipeline Documentation

## Overview

The model pipeline provides a modular, reusable framework for training, evaluating, and deploying machine learning models. It supports three model types: CNN classifiers, embedding models, and ranking models.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ModelPipeline                           │
│  (Orchestration & experiment tracking)                       │
└───┬─────────────┬─────────────┬─────────────┬──────────────┘
    │             │             │             │
┌───▼──────┐ ┌───▼──────┐ ┌───▼──────┐ ┌───▼──────────┐
│ Trainer  │ │Evaluator │ │ Deployer │ │ExperimentMgr │
└──────────┘ └──────────┘ └──────────┘ └──────────────┘
```

## Components

### 1. ModelPipeline (`services/model_pipeline/pipeline.py`)

**Purpose**: High-level orchestrator for ML workflows

**Key Methods**:

```python
def train(
    model_type: str,           # 'cnn', 'embedding', 'ranker'
    data_path: str,            # Path to training data
    hyperparams: Dict = None,  # Hyperparameter overrides
    experiment_name: str = None # Experiment tracking ID
) -> Dict[str, Any]:
    """
    Train a model with experiment tracking.

    Returns:
        {
            'experiment_id': str,
            'model_path': str,
            'metrics': dict,
            'artifacts': list
        }
    """
```

```python
def evaluate(
    model_path: str,          # Path to trained model
    test_data_path: str,      # Test dataset path
    experiment_id: str = None # Link to training experiment
) -> Dict[str, Any]:
    """
    Evaluate model performance.

    Returns:
        {
            'metrics': {
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1': float
            },
            'confusion_matrix': array,
            'per_class_metrics': dict
        }
    """
```

```python
def deploy(
    model_path: str,          # Model to deploy
    version: str,             # Version tag (e.g., 'v1.0')
    environment: str = 'staging' # 'staging' or 'production'
) -> Dict[str, Any]:
    """
    Deploy model to target environment.

    Returns:
        {
            'status': 'success',
            'endpoint': str,
            'version': str
        }
    """
```

**Usage Example**:

```python
from services.model_pipeline import ModelPipeline

# Initialize pipeline
pipeline = ModelPipeline(
    experiments_dir='experiments',
    models_dir='models'
)

# Train CNN model
result = pipeline.train(
    model_type='cnn',
    data_path='data/cnn_train.csv',
    hyperparams={
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50
    },
    experiment_name='cnn_v2_higher_lr'
)

print(f"Experiment ID: {result['experiment_id']}")
print(f"Model saved to: {result['model_path']}")
print(f"Test accuracy: {result['metrics']['accuracy']:.2%}")

# Evaluate on test set
eval_result = pipeline.evaluate(
    model_path=result['model_path'],
    test_data_path='data/cnn_test.csv',
    experiment_id=result['experiment_id']
)

# Deploy if metrics are good
if eval_result['metrics']['accuracy'] > 0.75:
    deploy_result = pipeline.deploy(
        model_path=result['model_path'],
        version='v2.0',
        environment='production'
    )
    print(f"Deployed to: {deploy_result['endpoint']}")
```

### 2. ModelTrainer (`services/model_pipeline/trainer.py`)

**Purpose**: Model training with different architectures

**Supported Model Types**:

#### 2.1 CNN Classifier

**Configuration**:
```python
hyperparams = {
    'image_size': 128,          # Input dimensions (128×128)
    'batch_size': 32,           # Training batch size
    'epochs': 50,               # Max training epochs
    'learning_rate': 0.001,     # Adam optimizer LR
    'early_stopping_patience': 5 # Stop if no improvement
}
```

**Architecture**:
```python
def _build_cnn_model(num_classes=10):
    model = keras.Sequential([
        # Block 1
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        # Block 2
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        # Block 3
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.4),

        # Block 4
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.4),

        # Dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```

**Training Process**:
1. Load and preprocess images (resize, normalize)
2. Create data generators with augmentation
3. Build model architecture
4. Setup callbacks (EarlyStopping, ModelCheckpoint, CSVLogger)
5. Train model with validation split
6. Save best model based on val_accuracy
7. Log training history and metrics

**Data Augmentation**:
```python
ImageDataGenerator(
    rotation_range=20,        # Rotate ±20°
    width_shift_range=0.2,    # Horizontal shift
    height_shift_range=0.2,   # Vertical shift
    horizontal_flip=True,     # Random flip
    zoom_range=0.2,           # Zoom in/out
    shear_range=0.15,         # Shear transform
    fill_mode='nearest'       # Fill missing pixels
)
```

**Callbacks**:
- **EarlyStopping**: Stops if val_loss doesn't improve for 5 epochs
- **ModelCheckpoint**: Saves best model based on val_accuracy
- **CSVLogger**: Logs metrics to CSV file
- **ReduceLROnPlateau**: Reduces LR if plateau detected

#### 2.2 Embedding Model

**Configuration**:
```python
hyperparams = {
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'max_seq_length': 128,
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 2e-5
}
```

**Training Process**:
1. Load pre-trained sentence-transformer
2. Fine-tune on product descriptions
3. Use triplet loss (anchor, positive, negative)
4. Optimize for cosine similarity
5. Save fine-tuned model

#### 2.3 Ranking Model

**Configuration**:
```python
hyperparams = {
    'num_features': 4,         # similarity, popularity, entity, price
    'hidden_layers': [64, 32],
    'learning_rate': 0.001,
    'epochs': 20
}
```

**Training Process**:
1. Load ranking features (similarity, popularity, etc.)
2. Train neural network for learning-to-rank
3. Optimize for ranking loss (ListNet, RankNet)
4. Save trained ranker

### 3. ModelEvaluator (`services/model_pipeline/evaluator.py`)

**Purpose**: Comprehensive model evaluation with metrics

**Evaluation Metrics**:

#### Classification Metrics
```python
metrics = {
    'accuracy': 0.7452,        # Overall accuracy
    'precision': 0.7389,       # Weighted precision
    'recall': 0.7452,          # Weighted recall
    'f1': 0.7398,             # Weighted F1-score
    'confusion_matrix': array,
    'per_class_metrics': {
        'class_0': {'precision': 0.82, 'recall': 0.79, 'f1': 0.80},
        ...
    }
}
```

#### Ranking Metrics
```python
metrics = {
    'ndcg@10': 0.85,          # Normalized DCG at 10
    'map@10': 0.78,           # Mean average precision at 10
    'mrr': 0.82,              # Mean reciprocal rank
    'precision@1': 0.75,      # Precision at rank 1
    'precision@5': 0.68       # Precision at rank 5
}
```

**Visualization**:
- Confusion matrix heatmap
- Per-class accuracy bar chart
- ROC curves (for binary classification)
- Precision-Recall curves
- Training history plots (loss, accuracy over epochs)

**Example**:
```python
from services.model_pipeline.evaluator import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate CNN
results = evaluator.evaluate(
    model_path='models/cnn_product_classifier.keras',
    test_data='data/cnn_test.csv',
    model_type='cnn'
)

print(f"Test Accuracy: {results['accuracy']:.2%}")
print(f"F1 Score: {results['f1']:.4f}")

# Per-class breakdown
for class_id, metrics in results['per_class_metrics'].items():
    print(f"{class_id}: Precision={metrics['precision']:.2f}, "
          f"Recall={metrics['recall']:.2f}")

# Save confusion matrix plot
evaluator.plot_confusion_matrix(
    results['confusion_matrix'],
    save_path='models/confusion_matrix.png'
)
```

### 4. ModelDeployer (`services/model_pipeline/deployer.py`)

**Purpose**: Model deployment and versioning

**Deployment Strategies**:

#### Canary Deployment
```python
deployer.deploy(
    model_path='models/cnn_v2.keras',
    version='v2.0',
    strategy='canary',
    traffic_percentage=10  # Route 10% traffic to new model
)
```

#### Blue-Green Deployment
```python
deployer.deploy(
    model_path='models/cnn_v2.keras',
    version='v2.0',
    strategy='blue-green',
    environment='production'
)

# After validation, switch traffic
deployer.switch_traffic(version='v2.0')
```

#### Shadow Deployment
```python
deployer.deploy(
    model_path='models/cnn_v2.keras',
    version='v2.0',
    strategy='shadow'  # Run in shadow mode, log predictions
)
```

**Model Versioning**:
```
models/
├── cnn_product_classifier/
│   ├── v1.0/
│   │   ├── model.keras
│   │   ├── weights.h5
│   │   ├── metadata.json
│   │   └── metrics.json
│   ├── v2.0/
│   │   ├── model.keras
│   │   ├── weights.h5
│   │   ├── metadata.json
│   │   └── metrics.json
│   └── production -> v1.0  # Symlink to current production
```

### 5. ExperimentTracker (`services/model_pipeline/experiment_tracker.py`)

**Purpose**: Track experiments, hyperparameters, and results

**Experiment Structure**:
```python
experiment = {
    'id': 'exp_20260111_143025',
    'name': 'cnn_v2_higher_lr',
    'model_type': 'cnn',
    'created_at': '2026-01-11T14:30:25',
    'hyperparameters': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50
    },
    'metrics': {
        'train_accuracy': 0.89,
        'val_accuracy': 0.78,
        'test_accuracy': 0.75
    },
    'artifacts': [
        'models/cnn_v2.keras',
        'experiments/exp_20260111_143025/training_plots.png',
        'experiments/exp_20260111_143025/confusion_matrix.png'
    ],
    'status': 'completed'
}
```

**Usage**:
```python
from services.model_pipeline.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(experiments_dir='experiments')

# Create experiment
experiment = tracker.create_experiment(
    name='cnn_v2_higher_lr',
    model_type='cnn',
    hyperparameters={'learning_rate': 0.001}
)

# Log metrics during training
tracker.log_metric(experiment['id'], 'train_loss', 0.45, epoch=10)
tracker.log_metric(experiment['id'], 'val_accuracy', 0.78, epoch=10)

# Log artifacts
tracker.log_artifact(experiment['id'], 'models/cnn_v2.keras')
tracker.log_artifact(experiment['id'], 'training_plots.png')

# Complete experiment
tracker.complete_experiment(experiment['id'], final_metrics={
    'test_accuracy': 0.75
})

# Compare experiments
best = tracker.get_best_experiment(
    model_type='cnn',
    metric='test_accuracy'
)
print(f"Best experiment: {best['name']} ({best['metrics']['test_accuracy']:.2%})")
```

## Training Workflows

### CNN Training Workflow

```python
from services.model_pipeline import ModelPipeline

# Initialize pipeline
pipeline = ModelPipeline()

# 1. Prepare data
# - Images scraped and organized: data/product_images/{class_name}/
# - CSV files created: data/cnn_train.csv, data/cnn_val.csv, data/cnn_test.csv

# 2. Train model
train_result = pipeline.train(
    model_type='cnn',
    data_path='data/cnn_train.csv',
    hyperparams={
        'image_size': 128,
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001
    },
    experiment_name='cnn_baseline'
)

# 3. Evaluate
eval_result = pipeline.evaluate(
    model_path=train_result['model_path'],
    test_data_path='data/cnn_test.csv',
    experiment_id=train_result['experiment_id']
)

# 4. Deploy if good
if eval_result['metrics']['accuracy'] > 0.70:
    pipeline.deploy(
        model_path=train_result['model_path'],
        version='v1.0',
        environment='production'
    )

    # Update CNNService to load new model
    # Restart application
```

### Embedding Model Training

```python
# 1. Prepare data
# - Product descriptions with similarity labels
# - Format: (anchor, positive, negative) triplets

# 2. Fine-tune sentence-transformer
train_result = pipeline.train(
    model_type='embedding',
    data_path='data/triplet_data.csv',
    hyperparams={
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'epochs': 10,
        'learning_rate': 2e-5
    },
    experiment_name='embedding_finetuned'
)

# 3. Regenerate embeddings for all products
from services.vector_db_service import VectorDBService

vector_service = VectorDBService(
    embedding_model=train_result['model_path']
)
vector_service.reindex_all_products('data/dataset_cleaned.csv')

# 4. A/B test new embeddings
```

### Ranking Model Training

```python
# 1. Collect training data
# - User queries with clicked/purchased products (positive examples)
# - Displayed but not clicked products (negative examples)

# 2. Extract features for each (query, product) pair
# - Similarity score
# - Popularity score
# - Entity match score
# - Price preference score

# 3. Train ranker
train_result = pipeline.train(
    model_type='ranker',
    data_path='data/ranking_training_data.csv',
    hyperparams={
        'num_features': 4,
        'hidden_layers': [64, 32],
        'learning_rate': 0.001
    },
    experiment_name='learned_ranker_v1'
)

# 4. Replace HybridRanker with LearnedRanker
from services.rankers import LearnedRanker

ranker = LearnedRanker(model_path=train_result['model_path'])
```

## Hyperparameter Tuning

### Grid Search

```python
from services.model_pipeline import ModelPipeline

pipeline = ModelPipeline()

param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64],
    'dropout_rate': [0.3, 0.5, 0.7]
}

best_params = None
best_accuracy = 0

for lr in param_grid['learning_rate']:
    for bs in param_grid['batch_size']:
        for dr in param_grid['dropout_rate']:
            result = pipeline.train(
                model_type='cnn',
                data_path='data/cnn_train.csv',
                hyperparams={
                    'learning_rate': lr,
                    'batch_size': bs,
                    'dropout_rate': dr
                },
                experiment_name=f'cnn_lr{lr}_bs{bs}_dr{dr}'
            )

            if result['metrics']['val_accuracy'] > best_accuracy:
                best_accuracy = result['metrics']['val_accuracy']
                best_params = {'lr': lr, 'bs': bs, 'dr': dr}

print(f"Best params: {best_params}")
print(f"Best val accuracy: {best_accuracy:.2%}")
```

### Random Search

```python
import random

def sample_hyperparams():
    return {
        'learning_rate': 10 ** random.uniform(-5, -2),
        'batch_size': random.choice([16, 32, 64]),
        'dropout_rate': random.uniform(0.2, 0.7)
    }

# Try 20 random combinations
for i in range(20):
    params = sample_hyperparams()
    result = pipeline.train(
        model_type='cnn',
        data_path='data/cnn_train.csv',
        hyperparams=params,
        experiment_name=f'cnn_random_{i}'
    )
```

### Bayesian Optimization

```python
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

def objective(params):
    lr, bs, dr = params

    result = pipeline.train(
        model_type='cnn',
        data_path='data/cnn_train.csv',
        hyperparams={
            'learning_rate': lr,
            'batch_size': bs,
            'dropout_rate': dr
        }
    )

    # Minimize negative accuracy (maximize accuracy)
    return -result['metrics']['val_accuracy']

space = [
    Real(1e-5, 1e-2, name='learning_rate', prior='log-uniform'),
    Integer(16, 64, name='batch_size'),
    Real(0.2, 0.7, name='dropout_rate')
]

result = gp_minimize(
    objective,
    space,
    n_calls=30,
    random_state=42
)

print(f"Best params: lr={result.x[0]}, bs={result.x[1]}, dr={result.x[2]}")
print(f"Best val accuracy: {-result.fun:.2%}")
```

## Model Monitoring

### Performance Tracking

```python
from services.model_pipeline import ModelMonitor

monitor = ModelMonitor()

# Log predictions
monitor.log_prediction(
    model_version='v1.0',
    input_data=image,
    prediction='LUNCH BAG WOODLAND',
    confidence=0.87,
    latency_ms=150
)

# Check model drift
drift_report = monitor.check_drift(
    model_version='v1.0',
    reference_data='data/cnn_train.csv',
    current_data='logs/recent_predictions.csv'
)

if drift_report['is_drifting']:
    print("Model drift detected! Consider retraining.")
    print(f"Drift score: {drift_report['drift_score']:.4f}")
```

### A/B Testing

```python
from services.model_pipeline import ABTester

tester = ABTester()

# Setup A/B test
test = tester.create_test(
    name='cnn_v1_vs_v2',
    model_a='models/cnn_v1.keras',
    model_b='models/cnn_v2.keras',
    traffic_split=0.5,  # 50/50 split
    duration_days=7
)

# Route traffic
prediction = tester.predict(
    test_id=test['id'],
    input_data=image
)

# Analyze results
results = tester.analyze_test(test['id'])
print(f"Model A accuracy: {results['model_a']['accuracy']:.2%}")
print(f"Model B accuracy: {results['model_b']['accuracy']:.2%}")
print(f"Winner: {results['winner']}")
```

## Best Practices

### 1. Experiment Tracking
- Always name experiments descriptively
- Log all hyperparameters, even defaults
- Save model artifacts and training plots
- Document experiment goals and findings

### 2. Data Management
- Version training data (DVC, Git LFS)
- Keep train/val/test splits consistent
- Monitor data quality over time
- Track data lineage

### 3. Model Validation
- Use held-out test set only once (final evaluation)
- Cross-validate on training data
- Check for overfitting (train vs val metrics)
- Validate on diverse data (edge cases)

### 4. Reproducibility
- Set random seeds (Python, NumPy, TensorFlow)
- Log environment (Python version, package versions)
- Save exact data splits
- Document preprocessing steps

### 5. Performance Optimization
- Profile training time bottlenecks
- Use mixed precision training (TF AMP)
- Optimize data pipeline (tf.data)
- Consider model quantization for inference

### 6. Model Deployment
- Gradual rollout (canary, blue-green)
- Monitor prediction latency and accuracy
- Have rollback plan ready
- Load test before production

## File Organization

```
models/
├── cnn_product_classifier/
│   ├── v1.0/
│   │   ├── model.keras
│   │   ├── weights.h5
│   │   ├── architecture.json
│   │   ├── class_mapping.json
│   │   └── metadata.json
│   ├── v2.0/
│   └── production -> v1.0
├── embedding_model/
│   ├── v1.0/
│   └── production -> v1.0
└── ranking_model/
    ├── v1.0/
    └── production -> v1.0

experiments/
├── exp_20260110_120530/  # CNN baseline
│   ├── config.json
│   ├── training_history.csv
│   ├── training_plots.png
│   ├── confusion_matrix.png
│   └── metrics.json
├── exp_20260111_143025/  # CNN higher LR
│   └── ...
└── experiments.db  # SQLite database of all experiments

data/
├── cnn_train.csv
├── cnn_val.csv
├── cnn_test.csv
├── product_embeddings.pkl
└── ranking_training_data.csv
```

## References

- [TensorFlow Model Training Best Practices](https://www.tensorflow.org/guide/keras/train_and_evaluate)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Weights & Biases Guide](https://docs.wandb.ai/)
- [Learning to Rank](https://arxiv.org/abs/2201.10545)
