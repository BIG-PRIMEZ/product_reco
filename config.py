"""
Application configuration management.

Centralized configuration for all services, with environment variable support.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AppConfig:
    """Main application configuration."""

    # Environment
    env: str = os.getenv('ENVIRONMENT', 'development')
    debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'

    # Logging
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_dir: Optional[str] = os.getenv('LOG_DIR', 'logs')
    json_logs: bool = os.getenv('JSON_LOGS', 'false').lower() == 'true'

    # Flask
    flask_port: int = int(os.getenv('FLASK_PORT', '8080'))
    flask_host: str = os.getenv('FLASK_HOST', '0.0.0.0')

    # Recommender settings
    use_new_recommender: bool = os.getenv('USE_NEW_RECOMMENDER', 'true').lower() == 'true'
    default_ranking_strategy: str = os.getenv('RANKING_STRATEGY', 'hybrid')
    default_top_k: int = int(os.getenv('DEFAULT_TOP_K', '10'))
    min_similarity_score: float = float(os.getenv('MIN_SIMILARITY_SCORE', '0.3'))

    # Query processing
    enable_query_expansion: bool = os.getenv('ENABLE_QUERY_EXPANSION', 'true').lower() == 'true'
    enable_spell_correction: bool = os.getenv('ENABLE_SPELL_CORRECTION', 'false').lower() == 'true'
    max_expanded_queries: int = int(os.getenv('MAX_EXPANDED_QUERIES', '3'))

    # LLM Integration (optional)
    enable_llm: bool = os.getenv('ENABLE_LLM', 'false').lower() == 'true'
    llm_provider: str = os.getenv('LLM_PROVIDER', 'openai')  # 'openai', 'anthropic', 'local'
    llm_api_key: Optional[str] = os.getenv('LLM_API_KEY')
    llm_model: str = os.getenv('LLM_MODEL', 'gpt-4o-mini')

    # Pinecone
    pinecone_api_key: Optional[str] = os.getenv('PINECONE_API_KEY')
    pinecone_index_name: str = os.getenv('PINECONE_INDEX_NAME', 'ecommerce-products')

    # Model paths
    model_dir: Path = Path(os.getenv('MODEL_DIR', 'models'))
    data_dir: Path = Path(os.getenv('DATA_DIR', 'data'))
    experiment_dir: Path = Path(os.getenv('EXPERIMENT_DIR', 'experiments'))

    # Caching
    enable_cache: bool = os.getenv('ENABLE_CACHE', 'true').lower() == 'true'
    cache_ttl_seconds: int = int(os.getenv('CACHE_TTL_SECONDS', '3600'))

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure directories exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        if self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class RankingConfig:
    """Configuration for ranking strategies."""

    # Hybrid ranker weights
    similarity_weight: float = float(os.getenv('RANKING_SIMILARITY_WEIGHT', '0.5'))
    popularity_weight: float = float(os.getenv('RANKING_POPULARITY_WEIGHT', '0.2'))
    entity_match_weight: float = float(os.getenv('RANKING_ENTITY_MATCH_WEIGHT', '0.2'))
    price_weight: float = float(os.getenv('RANKING_PRICE_WEIGHT', '0.1'))

    # Result diversification
    enable_diversification: bool = os.getenv('ENABLE_DIVERSIFICATION', 'true').lower() == 'true'
    diversity_lambda: float = float(os.getenv('DIVERSITY_LAMBDA', '0.5'))  # 0=no diversity, 1=max diversity

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = (
            self.similarity_weight +
            self.popularity_weight +
            self.entity_match_weight +
            self.price_weight
        )
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Ranking weights must sum to 1.0, got {total}")


@dataclass
class PipelineConfig:
    """Configuration for ML training pipeline."""

    # Directories
    data_dir: Path = Path(os.getenv('DATA_DIR', 'data'))
    model_dir: Path = Path(os.getenv('MODEL_DIR', 'models'))
    experiment_dir: Path = Path(os.getenv('EXPERIMENT_DIR', 'experiments'))

    # Training defaults
    default_epochs: int = int(os.getenv('TRAIN_EPOCHS', '50'))
    default_batch_size: int = int(os.getenv('TRAIN_BATCH_SIZE', '32'))
    default_learning_rate: float = float(os.getenv('TRAIN_LEARNING_RATE', '0.001'))

    # CNN architecture defaults
    cnn_input_shape: tuple = (128, 128, 3)
    cnn_conv_blocks: list = field(default_factory=lambda: [
        {'filters': 32, 'kernel_size': 3},
        {'filters': 64, 'kernel_size': 3},
        {'filters': 128, 'kernel_size': 3},
        {'filters': 128, 'kernel_size': 3}
    ])
    cnn_dense_units: int = 128
    cnn_dropout_rate: float = 0.5
    cnn_num_classes: int = 10

    # Experiment tracking
    track_experiments: bool = os.getenv('TRACK_EXPERIMENTS', 'true').lower() == 'true'
    save_checkpoints: bool = os.getenv('SAVE_CHECKPOINTS', 'true').lower() == 'true'

    # Early stopping
    early_stopping_patience: int = int(os.getenv('EARLY_STOPPING_PATIENCE', '5'))
    early_stopping_min_delta: float = float(os.getenv('EARLY_STOPPING_MIN_DELTA', '0.001'))

    def __post_init__(self):
        """Ensure directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instances
app_config = AppConfig()
ranking_config = RankingConfig()
pipeline_config = PipelineConfig()


def get_config() -> AppConfig:
    """Get the global application configuration."""
    return app_config


def get_ranking_config() -> RankingConfig:
    """Get the ranking configuration."""
    return ranking_config


def get_pipeline_config() -> PipelineConfig:
    """Get the pipeline configuration."""
    return pipeline_config
