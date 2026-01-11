"""
ML training pipeline orchestration.

Provides unified interface for training, evaluating, and deploying models.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid

from utils.logger import get_logger
from config import get_pipeline_config
from .trainer import Trainer
from .evaluator import Evaluator

logger = get_logger(__name__)


@dataclass
class Experiment:
    """
    Represents a single training experiment.

    Attributes:
        id: Unique experiment ID
        name: Human-readable experiment name
        model_type: Type of model ('cnn', 'embedding', 'ranker')
        config: Experiment configuration
        created_at: Timestamp
        model_path: Path to saved model
        metrics: Training/evaluation metrics
        status: Experiment status
    """
    id: str
    name: str
    model_type: str
    config: Dict[str, Any]
    created_at: str
    model_path: Optional[str] = None
    metrics: Dict[str, Any] = None
    status: str = 'created'  # created, training, completed, failed

    @property
    def model_dir(self) -> Path:
        """Get model directory for this experiment."""
        config = get_pipeline_config()
        return config.experiment_dir / self.id / 'model'

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class ExperimentTracker:
    """
    Tracks experiments and their results.

    Provides experiment logging, metrics tracking, and comparison.
    """

    def __init__(self, experiment_dir: Path):
        """
        Initialize experiment tracker.

        Args:
            experiment_dir: Directory to store experiments
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ExperimentTracker initialized",
            extra={'experiment_dir': str(self.experiment_dir)}
        )

    def create_experiment(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            config: Experiment configuration

        Returns:
            Experiment object
        """
        experiment_id = str(uuid.uuid4())[:8]

        experiment = Experiment(
            id=experiment_id,
            name=name,
            model_type=config.get('model_type', 'unknown'),
            config=config,
            created_at=datetime.now().isoformat(),
            status='created'
        )

        # Create experiment directory
        exp_dir = self.experiment_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        experiment.model_dir.mkdir(parents=True, exist_ok=True)

        # Save experiment metadata
        self._save_experiment(experiment)

        logger.info(
            f"Created experiment: {name}",
            extra={'experiment_id': experiment_id, 'model_type': experiment.model_type}
        )

        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment object or None if not found
        """
        exp_file = self.experiment_dir / experiment_id / 'experiment.json'

        if not exp_file.exists():
            logger.warning(f"Experiment not found: {experiment_id}")
            return None

        try:
            with open(exp_file, 'r') as f:
                data = json.load(f)
                return Experiment(**data)
        except Exception as e:
            logger.error(f"Error loading experiment: {str(e)}", exc_info=True)
            return None

    def log_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, Any],
        step: Optional[int] = None
    ):
        """
        Log metrics for an experiment.

        Args:
            experiment_id: Experiment ID
            metrics: Metrics dictionary
            step: Optional step/epoch number
        """
        try:
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                logger.error(f"Cannot log metrics for unknown experiment: {experiment_id}")
                return

            # Update experiment metrics
            if experiment.metrics is None:
                experiment.metrics = {}

            if step is not None:
                if 'history' not in experiment.metrics:
                    experiment.metrics['history'] = []
                experiment.metrics['history'].append({'step': step, **metrics})
            else:
                experiment.metrics.update(metrics)

            # Save updated experiment
            self._save_experiment(experiment)

            logger.debug(
                f"Logged metrics for experiment {experiment_id}",
                extra={'metrics': metrics, 'step': step}
            )

        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}", exc_info=True)

    def update_status(self, experiment_id: str, status: str):
        """Update experiment status."""
        try:
            experiment = self.get_experiment(experiment_id)
            if experiment:
                experiment.status = status
                self._save_experiment(experiment)

                logger.info(
                    f"Updated experiment status: {status}",
                    extra={'experiment_id': experiment_id}
                )
        except Exception as e:
            logger.error(f"Error updating status: {str(e)}", exc_info=True)

    def list_experiments(
        self,
        model_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Experiment]:
        """
        List experiments, optionally filtered by model type.

        Args:
            model_type: Filter by model type (optional)
            limit: Maximum number of experiments to return

        Returns:
            List of Experiment objects
        """
        try:
            experiments = []

            for exp_dir in self.experiment_dir.iterdir():
                if exp_dir.is_dir():
                    experiment = self.get_experiment(exp_dir.name)
                    if experiment:
                        if model_type is None or experiment.model_type == model_type:
                            experiments.append(experiment)

            # Sort by creation time (newest first)
            experiments.sort(key=lambda x: x.created_at, reverse=True)

            return experiments[:limit]

        except Exception as e:
            logger.error(f"Error listing experiments: {str(e)}", exc_info=True)
            return []

    def _save_experiment(self, experiment: Experiment):
        """Save experiment metadata to disk."""
        exp_file = self.experiment_dir / experiment.id / 'experiment.json'
        exp_file.parent.mkdir(parents=True, exist_ok=True)

        with open(exp_file, 'w') as f:
            json.dump(experiment.to_dict(), f, indent=2)


class ModelPipeline:
    """
    Orchestrates ML training, evaluation, and deployment workflows.

    Provides standardized interface for:
    - Training models with different configurations
    - Evaluating model performance
    - Versioning and tracking experiments
    - Deploying models to production
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize model pipeline.

        Args:
            config: Pipeline configuration (uses default if None)
        """
        self.config = config or get_pipeline_config()
        self.trainer = Trainer(self.config)
        self.evaluator = Evaluator(self.config)
        self.experiment_tracker = ExperimentTracker(self.config.experiment_dir)

        logger.info("ModelPipeline initialized")

    def train(
        self,
        model_type: str,
        data_path: str,
        hyperparams: Optional[Dict] = None,
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train a model with specified configuration.

        Args:
            model_type: Type of model ('cnn', 'embedding', 'ranker')
            data_path: Path to training data
            hyperparams: Hyperparameter overrides
            experiment_name: Name for experiment tracking

        Returns:
            Dictionary with training results

        Example:
            >>> pipeline = ModelPipeline()
            >>> result = pipeline.train(
            ...     model_type='cnn',
            ...     data_path='data/cnn_train.csv',
            ...     hyperparams={'epochs': 30, 'batch_size': 64}
            ... )
        """
        logger.info(
            f"Starting training: {model_type}",
            extra={'data_path': data_path, 'hyperparams': hyperparams}
        )

        try:
            # Create experiment
            experiment = self.experiment_tracker.create_experiment(
                name=experiment_name or f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    'model_type': model_type,
                    'hyperparams': hyperparams or {},
                    'data_path': data_path
                }
            )

            # Update status
            self.experiment_tracker.update_status(experiment.id, 'training')

            # Train model
            result = self.trainer.train(
                model_type=model_type,
                data_path=data_path,
                hyperparams=hyperparams,
                experiment=experiment
            )

            # Log final metrics
            if 'metrics' in result:
                self.experiment_tracker.log_metrics(
                    experiment_id=experiment.id,
                    metrics=result['metrics']
                )

            # Update experiment with model path
            experiment.model_path = result.get('model_path')
            experiment.status = 'completed'
            self.experiment_tracker._save_experiment(experiment)

            logger.info(
                f"Training completed successfully",
                extra={
                    'experiment_id': experiment.id,
                    'model_path': result.get('model_path')
                }
            )

            return {
                'experiment_id': experiment.id,
                'model_path': result.get('model_path'),
                'metrics': result.get('metrics', {}),
                'training_time': result.get('training_time', 0)
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)

            # Mark experiment as failed
            if 'experiment' in locals():
                self.experiment_tracker.update_status(experiment.id, 'failed')
                self.experiment_tracker.log_metrics(
                    experiment_id=experiment.id,
                    metrics={'error': str(e)}
                )

            return {
                'error': str(e),
                'success': False
            }

    def evaluate(
        self,
        model_path: str,
        test_data_path: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model.

        Args:
            model_path: Path to trained model
            test_data_path: Path to test data
            metrics: List of metrics to compute

        Returns:
            Dictionary with evaluation results

        Example:
            >>> pipeline = ModelPipeline()
            >>> results = pipeline.evaluate(
            ...     model_path='models/cnn_model.keras',
            ...     test_data_path='data/cnn_test.csv',
            ...     metrics=['accuracy', 'precision', 'recall']
            ... )
        """
        logger.info(
            f"Evaluating model",
            extra={'model_path': model_path, 'test_data_path': test_data_path}
        )

        try:
            results = self.evaluator.evaluate(
                model_path=model_path,
                test_data_path=test_data_path,
                metrics=metrics or ['accuracy', 'precision', 'recall', 'f1']
            )

            logger.info(
                f"Evaluation completed",
                extra={'metrics': results.get('metrics', {})}
            )

            return results

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'success': False
            }

    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare

        Returns:
            Comparison results with metrics and rankings

        Example:
            >>> comparison = pipeline.compare_experiments(['abc123', 'def456', 'ghi789'])
            >>> print(comparison['best_model'])
            'abc123'
        """
        logger.info(f"Comparing {len(experiment_ids)} experiments")

        try:
            experiments = [
                self.experiment_tracker.get_experiment(exp_id)
                for exp_id in experiment_ids
            ]

            # Filter out None (experiments that don't exist)
            experiments = [e for e in experiments if e is not None]

            if not experiments:
                logger.warning("No valid experiments found for comparison")
                return {'error': 'No valid experiments found'}

            # Extract metrics for comparison
            comparison_data = []
            for exp in experiments:
                metrics = exp.metrics or {}
                comparison_data.append({
                    'id': exp.id,
                    'name': exp.name,
                    'model_type': exp.model_type,
                    'created_at': exp.created_at,
                    'metrics': metrics
                })

            # Find best model (highest accuracy/f1)
            best_model_id = None
            best_score = -1

            for data in comparison_data:
                score = data['metrics'].get('accuracy', data['metrics'].get('f1', 0))
                if score > best_score:
                    best_score = score
                    best_model_id = data['id']

            logger.info(
                f"Comparison complete, best model: {best_model_id}",
                extra={'best_score': best_score}
            )

            return {
                'experiments': comparison_data,
                'best_model': best_model_id,
                'best_score': best_score
            }

        except Exception as e:
            logger.error(f"Comparison failed: {str(e)}", exc_info=True)
            return {'error': str(e)}

    def deploy(
        self,
        experiment_id: str,
        deployment_target: str = 'production'
    ) -> Dict[str, Any]:
        """
        Deploy a trained model to production.

        Args:
            experiment_id: Experiment containing model to deploy
            deployment_target: 'production', 'staging', 'canary'

        Returns:
            Deployment information

        Example:
            >>> deployment = pipeline.deploy('abc123', 'production')
            >>> print(deployment['deployment_id'])
        """
        logger.info(
            f"Deploying experiment {experiment_id} to {deployment_target}"
        )

        try:
            experiment = self.experiment_tracker.get_experiment(experiment_id)

            if not experiment:
                raise ValueError(f"Experiment not found: {experiment_id}")

            if not experiment.model_path or not Path(experiment.model_path).exists():
                raise ValueError(f"Model file not found: {experiment.model_path}")

            # Copy model to deployment directory
            deployment_dir = self.config.model_dir / deployment_target
            deployment_dir.mkdir(parents=True, exist_ok=True)

            model_filename = f"{experiment.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
            deployment_path = deployment_dir / model_filename

            shutil.copy2(experiment.model_path, deployment_path)

            # Create deployment record
            deployment_id = str(uuid.uuid4())[:8]
            deployment_record = {
                'deployment_id': deployment_id,
                'experiment_id': experiment_id,
                'model_path': str(deployment_path),
                'target': deployment_target,
                'deployed_at': datetime.now().isoformat(),
                'status': 'active',
                'model_type': experiment.model_type,
                'metrics': experiment.metrics
            }

            # Save deployment record
            deployment_file = deployment_dir / f"deployment_{deployment_id}.json"
            with open(deployment_file, 'w') as f:
                json.dump(deployment_record, f, indent=2)

            logger.info(
                f"Deployment successful",
                extra={
                    'deployment_id': deployment_id,
                    'model_path': str(deployment_path)
                }
            )

            return deployment_record

        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}", exc_info=True)
            return {'error': str(e), 'success': False}
