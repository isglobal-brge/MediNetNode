"""
Generic ML Flower Client.

This module provides a generic Flower client for federated machine learning
algorithms. It uses the Strategy pattern to delegate algorithm-specific logic
to FederatedMLAlgorithm implementations.

The client is algorithm-agnostic and can work with any algorithm that
implements the FederatedMLAlgorithm interface (FedSVM, FedRandomForest, etc.)
"""

import numpy as np
from typing import Tuple
from flwr.client import NumPyClient
from .utils import update_training_progress, fail_training_session
from .algorithms.base import FederatedMLAlgorithm


class MLFlowerClient(NumPyClient):
    """
    Generic ML client for Flower federated learning.

    This client delegates all algorithm-specific operations to a
    FederatedMLAlgorithm instance, making it completely algorithm-agnostic.

    Architecture:
        MLFlowerClient (orchestration) → FederatedMLAlgorithm (algorithm logic)

    Supported algorithms:
        - FedSVM (Support Vector Machine)
        - Future: FedRandomForest, FedKNN, FedNaiveBayes, etc.

    Usage:
        from .algorithms import get_algorithm

        AlgorithmClass = get_algorithm('fedsvm')
        algorithm = AlgorithmClass(X_train, y_train, config)

        client = MLFlowerClient(
            algorithm_instance=algorithm,
            validation_data=(X_val, y_val),
            ...
        )
    """

    def __init__(self, algorithm_instance: FederatedMLAlgorithm,
                 validation_data: Tuple[np.ndarray, np.ndarray],
                 model_json, training_session, client_ip, table_name,
                 current_process, partition_id=0):
        """
        Initialize generic ML Flower client.

        Args:
            algorithm_instance: Instance of FederatedMLAlgorithm subclass
                               (e.g., FedSVMAlgorithm, FedRandomForestAlgorithm)
            validation_data: Tuple of (X_val, y_val) as numpy arrays
            model_json: Full model configuration JSON
            training_session: Django TrainingSession instance for tracking
            client_ip: Client IP address
            table_name: Dataset table name/ID
            current_process: psutil Process for resource monitoring
            partition_id: Client partition ID (default: 0)
        """
        # Store algorithm instance (Strategy pattern)
        self.algorithm = algorithm_instance

        # Validation data
        self.X_val, self.y_val = validation_data

        # Client metadata
        self.partition_id = partition_id
        self.assigned_client_id = None

        # Configuration and tracking
        self.model_json = model_json
        self.training_session = training_session
        self.client_ip = client_ip
        self.table_name = table_name
        self.current_process = current_process

        # Get algorithm info
        algo_info = self.algorithm.get_model_info()

        print(f"\n{'='*60}")
        print(f"[INFO] MLFlowerClient Initialized")
        print(f"{'='*60}")
        print(f"   Algorithm: {algo_info['algorithm']}")
        print(f"   Partition ID: {partition_id}")
        print(f"   Training samples: {algo_info['n_train_samples']}")
        print(f"   Validation samples: {len(self.X_val)}")
        print(f"   Features: {algo_info['n_features']}")
        print(f"{'='*60}\n")

    def set_client_id(self, client_id):
        """
        Assign client ID from server configuration.

        Args:
            client_id: Unique client identifier
        """
        self.assigned_client_id = client_id
        print(f"[INFO] ML CLIENT_ID_SET: {client_id}")

    def get_parameters(self, config):
        """
        Get model parameters (delegates to algorithm).

        Args:
            config: Configuration dictionary from server

        Returns:
            List of numpy arrays representing model state
        """
        print(f"[ML CLIENT] get_parameters called with config: {config}")
        params = self.algorithm.get_parameters()
        print(f"[ML CLIENT] get_parameters returning {len(params)} arrays")
        return params

    def fit(self, parameters, config):
        """
        Train model for one federated round (delegates to algorithm).

        Workflow:
        1. Check for convergence flag from server
        2. If converged: Return current parameters without training (no-op)
        3. If not converged: Delegate training to algorithm.fit()
        4. Update training progress tracking
        5. Prepare metrics for server
        6. Return updated parameters

        Args:
            parameters: Model parameters from server (aggregated from other clients)
            config: Training configuration from server (includes 'converged' flag)

        Returns:
            Tuple of (updated_parameters, num_examples, metrics_dict)
        """
        try:
            print(f"\n{'='*70}")
            print(f"[ML CLIENT] fit() called")
            print(f"[ML CLIENT] Partition ID: {self.partition_id}")
            print(f"[ML CLIENT] Config: {config}")
            print(f"[ML CLIENT] Parameters received: {len(parameters) if parameters else 0} arrays")
            if parameters:
                print(f"[ML CLIENT] Parameter shapes: {[p.shape if hasattr(p, 'shape') else 'no shape' for p in parameters]}")
            print(f"{'='*70}")

            # Determine current round number
            if self.training_session:
                current_round = self.training_session.current_round + 1
                print(f"[INFO] Round {current_round} (from persistent session state)")
            else:
                current_round = getattr(self, '_round_counter', 0) + 1
                setattr(self, '_round_counter', current_round)
                print(f"[INFO] Round {current_round} (local fallback counter)")

            # CHECK CONVERGENCE FLAG
            converged = config.get("converged", 0)

            if converged:
                print(f"\n{'='*70}")
                print(f"🏁 CONVERGENCE DETECTED - Round {current_round}")
                print(f"{'='*70}")
                print(f"   Server signaled convergence - skipping training (no-op)")
                print(f"   Returning current parameters without modification")
                print(f"{'='*70}\n")

                # Return current parameters without training
                current_params = self.algorithm.get_parameters()
                num_examples = max(len(self.algorithm.X_train), 1)

                # Prepare convergence metrics
                convergence_metrics = {
                    "converged": True,
                    "client_name": f"ML_Client_{self.partition_id}",
                    "client_ip": self.client_ip,
                    "dataset_name": self.table_name,
                    "client_id": self.assigned_client_id,
                    "train_samples": len(self.algorithm.X_train),
                    "algorithm": type(self.algorithm).__name__,
                    "message": "Training converged - no-op round"
                }

                return current_params, num_examples, convergence_metrics

            # NORMAL TRAINING: Algorithm has not converged
            print(f"[ML CLIENT] Training in progress - executing fit()")

            # [INIT] DELEGATE to algorithm
            updated_params, metrics = self.algorithm.fit(parameters)

            # Prepare metrics for tracking
            round_metrics = {
                'loss': float(metrics.get('loss', 0.0)),
                'accuracy': float(metrics.get('accuracy', 0.0)),
                'precision': float(metrics.get('precision', 0.0)),
                'recall': float(metrics.get('recall', 0.0)),
                'f1': float(metrics.get('f1', 0.0))
            }

            # Update training progress in Django
            update_training_progress(
                self.training_session,
                current_round,
                self.current_process,
                round_metrics
            )

            # Prepare full metrics for Flower server
            full_metrics = {
                **metrics,
                "client_name": f"ML_Client_{self.partition_id}",
                "client_ip": self.client_ip,
                "dataset_name": self.table_name,
                "client_id": self.assigned_client_id,
                "train_samples": len(self.algorithm.X_train),
                "algorithm": type(self.algorithm).__name__
            }

            # Log summary
            print(f"\n[INFO] Round {current_round} Summary:")
            print(f"   Accuracy:  {metrics.get('accuracy', 0):.4f}")
            print(f"   Loss:      {metrics.get('loss', 0):.4f}")
            print(f"   Precision: {metrics.get('precision', 0):.4f}")
            print(f"   Recall:    {metrics.get('recall', 0):.4f}")
            print(f"   F1 Score:  {metrics.get('f1', 0):.4f}")
            print(f"{'='*70}\n")

            # Return parameters weighted by number of training samples
            num_examples = max(len(self.algorithm.X_train), 1)

            return updated_params, num_examples, full_metrics

        except Exception as e:
            print(f"\n{'='*70}")
            print(f"[ERROR] ERROR IN ML FIT")
            print(f"{'='*70}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*70}\n")

            # Mark training session as failed
            fail_training_session(
                self.training_session,
                str(e),
                traceback.format_exc()
            )

            # Return empty parameters on error
            return parameters, 0, {}

    def evaluate(self, parameters, config):
        """
        Evaluate model on validation data (delegates to algorithm).

        Args:
            parameters: Model parameters from server
            config: Evaluation configuration from server

        Returns:
            Tuple of (loss, num_examples, metrics_dict)
        """
        try:
            print(f"\n{'='*70}")
            print(f"[INFO] ML CLIENT {self.partition_id} - EVALUATION")
            print(f"{'='*70}")

            # [INIT] DELEGATE to algorithm
            loss, accuracy = self.algorithm.evaluate(
                parameters,
                self.X_val,
                self.y_val
            )

            print(f"[OK] Evaluation Results:")
            print(f"   Loss:     {loss:.4f}")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Samples:  {len(self.X_val)}")
            print(f"{'='*70}\n")

            metrics = {
                "accuracy": float(accuracy),
                "loss": float(loss)
            }

            return float(loss), len(self.X_val), metrics

        except Exception as e:
            print(f"\n{'='*70}")
            print(f"[ERROR] ERROR IN ML EVALUATE")
            print(f"{'='*70}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*70}\n")

            # Return default values on error
            return 0.0, len(self.X_val), {"accuracy": 0.0, "loss": 0.0}
