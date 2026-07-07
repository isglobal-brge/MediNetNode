import sys
sys.modules.setdefault('magic', None)

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestMLClientRecordsEpsilon:

    def test_fit_includes_privacy_epsilon_in_metrics(self):
        from api.federated.ml_client import MLFlowerClient

        algorithm = MagicMock()
        algorithm.fit.return_value = (
            [np.array([1.0])],
            {'loss': 0.5, 'accuracy': 0.8, 'precision': 0.8, 'recall': 0.8, 'f1': 0.8},
        )
        algorithm.get_parameters.return_value = [np.array([1.0])]

        session = MagicMock()
        session.current_round = 0
        session.total_rounds = 3

        client = MLFlowerClient(
            algorithm_instance=algorithm,
            validation_data=(np.array([[1, 2]]), np.array([1])),
            model_json={
                'model': {'metadata': {'model_type': 'ml'},
                          'dataset': {'selected_datasets': [{'dataset_id': 1}]},
                          'training': {'dp': {'noise_multiplier': 1.1}}},
                'train': {'rounds': 3, 'epochs': 1, 'batch_size': 32},
                'federated': {'name': 'FedAvg', 'parameters': {}},
            },
            training_session=session,
            client_ip='127.0.0.1',
            table_name=1,
            current_process=MagicMock(),
        )

        with patch('api.federated.ml_client.update_training_progress') as mock_update:
            params, n, metrics = client.fit([np.array([1.0])], {})

        call_kwargs = mock_update.call_args
        if call_kwargs.kwargs:
            round_metrics = call_kwargs.kwargs.get('round_metrics')
        else:
            round_metrics = call_kwargs.args[3] if len(call_kwargs.args) > 3 else None

        assert round_metrics is not None, "round_metrics not passed to update_training_progress"
        assert 'privacy_epsilon' in round_metrics
        assert round_metrics['privacy_epsilon'] > 0
