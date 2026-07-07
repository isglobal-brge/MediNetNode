import sys
sys.modules.setdefault('magic', None)

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestDPParamVerification:

    def test_fit_aborts_if_noise_multiplier_tampered(self):
        """If Opacus uses a different noise_multiplier than config, abort training."""
        from api.federated.dl_client import DLFlowerClient

        net = MagicMock()
        net.state_dict.return_value = {}

        session = MagicMock()
        session.current_round = 0
        session.total_rounds = 3

        model_json = {
            'model': {
                'metadata': {'model_type': 'dl'},
                'dataset': {'selected_datasets': [{'dataset_id': 1}]},
                'training': {
                    'optimizer': {'type': 'Adam', 'learning_rate': 0.001},
                    'dp': {'noise_multiplier': 1.1, 'max_grad_norm': 1.0},
                }
            },
            'train': {'rounds': 3, 'epochs': 1, 'batch_size': 32},
            'federated': {'name': 'FedAvg', 'parameters': {}},
        }

        client = DLFlowerClient(
            net=net,
            trainloader=MagicMock(),
            valloader=MagicMock(),
            testloader=MagicMock(),
            model_json=model_json,
            training_session=session,
            client_ip='127.0.0.1',
            table_name=1,
            device='cpu',
            current_process=MagicMock(),
        )

        with patch('api.federated.dl_client.set_parameters'), \
             patch('api.federated.dl_client.train') as mock_train, \
             patch('api.federated.dl_client.fail_training_session') as mock_fail:

            # train() returns: loss, accuracy, precision, recall, f1, epsilon, actual_noise
            # actual_noise=0.5 but config says 1.1 — tampered!
            mock_train.return_value = (0.5, 0.8, 0.8, 0.8, 0.8, 0.3, 0.5)

            client.fit([np.zeros(1)], {})

            mock_fail.assert_called_once()
