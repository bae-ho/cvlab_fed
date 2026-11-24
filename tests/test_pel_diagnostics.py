import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from fed.client import FedAvgClient
from fed.server import FedAvgServer
from utils.pel_diagnostics import gradient_diagnostics, update_diagnostics
from test_fixed_pel import TinyModel


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)

    def test_diagnostic_client_preserves_updates_optimizer_gradients_and_rng(self):
        model = TinyModel()
        bank = torch.randn(3, 4)
        labels = {'Normal': 'normal', 'Fighting': 'fighting', 'Arson': 'arson'}
        loaders = tuple(DataLoader([(torch.randn(8, 4), label, torch.tensor(6))
                                    for _ in range(2)], batch_size=1, shuffle=True)
                        for label in ('Normal', 'Fighting'))
        for enabled in (False, True):
            plain = FedAvgClient(model, 1e-4, loaders, 'ucf', 1, labels, [], .1, 'cpu',
                                 enabled, 'fixed_bank' if enabled else 'legacy', pel_text_bank=bank)
            measured = copy.deepcopy(plain)
            measured.pel_diagnostics = True
            measured.diagnostic_round = 1
            rng = torch.random.get_rng_state()
            expected, loss, counts = plain.train()
            final_rng = torch.random.get_rng_state()
            torch.random.set_rng_state(rng)
            actual, actual_loss, actual_counts = measured.train()
            self.assertEqual(loss, actual_loss)
            self.assertEqual(counts, actual_counts)
            torch.testing.assert_close(final_rng, torch.random.get_rng_state(), rtol=0, atol=0)
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
            for left, right in zip(plain.model.parameters(), measured.model.parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                for key in ('exp_avg', 'exp_avg_sq', 'step'):
                    torch.testing.assert_close(plain.optimizer.state[left][key],
                                               measured.optimizer.state[right][key], rtol=0, atol=0)
            json.dumps(measured.last_diagnostics, allow_nan=False)
            self.assertEqual(measured.last_diagnostics['gradient']['status'],
                             'measured' if enabled else 'pel_disabled')
            torch.testing.assert_close(bank, measured.pel_text_bank, rtol=0, atol=0)
            measured.diagnostic_round = 2
            measured.train()
            self.assertIsNone(measured.last_diagnostics)

    def test_gradient_zero_norm_and_opposite_direction(self):
        model = TinyModel()
        a = model.temporal.weight.sum()
        result = gradient_diagnostics(model, a, -a, .3)['modules']['all_visual']
        self.assertAlmostEqual(result['cosine'], -1)
        self.assertAlmostEqual(result['weighted_pel_to_clasm_norm'], .3)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        result = gradient_diagnostics(model, a * 0, -a * 0, 1)['modules']['all_visual']
        self.assertIsNone(result['cosine'])
        json.dumps(result, allow_nan=False)

    def test_updates_use_deltas_actual_coefficients_and_zero_as_missing(self):
        start = {'temporal.weight': torch.tensor([10., 10.])}
        clients = [{'temporal.weight': torch.tensor([11., 10.])},
                   {'temporal.weight': torch.tensor([9., 10.])}]
        aggregate = {'temporal.weight': torch.tensor([10., 10.])}
        result = update_diagnostics(start, clients, [2, 2], aggregate)
        group = result['modules']['all_shared']
        self.assertEqual(group['retention_ratio'], 0)
        self.assertEqual(group['pairs'][0]['cosine'], -1)
        self.assertIsNone(group['clients'][0]['cosine_with_aggregate'])
        result = update_diagnostics(start, [start, start], [2, 4], start)
        self.assertEqual(result['coefficients'], [1 / 3, 2 / 3])
        self.assertIsNone(result['modules']['all_shared']['retention_ratio'])
        json.dumps(result, allow_nan=False)

    def test_server_writes_diagnostics_without_changing_aggregation(self):
        model = TinyModel(14)
        def loaders(n):
            return tuple(DataLoader([(torch.randn(8, 4), label, torch.tensor(8))
                                     for _ in range(n)], batch_size=1)
                         for label in ('Normal', 'Fighting'))
        data = [loaders(1), loaders(2)]
        servers = [FedAvgServer('ucf', data, None, 2, 1, 1, 1e-4, 'event', [], .1,
                               'cpu', model, pel_diagnostics=enabled) for enabled in (False, True)]
        rng = torch.random.get_rng_state()
        for server in servers:
            torch.random.set_rng_state(rng)
            def evaluate(r):
                server.last_evaluation = {'roc': .5, 'ap': .4}
                return {'ucf': .5}
            with tempfile.TemporaryDirectory() as directory, patch.object(server, 'evaluate', side_effect=evaluate):
                server.train(directory)
                path = Path(directory, 'pel_diagnostics.jsonl')
                self.assertEqual(path.exists(), server.pel_diagnostics)
                if path.exists():
                    record = json.loads(path.read_text())
                    self.assertEqual(record['coefficients'], [1 / 3, 2 / 3])
                    self.assertEqual(record['client_batches'][0]['round'], 1)
        for key in servers[0].global_parameter:
            torch.testing.assert_close(servers[0].global_parameter[key], servers[1].global_parameter[key],
                                       rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
