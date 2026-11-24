import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from test_fixed_pel import TinyModel
from utils.fixed_pel import fixed_bank_pel_loss
from utils.pel_experiments import (scheduled_weight, setup_pel_experiment, prepare_pel_round,
                                   combine_pel, backward_pel, validate_pel_cli)
from utils.config import parser
from scripts.prepare_pel_experiments import commands
from scripts.summarize_pel_runs import summarize


class TinyShared(TinyModel):
    local_param_policy = 'none'
    def forward(self, visual, prompts, lengths, return_pel4vad_aux=False, return_contrastive=False):
        return super().forward(visual, prompts, lengths, return_pel4vad_aux)
    def shared_parameters(self):
        return list(self.parameters())
    def personalized_parameters(self):
        return []
    def get_shared_state_dict(self):
        return {n: p.detach().clone() for n, p in self.named_parameters()}
    def get_personalized_state_dict(self):
        return {}
    def load_shared_state_dict(self, state):
        self.load_state_dict(state)
    def _is_local_param(self, name):
        return False


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)

    def test_schedule(self):
        values = [scheduled_weight(.3, r, 3, 3) for r in range(1, 8)]
        torch.testing.assert_close(torch.tensor(values), torch.tensor([0, 0, 0, .1, .2, .3, .3]))
        for r in range(1, 21):
            self.assertEqual(scheduled_weight(1, r), 1)
        with self.assertRaises(ValueError):
            scheduled_weight(1, 0)

    def test_warmup_matches_baseline_and_does_not_decay_projection(self):
        from fed.client import FedAvgClient
        model = TinyModel()
        model.pel_projection = nn.Linear(4, 4, bias=False)
        bank = torch.randn(3, 4)
        labels = {'Normal': 'normal', 'Fighting': 'fighting', 'Arson': 'arson'}
        data = tuple(DataLoader([(torch.randn(8, 4), label, torch.tensor(8))], batch_size=1)
                     for label in ('Normal', 'Fighting'))
        baseline = FedAvgClient(model, 1e-4, data, 'ucf', 1, labels, [], .1, 'cpu')
        candidate = FedAvgClient(model, 1e-4, data, 'ucf', 1, labels, [], .1, 'cpu', True, 'fixed_bank',
                                 pel_text_bank=bank, pel_options={'warmup_rounds': 3, 'ramp_rounds': 3})
        rng = torch.random.get_rng_state()
        expected = baseline.train()[0]
        torch.random.set_rng_state(rng)
        actual = candidate.train()[0]
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
        self.assertEqual(candidate.active_pel_weight, 0)
        self.assertIsNone(candidate.model.pel_projection.weight.grad)

    def test_uniform_padding_and_class_balancing(self):
        features, logits, bank = torch.randn(3, 6, 4), torch.randn(3, 6, 3), torch.randn(3, 4)
        lengths, indices = torch.tensor([6, 4, 3]), torch.tensor([0, 1, 1])
        def loss(h, scores, balance='none'):
            return fixed_bank_pel_loss(h, scores, lengths, bank, indices,
                                       pooling='uniform', class_balance=balance)
        expected = loss(features, logits)
        changed = features.clone()
        for row, length in enumerate(lengths):
            changed[row, length:] = 1e5
        torch.testing.assert_close(expected, loss(changed, logits * 100))
        h = features.requires_grad_()
        balanced = loss(h, logits, 'batch')
        balanced.backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertFalse(torch.isclose(expected, balanced))

    def test_last_block_routing_retains_base_gradients(self):
        model = nn.Module()
        model.temporal = nn.Module()
        model.temporal.resblocks = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        model.frame_position_embeddings = nn.Embedding(2, 4)
        model.pel_projection = nn.Linear(4, 4, bias=False)
        client = SimpleNamespace(model=model, pel4vad_weight=.3)
        setup_pel_experiment(client, {'gradient_scope': 'last_block'})
        prepare_pel_round(client)
        h = model.temporal.resblocks(model.frame_position_embeddings(torch.tensor([0, 1])))
        base, pel = h.square().sum(), model.pel_projection(h).sum()
        params = list(model.parameters())
        base_grad = torch.autograd.grad(base, params, retain_graph=True, allow_unused=True)
        pel_grad = torch.autograd.grad(pel, params, retain_graph=True, allow_unused=True)
        total = combine_pel(client, base, pel)
        backward_pel(client, total)
        allowed = {id(p) for p in model.temporal.resblocks[-1].parameters()} | {id(p) for p in model.pel_projection.parameters()}
        for p, a, b in zip(params, base_grad, pel_grad):
            expected = torch.zeros_like(p) if a is None else a.clone()
            if id(p) in allowed and b is not None:
                expected += .3 * b
            torch.testing.assert_close(p.grad, expected)

    def test_projection_actual_model_rng_logits_and_checkpoint(self):
        from utils.model import Model
        class FakeClip(nn.Module):
            dtype = torch.float32
            def __init__(self):
                super().__init__()
                class TokenEmbedding(nn.Embedding):
                    def forward(self, tokens):
                        return super().forward(tokens % self.num_embeddings)
                self.token_embedding = TokenEmbedding(128, 512)
                self.logit_scale = nn.Parameter(torch.tensor(1.))
            def encode_text_original(self, tokens):
                return self.token_embedding(tokens).mean(1)
            def encode_text(self, embeddings, tokens):
                return embeddings.mean(1)
        clip = FakeClip()
        rng = torch.random.get_rng_state()
        models = []
        for projection in (False, True):
            torch.random.set_rng_state(rng)
            with patch('utils.model.clip.load', return_value=(copy.deepcopy(clip), None)):
                models.append(Model(512, 256, 10, 10, 512, 1, 1, 8, 'cpu',
                                    personalization_type='low_rank', pel_projection=projection))
            if not projection:
                expected_rng = torch.random.get_rng_state()
            else:
                torch.testing.assert_close(expected_rng, torch.random.get_rng_state(), rtol=0, atol=0)
        x, lengths = torch.randn(2, 256, 512), torch.tensor([256, 200])
        for model in models:
            model.eval()
        torch.testing.assert_close(models[0](x, ['normal', 'fighting'], lengths),
                                   models[1](x, ['normal', 'fighting'], lengths), rtol=0, atol=0)
        self.assertIn('pel_projection.weight', models[1].get_shared_state_dict())
        self.assertNotIn('pel_projection.weight', models[1].get_local_state_dict())
        local = {key: value.clone() for key, value in models[1].get_local_state_dict().items()}
        self.assertTrue(local)
        shared = models[1].get_shared_state_dict()
        shared['pel_projection.weight'] += .1
        models[1].load_shared_state_dict(shared)
        for key, value in local.items():
            torch.testing.assert_close(value, models[1].get_local_state_dict()[key], rtol=0, atol=0)
        models[1].load_state_dict(models[1].state_dict(), strict=True)

    def test_cli_rejects_implicit_integration_and_legacy_ablations(self):
        with self.assertRaises(ValueError):
            validate_pel_cli(parser.parse_args(['--pel_pooling', 'uniform']))
        for method in ('fednova', 'fedpgp'):
            argv = ['--use_pel4vad_contrastive', '--pel4vad_variant', 'fixed_bank', '--aggregation_method', method]
            with self.assertRaises(ValueError):
                validate_pel_cli(parser.parse_args(argv))
            validate_pel_cli(parser.parse_args(argv + ['--pel_allow_aggregation_integration']))

    def test_integration_fixed_bank_and_fednova_steps_preserved(self):
        from fed.server import FedStrategyServer, FedPGPServer
        from fed.aggregation import fednova_delta
        model = TinyShared(14)
        def loaders(n):
            return tuple(DataLoader([(torch.randn(8, 4), label, torch.tensor(8)) for _ in range(n)], batch_size=1)
                         for label in ('Normal', 'Fighting'))
        data, bank = [loaders(1), loaders(2)], torch.randn(14, 4)
        for kind in ('fednova', 'fedpgp'):
            with patch('fed.server.build_text_bank', return_value=(bank, {'classes': list(range(14))})):
                cls = FedStrategyServer if kind == 'fednova' else FedPGPServer
                kwargs = {'aggregation_method': kind} if kind == 'fednova' else {}
                server = cls('ucf', data, None, 2, 1, 1, 1e-4, 'event', [], .1, 'cpu', model,
                             use_pel4vad_contrastive=True, pel4vad_variant='fixed_bank', **kwargs)
            self.assertIs(server.clients[0].pel_text_bank, server.clients[1].pel_text_bank)
            server.pel_experiment_logging = True
            start = model.get_shared_state_dict()
            def evaluate(r):
                server.last_evaluation = {'roc': .6, 'ap': .4}
                return {'ucf': .6}
            with tempfile.TemporaryDirectory() as directory, patch.object(server, 'evaluate', side_effect=evaluate), \
                    patch.object(server, 'evaluate_personalized', return_value={}):
                server.train(directory)
                row = json.loads(Path(directory, 'round_metrics.jsonl').read_text())
                self.assertEqual(row['local_steps'], [1, 2])
                self.assertEqual(row['aggregation_batch_counts'], [2, 4])
                self.assertTrue(Path(directory, 'final_model.pth').exists())
            if kind == 'fednova':
                delta = fednova_delta(start, server.local_weights, [1, 2], [2, 4])
                for key in start:
                    torch.testing.assert_close(server.global_parameter[key], start[key] + delta[key])

    def _integration_argv(self, *extra):
        return ['--dataset', 'ucf', '--use_pel4vad_contrastive', '--pel4vad_variant', 'fixed_bank',
                '--pel_allow_aggregation_integration', *extra]

    def test_aggregation_axes_need_the_opt_in(self):
        """Every aggregation axis is reachable with the flag and blocked without it."""
        for axis in (('--aggregation_method', 'fednova'), ('--server_optimizer', 'momentum'),
                     ('--personalization_type', 'low_rank'), ('--local_param_policy', 'norm')):
            validate_pel_cli(parser.parse_args(self._integration_argv(*axis)))
            without = [a for a in self._integration_argv(*axis)
                       if a != '--pel_allow_aggregation_integration']
            with self.assertRaises(ValueError):
                validate_pel_cli(parser.parse_args(without))
        # The whole aggregation-track candidate, and the flag refusing a no-op.
        validate_pel_cli(parser.parse_args(self._integration_argv(
            '--aggregation_method', 'fednova', '--personalization_type', 'low_rank',
            '--server_optimizer', 'momentum')))
        with self.assertRaises(ValueError):
            validate_pel_cli(parser.parse_args(self._integration_argv()))

    def test_fednova_round_diagnostics_are_labelled_and_reweighted(self):
        from fed.server import FedStrategyServer
        model = TinyShared(14)
        def loaders(n):
            return tuple(DataLoader([(torch.randn(8, 4), label, torch.tensor(8)) for _ in range(n)], batch_size=1)
                         for label in ('Normal', 'Fighting'))
        data, bank = [loaders(1), loaders(2)], torch.randn(14, 4)
        with patch('fed.server.build_text_bank', return_value=(bank, {'classes': list(range(14))})):
            server = FedStrategyServer('ucf', data, None, 2, 1, 1, 1e-4, 'event', [], .1, 'cpu', model,
                                       aggregation_method='fednova', use_pel4vad_contrastive=True,
                                       pel4vad_variant='fixed_bank', pel_diagnostics=True)
        server.pel_experiment_logging = True
        def evaluate(r):
            server.last_evaluation = {'roc': .6, 'ap': .4}
            return {'ucf': .6}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(server, 'evaluate', side_effect=evaluate), \
                patch.object(server, 'evaluate_personalized', return_value={}):
            server.train(directory)
            record = json.loads(Path(directory, 'pel_diagnostics.jsonl').read_text())
            row = json.loads(Path(directory, 'round_metrics.jsonl').read_text())
            summary = json.loads(Path(directory, 'summary.json').read_text())

        # The residual against the weighted mean is FedNova's reweighting, not float error,
        # so the rule has to travel with it.
        self.assertEqual(record['aggregation_rule'], 'fednova')
        self.assertEqual(record['local_steps'], [1, 2])
        self.assertNotEqual(record['rule_coefficients'], record['coefficients'])
        self.assertAlmostEqual(sum(record['coefficients']), 1)
        self.assertTrue(record['modules'])
        self.assertEqual([b['gradient']['status'] for b in record['client_batches']],
                         ['measured', 'measured'])
        # Client rows must carry the keys the existing comparison scripts read.
        for client in row['clients']:
            self.assertLessEqual({'clasm', 'pel', 'total', 'steps', 'used_class_counts'},
                                 set(client))
        self.assertEqual(row['selection_metric'], .6)
        self.assertEqual(summary['best_checkpoint_evaluation'], {'roc': .6, 'ap': .4})
        self.assertEqual(summary['independent_best_ap'], .4)

    def test_aggregation_weight_unit(self):
        """The paired ucf loop stops at the shorter loader, so only that many batches count."""
        from fed.client import aggregation_weight
        unbalanced = (DataLoader([0] * 17), DataLoader([0] * 4))
        self.assertEqual(aggregation_weight(unbalanced, 'ucf', 'loader_batches'), 21)
        self.assertEqual(aggregation_weight(unbalanced, 'ucf', 'effective_batches'), 4)
        self.assertEqual(aggregation_weight(unbalanced, 'cctv', 'effective_batches'), 4)
        # XD has one unpaired loader per client, so the two units cannot disagree.
        single = DataLoader([0] * 9)
        self.assertEqual(aggregation_weight(single, 'xd', 'loader_batches'),
                         aggregation_weight(single, 'xd', 'effective_batches'))
        for dataset in ('ucf', 'xd'):
            with self.assertRaises(ValueError):
                aggregation_weight(unbalanced, dataset, 'samples')

    def test_effective_batches_changes_recorded_weights(self):
        from fed.server import FedStrategyServer
        model = TinyShared(14)
        def loaders(normal, anomaly):
            return tuple(DataLoader([(torch.randn(8, 4), label, torch.tensor(8)) for _ in range(n)],
                                    batch_size=1)
                         for n, label in ((normal, 'Normal'), (anomaly, 'Fighting')))
        # client 0 is balanced, client 1 holds 4 normal batches but only 1 anomaly batch,
        # so it can only take 1 paired step per epoch.
        data = [loaders(2, 2), loaders(4, 1)]
        recorded = {}
        for unit, expected in (('loader_batches', [4, 5]), ('effective_batches', [2, 1])):
            server = FedStrategyServer('ucf', data, None, 2, 1, 1, 1e-4, 'event', [], .1, 'cpu',
                                       copy.deepcopy(model), aggregation_method='fednova',
                                       aggregation_weight_unit=unit)
            server.pel_experiment_logging = True
            def evaluate(r):
                server.last_evaluation = {'roc': .6, 'ap': .4}
                return {'ucf': .6}
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(server, 'evaluate', side_effect=evaluate), \
                    patch.object(server, 'evaluate_personalized', return_value={}):
                server.train(directory)
                row = json.loads(Path(directory, 'round_metrics.jsonl').read_text())
            self.assertEqual(row['aggregation_batch_counts'], expected)
            self.assertEqual(row['local_steps'], [2, 1])
            recorded[unit] = row['aggregation_batch_counts']
        # Under the corrected unit the weight is exactly the per-epoch step count, so the
        # weighting and FedNova's tau stop disagreeing about how much work each client did.
        self.assertEqual(recorded['effective_batches'], [2, 1])
        self.assertNotEqual(recorded['loader_batches'], recorded['effective_batches'])

    def test_protocol_selection(self):
        from scripts.prepare_pel_experiments import PROTOCOLS
        # The existing protocol's argv must not drift: every record in docs/experiments/
        # was generated with it.
        default = commands('integration', .3)[0]['argv']
        self.assertEqual(default[:14],
                         ['python', '-u', 'train.py', '--dataset', 'ucf', '--batch_size', '64',
                          '--learning_rate', '0.0001', '--global_rounds', '20',
                          '--local_epochs', '10', '--server_optimizer'])
        aggregation = commands('integration', .3, protocol='aggregation_track')[0]['argv']
        self.assertEqual(aggregation[7:11], ['--learning_rate', '1e-05', '--global_rounds', '15'])
        self.assertEqual(commands('integration', .3, protocol='aggregation_track')[0]['protocol'],
                         'aggregation_track')
        pinned = commands('integration', .3, weight_unit='effective_batches')[0]['argv']
        self.assertIn('--aggregation_weight_unit', pinned)
        self.assertNotIn('--aggregation_weight_unit', default)
        with self.assertRaises(ValueError):
            commands('integration', .3, protocol='paper')
        for name in PROTOCOLS:
            for row in commands('integration', .3, protocol=name, diagnostics=True,
                                arms=['fednova_adapter']):
                validate_pel_cli(parser.parse_args(row['argv'][3:]))

    def test_command_generation_is_only_data(self):
        for stage in ('strength', 'timing', 'pooling', 'modules', 'repeat', 'integration'):
            rows = commands(stage, .3 if stage != 'strength' else None)
            self.assertTrue(rows)
            for row in rows:
                validate_pel_cli(parser.parse_args(row['argv'][3:]))
        self.assertEqual({row['seed'] for row in commands('repeat', .3)}, {42, 2026})

    def test_paired_summary_missing_and_duplicate_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for enabled, seed, roc in [(False, 42, .8), (True, 42, .82), (True, 2026, .83)]:
                path = Path(directory, f'{enabled}_{seed}')
                path.mkdir()
                args = {'dataset': 'ucf', 'seed': seed, 'use_pel4vad_contrastive': enabled,
                        'pel4vad_weight': .3, 'global_rounds': 1}
                (path / 'manifest.json').write_text(json.dumps({'arguments': args}))
                (path / 'round_metrics.jsonl').write_text(json.dumps({'round': 1, 'evaluation': {'roc': roc, 'ap': .2}}))
                paths.append(path)
            row = summarize(paths)[0]
            self.assertEqual(row['paired_seeds'], ['42'])
            self.assertEqual(row['unpaired_candidate_seeds'], ['2026'])
            self.assertAlmostEqual(row['delta_statistics']['best_roc']['mean'], .02)
            self.assertIsNone(row['delta_statistics']['best_roc']['sample_std'])
            with self.assertRaises(ValueError):
                summarize(paths + paths[:1])


if __name__ == '__main__':
    unittest.main()
