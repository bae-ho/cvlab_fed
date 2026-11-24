import copy
import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from fed.client import FedAvgClient
from utils.fixed_pel import build_text_bank, fixed_bank_pel_loss, single_class_indices
from utils.tools import CLASM, get_prompt_text, get_batch_label


class TinyModel(nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.temporal = nn.Linear(4, 4)
        self.prompt = nn.Parameter(torch.randn(4, classes))

    def forward(self, visual, prompts, lengths, return_pel4vad_aux=False):
        h = self.temporal(visual)
        logits = h @ self.prompt
        return (logits, {'visual_features': h, 'text_features': self.prompt.T}) if return_pel4vad_aux else logits


class FixedPELTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.bank = torch.randn(3, 4)
        self.model = TinyModel()
        self.visual = torch.randn(3, 8, 4)
        self.lengths = torch.tensor([8, 5, 6])
        self.classes = torch.tensor([0, 1, 2])

    def loss(self, h, logits, bank=None):
        return fixed_bank_pel_loss(h, logits, self.lengths,
                                   self.bank if bank is None else bank, self.classes)

    def test_gradient_boundary_and_padding(self):
        logits, aux = self.model(self.visual, [], self.lengths, True)
        h = aux['visual_features']
        loss = self.loss(h, logits)
        loss.backward()
        self.assertIsNone(self.model.prompt.grad)
        self.assertTrue(torch.isfinite(self.model.temporal.weight.grad).all())
        self.assertGreater(self.model.temporal.weight.grad.abs().sum().item(), 0)
        padded_h, padded_logits = h.detach().clone(), logits.detach().clone()
        for i, length in enumerate(self.lengths):
            padded_h[i, length:] = 100
            padded_logits[i, length:] = -100
        torch.testing.assert_close(loss.detach(), self.loss(padded_h, padded_logits))

    def test_absent_class_still_in_denominator(self):
        h = torch.randn(1, 8, 4)
        logits = torch.zeros(1, 8, 3)
        def run(bank):
            return fixed_bank_pel_loss(h, logits, torch.tensor([8]), bank, torch.tensor([1]))
        changed = self.bank.clone()
        changed[2] *= -1
        self.assertFalse(torch.isclose(run(self.bank), run(changed)))

    def test_degenerate_scores_and_validation(self):
        for normal_logit in [-1000, 1000]:
            logits = torch.zeros(3, 8, 3)
            logits[..., 0] = normal_logit
            self.assertTrue(torch.isfinite(self.loss(self.visual, logits)))
        for labels in [torch.zeros(3, 3), torch.ones(3, 3)]:
            with self.assertRaises(ValueError):
                single_class_indices(labels)
        with self.assertRaises(ValueError):
            fixed_bank_pel_loss(self.visual, torch.zeros(3, 8, 3),
                               torch.tensor([0, 5, 6]), self.bank, self.classes)

    def test_bank_preserves_rng_modes_and_is_frozen(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4), requires_grad=False)
            def encode_text_original(self, tokens):
                return tokens.float() @ self.weight + torch.rand_like(tokens.float())
        holder = nn.Module()
        holder.clipmodel = Encoder()
        state = torch.random.get_rng_state().clone()
        bank, metadata = build_text_bank(holder, {'Normal': 'normal', 'Fight': 'fighting'},
                                        tokenize=lambda texts: torch.ones(len(texts), 4))
        torch.testing.assert_close(state, torch.random.get_rng_state())
        self.assertTrue(holder.clipmodel.training)
        self.assertFalse(bank.requires_grad)
        self.assertEqual(len(metadata['sha256']), 64)

    def test_client_off_matches_original_update_and_mixed_events_train(self):
        label_map = {'Normal': 'normal', 'Fighting': 'fighting', 'Arson': 'arson'}
        normal = [(torch.randn(8, 4), 'Normal', torch.tensor(8)) for _ in range(2)]
        anomaly = [(torch.randn(8, 4), label, torch.tensor(8)) for label in ['Fighting', 'Arson']]
        loaders = (DataLoader(normal, batch_size=2), DataLoader(anomaly, batch_size=2))
        reference = copy.deepcopy(self.model)
        optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4)
        prompts = get_prompt_text(label_map)
        visual = torch.cat([next(iter(loader))[0] for loader in loaders])
        lengths = torch.full((4,), 8)
        labels = get_batch_label(['Normal', 'Normal', 'Fighting', 'Arson'], prompts, label_map, 'ucf')
        expected = CLASM(reference(visual, prompts, lengths), labels, lengths, 'cpu')
        optimizer.zero_grad()
        expected.backward()
        optimizer.step()
        baseline = FedAvgClient(self.model, 1e-4, loaders, 'ucf', 1, label_map, [], 0.1, 'cpu')
        parameters, loss, count = baseline.train()
        self.assertAlmostEqual(loss, expected.item())
        self.assertEqual(count, 2)
        for name, value in reference.named_parameters():
            torch.testing.assert_close(parameters[name], value)
        candidate = FedAvgClient(self.model, 1e-4, loaders, 'ucf', 1, label_map, [], 0.1,
                                 'cpu', True, 'fixed_bank', pel_text_bank=self.bank)
        original_bank = self.bank.clone()
        candidate.train()
        torch.testing.assert_close(original_bank, candidate.pel_text_bank)
        self.assertGreater(candidate.last_train_metrics['pel'], 0)
        self.assertEqual(set(parameters), set(candidate.get_global_parameters()))


    def test_two_client_server_keeps_batch_weighting_and_saves_metrics(self):
        from fed.server import FedAvgServer, UCF_LABEL_MAP, CCTV_LABEL_MAP
        self.assertEqual(len(UCF_LABEL_MAP), 14)
        self.assertEqual(len(CCTV_LABEL_MAP), 7)
        model = TinyModel(14)
        def loaders(n):
            return tuple(DataLoader([(torch.randn(8, 4), label, torch.tensor(8))
                                     for _ in range(n)], batch_size=1)
                         for label in ['Normal', 'Fighting'])
        bank = torch.randn(14, 4)
        with patch('fed.server.build_text_bank', return_value=(bank, {'classes': list(UCF_LABEL_MAP)})):
            server = FedAvgServer('ucf', [loaders(1), loaders(2)], None, 2, 2, 1,
                                  1e-4, 'event', [], 0.1, 'cpu', model, True, 'fixed_bank')
        self.assertIs(server.clients[0].pel_text_bank, server.clients[1].pel_text_bank)
        def evaluate(r):
            server.last_evaluation = {'roc': .5 + .01 * r, 'ap': .4}
            return {'ucf': server.last_evaluation['roc']}
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'evaluate', side_effect=evaluate):
            server.train(directory)
            rows = [json.loads(row) for row in Path(directory, 'round_metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['aggregation_batch_counts'], [2, 4])
            for key in server.global_parameter:
                expected = server.local_weights[0][key] / 3 + server.local_weights[1][key] * 2 / 3
                torch.testing.assert_close(expected, server.global_parameter[key])
            final = torch.load(Path(directory, 'final_model.pth'), weights_only=True)
            self.assertEqual(set(final), set(model.state_dict()))
            self.assertTrue(Path(directory, 'summary.json').exists())

    def test_actual_model_temporal_and_position_receive_only_pel_gradients(self):
        from utils.model import Model
        class TokenEmbedding(nn.Embedding):
            def forward(self, tokens):
                return super().forward(tokens % self.num_embeddings)
        class FakeClip(nn.Module):
            dtype = torch.float32
            def __init__(self):
                super().__init__()
                self.token_embedding = TokenEmbedding(128, 512)
                self.logit_scale = nn.Parameter(torch.tensor(1.0))
            def encode_text_original(self, tokens):
                return self.token_embedding(tokens).mean(1)
            def encode_text(self, embeddings, tokens):
                return embeddings.mean(1)
        with patch('utils.model.clip.load', return_value=(FakeClip(), None)):
            model = Model(512, 256, 10, 10, 512, 1, 1, 8, 'cpu')
        keys = set(model.state_dict())
        prompts = ['normal', 'fighting', 'arson']
        bank, _ = build_text_bank(model, dict(zip(['Normal', 'Fighting', 'Arson'], prompts)))
        visual, lengths = torch.randn(2, 256, 512), torch.tensor([256, 200])
        model.eval()
        plain = model(visual, prompts, lengths)
        logits, aux = model(visual, prompts, lengths, return_pel4vad_aux=True)
        torch.testing.assert_close(plain, logits)
        loss = fixed_bank_pel_loss(aux['visual_features'], logits, lengths, bank, torch.tensor([0, 1]))
        from utils.pel_diagnostics import gradient_diagnostics
        mil = CLASM(logits, torch.tensor([[1., 0., 0.], [0., 1., 0.]]), lengths, 'cpu')
        diagnostics = gradient_diagnostics(model, mil, loss, 1)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for group in ('temporal', 'frame_position_embeddings'):
            self.assertGreater(diagnostics['modules'][group]['pel_norm'], 0)
        loss.backward()
        for module in [model.temporal, model.frame_position_embeddings]:
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertTrue(all(p.grad is None for p in model.prompt_learner.parameters()))
        self.assertTrue(all(p.grad is None for p in model.clipmodel.parameters()))
        model.zero_grad()
        logits = model(visual, prompts, lengths)
        CLASM(logits, torch.tensor([[1., 0., 0.], [0., 1., 0.]]), lengths, 'cpu').backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.prompt_learner.parameters()))
        self.assertEqual(keys, set(model.state_dict()))


if __name__ == '__main__':
    unittest.main()
