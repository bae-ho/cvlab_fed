import os
import json
from collections import OrderedDict

import torch
import numpy as np
from fed.client import FedAvgClient, FedPGPClient, FedStrategyClient
from fed.aggregation import (build_server_optimizer, fednova_delta, subtract,
                              weighted_average)
from inference import inference
from utils.fixed_pel import build_text_bank
from utils.pel_diagnostics import update_diagnostics
from time import perf_counter


def _record_pel_integration_round(server, directory, round_index, taus=None):
    if not getattr(server, 'pel_experiment_logging', False):
        return
    # 'selection_metric' mirrors FedAvgServer's row so summarize_pel_runs.py and
    # record_event_pel_comparison.py consume either server's output unchanged.
    row = {'round': round_index, 'evaluation': dict(server.last_evaluation),
           'selection_metric': float(server.last_selection_metric),
           'clients': [client.last_train_metrics for client in server.clients],
           'aggregation_batch_counts': list(server.len_dataset), 'local_steps': taus,
           'checkpoint_selection': 'best round on test set (existing protocol)'}
    with open(os.path.join(directory, 'round_metrics.jsonl'), 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')


def _record_pel_integration_final(server, directory):
    if not getattr(server, 'pel_experiment_logging', False):
        return
    torch.save(server.model.state_dict(), os.path.join(directory, 'final_model.pth'))
    with open(os.path.join(directory, 'summary.json'), 'w', encoding='utf-8') as stream:
        json.dump({'final_evaluation': server.last_evaluation, 'best_selection_metric': float(server.best),
                   'best_checkpoint_evaluation': getattr(server, 'best_evaluation', None),
                   'independent_best_ap': getattr(server, 'best_ap', None),
                   'final_selection_metric': float(server.last_selection_metric),
                   'checkpoint_selection': 'best round on test set (existing protocol)',
                   'pel_bank': getattr(server, 'pel_bank_metadata', None)}, stream, indent=2)


UCF_LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism'}

XD_LABEL_MAP = {
    'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot',
    'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'}

CCTV_LABEL_MAP = {
    'Normal': 'normal', 'Fight': 'fighting', 'Flood': 'flood', 'Crowd': 'crowd',
    'Invasion': 'trespassing', 'Gathering': 'gathering', 'Falldown': 'falling down'}


class BaseServer:
    def __init__(self,
                 dataset: str,
                 train_loaders, test_loader,
                 clients_num: int,
                 global_rounds: int,
                 local_epochs: int,
                 model,
                 ):
        super().__init__()
        self.clients = []
        self.dataset = dataset
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.global_rounds = global_rounds
        self.model = model
        self.local_epochs = local_epochs
        self.clients_num = clients_num
        self.local_weights = []
        self.local_losses = []
        self.len_dataset = []
        # The metric checkpoint selection actually uses, recorded by the train loops from
        # the same res_dict they compare against self.best -- so a logged round can never
        # disagree with the round that was selected.
        self.last_selection_metric = None
        self.last_evaluation = {}

    def aggregate_parameters(self):
        """Sample-weighted FedAvg over whatever keys are present in self.local_weights.
        For FedAvgServer that is every trainable parameter; for FedPGPServer each client's
        train() only ever returns shared-parameter keys (personalized_prompt.* is excluded
        upstream in FedPGPClient.get_shared_parameters), so this same formula only averages
        the shared/global parameters."""
        temp_dict = OrderedDict()
        total_num = sum(self.len_dataset)
        for key, value in self.local_weights[0].items():
            temp_dict[key] = torch.zeros_like(value)

        for i in range(len(self.local_weights)):
            for key, value in self.local_weights[i].items():
                temp_dict[key] += value * self.len_dataset[i] / total_num

        return temp_dict

    def set_global_parameter(self, para):
        state_dict = self.model.state_dict()
        for key, value in para.items():
            state_dict[key] = value.data.clone()
        self.model.load_state_dict(state_dict)

    def send_global_parameter(self, para):
        for client in self.clients:
            client.set_parameters(para)

    def evaluate(self, r):
        """The official round-by-round metric: the shared/global model with personalization
        disabled (MODE 1 / global inference), so it stays comparable to the original
        Fed-WSVAD protocol regardless of aggregation_method."""
        if self.dataset == 'ucf':
            gt = np.load("./data/gt_ucf.npy")
        elif self.dataset == 'cctv':
            gt = np.load("./data/gt_cctv.npy")
        else:
            gt = np.load("./data/gt_xd.npy")

        roc, ap = inference(self.dataset, self.model, self.test_loader, gt, self.device, use_personalization=False)
        self.last_evaluation = {'roc': float(roc), 'ap': float(ap)}
        print(f"round {r + 1} : roc: {roc} , ap: {ap}")
        res_dict = {
            'ucf': roc,
            'cctv': roc,
            'xd': ap
        }
        return res_dict


class FedAvgServer(BaseServer):
    def __init__(self,
                 dataset: str,
                 train_loaders, test_loader,
                 clients_num: int,
                 global_rounds: int,
                 local_epochs: int,
                 learning_rate: float,
                 split_mode,
                 scheduler_milestones,
                 scheduler_rate,
                 device,
                 model,
                 use_pel4vad_contrastive=False,
                 pel4vad_variant='legacy',
                 pel4vad_weight=1.0,
                 pel4vad_temperature=0.09,
                 pel4vad_prompts=None,
                 pel_diagnostics=False,
                 pel_options=None,
                 aggregation_weight_unit='loader_batches',
                 ):
        super().__init__(dataset, train_loaders, test_loader, clients_num, global_rounds, local_epochs, model)
        self.dataset = dataset
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.split_mode = split_mode
        self.model = model
        self.global_parameter = None
        self.best = 0
        self.best_model = None
        self.device = device

        if self.dataset == 'ucf':
            label_map = dict(
                {'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault',
                 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting',
                 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting',
                 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})
        elif self.dataset == 'cctv':
            label_map = dict(CCTV_LABEL_MAP)
        else:
            label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot',
                              'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

        if len(train_loaders) != clients_num:
            raise ValueError('clients_num must match the number of constructed client loaders')
        if global_rounds <= 0:
            raise ValueError('global_rounds must be positive')
        self.pel_bank_metadata = None
        self.pel_diagnostics = pel_diagnostics
        self.aggregation_weight_unit = aggregation_weight_unit
        bank = None
        if use_pel4vad_contrastive and pel4vad_variant == 'fixed_bank':
            if dataset not in ('ucf', 'cctv'):
                raise ValueError('Fixed-bank PEL supports only single-label UCF/CCTV')
            bank, self.pel_bank_metadata = build_text_bank(model, label_map, pel4vad_prompts)
        self.label_map = label_map
        for i in range(clients_num):

            client = FedAvgClient(model, learning_rate, train_loaders[i], dataset,
                                  local_epochs, label_map, scheduler_milestones,
                                  scheduler_rate, device,
                                  use_pel4vad_contrastive=use_pel4vad_contrastive,
                                  pel4vad_variant=pel4vad_variant,
                                  pel4vad_weight=pel4vad_weight,
                                  pel4vad_temperature=pel4vad_temperature,
                                  pel_text_bank=bank,
                                  pel_diagnostics=pel_diagnostics,
                                  pel_options=pel_options,
                                  aggregation_weight_unit=aggregation_weight_unit)

            self.clients.append(client)

    def evaluate_local(self, r):
        if self.dataset == 'ucf':
            gt = np.load("./data/gt_ucf.npy")
        elif self.dataset == 'cctv':
            gt = np.load("./data/gt_cctv.npy")
        else:
            gt = np.load("./data/gt_xd.npy")

        res_dict = {
            'ucf': [],
            'xd': []
        }
        for index in range(len(self.local_weights)):
            self.set_global_parameter(self.local_weights[index])
            roc, ap = inference(self.dataset, self.model, self.test_loader, gt, self.device)
            res_dict['ucf'].append(roc)
            res_dict['xd'].append(ap)

        print(f"round {r + 1} : roc: {sum(res_dict['ucf']) / len(res_dict['ucf'])} ,"
              f" ap: {sum(res_dict['ap']) / len(res_dict['ap'])}")

    def train(self, dir_name):
        best_evaluation = None
        best_ap = None
        for g in range(self.global_rounds):
            print(f"-------- round: {g + 1} / {self.global_rounds} --------")
            round_start = ({n: p.detach().cpu().clone() for n, p in self.model.named_parameters()
                            if p.requires_grad} if self.pel_diagnostics else None)
            self.local_weights.clear()
            self.local_losses.clear()
            self.len_dataset.clear()

            i = 0
            for client in self.clients:
                i += 1
                print(f"round {g + 1}, client: {i}")
                client.diagnostic_round = g + 1
                client.pel_round = g + 1
                w, loss, l_data = client.train()

                self.local_weights.append(w)
                self.local_losses.append(loss)
                self.len_dataset.append(l_data)
                client.scheduler.step()

            self.global_parameter = self.aggregate_parameters()
            if self.pel_diagnostics:
                diagnostic_start = perf_counter()
                diagnostics = update_diagnostics(round_start, self.local_weights,
                                                 self.len_dataset, self.global_parameter,
                                                 unit=self.aggregation_weight_unit)
                diagnostics.update({'schema_version': 1, 'round': g + 1,
                                    'classes': list(self.label_map), 'bank': self.pel_bank_metadata,
                                    'client_batches': [c.last_diagnostics for c in self.clients],
                                    'update_diagnostic_seconds': perf_counter() - diagnostic_start})
                with open(os.path.join(dir_name, 'pel_diagnostics.jsonl'), 'a', encoding='utf-8') as stream:
                    stream.write(json.dumps(diagnostics, allow_nan=False) + '\n')
            self.set_global_parameter(self.global_parameter)
            res = self.evaluate(g)
            evaluation = getattr(self, 'last_evaluation', {})
            if 'ap' in evaluation:
                best_ap = evaluation['ap'] if best_ap is None else max(best_ap, evaluation['ap'])

            with open(os.path.join(dir_name, 'round_metrics.jsonl'), 'a', encoding='utf-8') as stream:
                stream.write(json.dumps({'round': g + 1, 'selection_metric': float(res[self.dataset]),
                                         'evaluation': getattr(self, 'last_evaluation', {}),
                                         'clients': [c.last_train_metrics for c in self.clients],
                                         'aggregation_batch_counts': self.len_dataset}) + '\n')

            self.send_global_parameter(self.global_parameter)
            if self.best_model is None or res[self.dataset] > self.best:
                self.best = res[self.dataset]
                self.best_model = self.global_parameter
                best_evaluation = dict(evaluation)

                torch.save(self.model.state_dict(), dir_name + "/model.pth")

            print(f"best: {self.best}")

        torch.save(self.model.state_dict(), os.path.join(dir_name, 'final_model.pth'))
        with open(os.path.join(dir_name, 'summary.json'), 'w', encoding='utf-8') as stream:
            json.dump({'best_selection_metric': float(self.best),
                       'best_checkpoint_evaluation': best_evaluation,
                       'independent_best_ap': best_ap,
                       'final_selection_metric': float(res[self.dataset]),
                       'final_evaluation': getattr(self, 'last_evaluation', {}),
                       'checkpoint_selection': 'best round on test set (existing protocol)',
                       'pel_bank': self.pel_bank_metadata}, stream, indent=2)
        self.set_global_parameter(self.best_model)
        torch.save(self.model.state_dict(), dir_name + "/model.pth")


class FedPGPServer(BaseServer):
    """Same communication-round structure as FedAvgServer. The only structural change is
    *what* gets aggregated: FedPGPClient.train() returns shared-parameter-only state dicts
    (personalized_prompt.* excluded), so aggregate_parameters()/set_global_parameter()/
    send_global_parameter() inherited from BaseServer transparently only ever touch shared
    parameters. Each client's low-rank adapter (A_i, B_i) lives exclusively on that client's
    persistent FedPGPClient.model and is never part of self.local_weights."""

    def __init__(self,
                 dataset: str,
                 train_loaders, test_loader,
                 clients_num: int,
                 global_rounds: int,
                 local_epochs: int,
                 learning_rate: float,
                 split_mode,
                 scheduler_milestones,
                 scheduler_rate,
                 device,
                 model,
                 personalized_lr: float = None,
                 use_fedpgp_contrastive: bool = False,
                 contrastive_weight: float = 0.1,
                 contrastive_temperature: float = 0.07,
                 use_pel4vad_contrastive: bool = False,
                 pel4vad_weight: float = 1.0,
                 pel4vad_temperature: float = 0.09,
                 verbose_fed_logging: bool = False,
                 pel4vad_variant='legacy',
                 pel4vad_prompts=None,
                 pel_options=None,
                 aggregation_weight_unit='loader_batches',
                 ):
        super().__init__(dataset, train_loaders, test_loader, clients_num, global_rounds, local_epochs, model)
        self.dataset = dataset
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.split_mode = split_mode
        self.model = model
        self.global_parameter = None
        self.best = 0
        self.best_model = None
        self.best_evaluation = None
        self.best_ap = None
        self.device = device
        self.verbose_fed_logging = verbose_fed_logging

        if self.dataset == 'ucf':
            label_map = dict(
                {'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault',
                 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting',
                 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting',
                 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})
        elif self.dataset == 'cctv':
            label_map = dict(CCTV_LABEL_MAP)
        else:
            label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot',
                              'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

        self.pel_bank_metadata = None
        bank = None
        if use_pel4vad_contrastive and pel4vad_variant == 'fixed_bank':
            bank, self.pel_bank_metadata = build_text_bank(model, label_map, pel4vad_prompts)
        for i in range(clients_num):
            client = FedPGPClient(model, learning_rate, train_loaders[i], dataset,
                                   local_epochs, label_map, scheduler_milestones,
                                   scheduler_rate, device,
                                   personalized_lr=personalized_lr,
                                   use_fedpgp_contrastive=use_fedpgp_contrastive,
                                   contrastive_weight=contrastive_weight,
                                   contrastive_temperature=contrastive_temperature,
                                   use_pel4vad_contrastive=use_pel4vad_contrastive,
                                   pel4vad_weight=pel4vad_weight,
                                   pel4vad_temperature=pel4vad_temperature,
                                   pel4vad_variant=pel4vad_variant,
                                   pel_options=pel_options,
                                   pel_text_bank=bank,
                                   aggregation_weight_unit=aggregation_weight_unit)
            self.clients.append(client)

    def evaluate_personalized(self, client_id: int, r: int):
        """MODE 2 (personalized inference), evaluated on the shared/global test set since the
        current data pipeline only provides one global test split (not a held-out split per
        client). This is therefore a proxy metric -- 'client_id's adapter applied to the
        global test distribution' -- not a true personalized-domain evaluation. See report.

        Uses whatever shared parameters are already loaded on self.model (the caller is
        responsible for that, e.g. the best/final checkpoint); only the personalized adapter
        is swapped in and restored afterwards, so this never perturbs the shared weights."""
        if self.dataset == 'ucf':
            gt = np.load("./data/gt_ucf.npy")
        elif self.dataset == 'cctv':
            gt = np.load("./data/gt_cctv.npy")
        else:
            gt = np.load("./data/gt_xd.npy")

        personalized_backup = self.model.get_personalized_state_dict()
        self.model.load_personalized_state_dict(self.clients[client_id].get_personalized_parameters())

        roc, ap = inference(self.dataset, self.model, self.test_loader, gt, self.device, use_personalization=True)
        print(f"round {r + 1} : client {client_id} personalized (on global test set) roc: {roc} , ap: {ap}")

        if personalized_backup:
            self.model.load_personalized_state_dict(personalized_backup)
        return {'ucf': roc, 'cctv': roc, 'xd': ap}

    def _log_round_diagnostics(self, g, client_updates):
        if self.model.personalized_parameters():
            for i, client in enumerate(self.clients):
                delta_norm = sum(p.data.norm().item() for p in client.model.personalized_parameters())
                print(f"    client {i} personalized_delta_norm: {delta_norm:.6f}")

        agg_flat = torch.cat([v.flatten() for v in self.global_parameter.values()])
        for i, w in enumerate(client_updates):
            update_flat = torch.cat([v.flatten() for v in w.values()])
            cos = torch.nn.functional.cosine_similarity(update_flat.unsqueeze(0), agg_flat.unsqueeze(0)).item()
            print(f"    client {i} shared_update_norm: {update_flat.norm().item():.6f}, "
                  f"cos(client_update, aggregated_update): {cos:.4f}")

    def train(self, dir_name):
        for g in range(self.global_rounds):
            print(f"-------- round: {g + 1} / {self.global_rounds} --------")
            self.local_weights.clear()
            self.local_losses.clear()
            self.len_dataset.clear()

            i = 0
            for client in self.clients:
                i += 1
                print(f"round {g + 1}, client: {i}")
                client.pel_round = g + 1
                w, loss, l_data = client.train()

                self.local_weights.append(w)
                self.local_losses.append(loss)
                self.len_dataset.append(l_data)
                client.scheduler.step()

            self.global_parameter = self.aggregate_parameters()
            self.set_global_parameter(self.global_parameter)
            res = self.evaluate(g)
            self.last_selection_metric = res[self.dataset]

            _record_pel_integration_round(self, dir_name, g + 1,
                                          [c.local_steps for c in self.clients])

            if self.verbose_fed_logging:
                self._log_round_diagnostics(g, self.local_weights)

            self.send_global_parameter(self.global_parameter)
            self.best_ap = (self.last_evaluation['ap'] if self.best_ap is None
                            else max(self.best_ap, self.last_evaluation['ap']))
            if self.best_model is None or res[self.dataset] > self.best:
                self.best = res[self.dataset]
                self.best_model = self.global_parameter
                self.best_evaluation = dict(self.last_evaluation)

                torch.save(self.model.state_dict(), dir_name + "/model.pth")

            print(f"best: {self.best}")

        _record_pel_integration_final(self, dir_name)
        self.set_global_parameter(self.best_model)
        torch.save(self.model.state_dict(), dir_name + "/model.pth")

        for i, client in enumerate(self.clients):
            torch.save(client.get_personalized_parameters(), os.path.join(dir_name, f"client_{i}_personalized.pth"))

        for i in range(len(self.clients)):
            self.evaluate_personalized(i, self.global_rounds - 1)


class FedStrategyServer(BaseServer):
    """Generalized federated server composing three orthogonal choices.

    FedAvgServer hard-codes all three; this one lets each be selected independently:

      1. aggregation rule      -- 'plain' weighted average, or FedNova's
                                  step-count-normalized update
      2. server optimizer      -- none / momentum (FedAvgM) / adam (FedAdam) / yogi (FedYogi)
      3. local parameter policy -- which trainable params never leave the client
                                  ('none', 'adapter' = FedPGP, 'norm' = FedBN-style, or both)

    plus optional client-side drift control: FedProx's proximal mu and SCAFFOLD's
    control variates. Because the axes are independent, combinations like
    "FedNova aggregation + FedAdam server + local LayerNorms" are one flag each.

    FedAvgServer and FedPGPServer are intentionally left untouched so the results
    already produced with them stay bit-for-bit reproducible.
    """

    def __init__(self,
                 dataset: str,
                 train_loaders, test_loader,
                 clients_num: int,
                 global_rounds: int,
                 local_epochs: int,
                 learning_rate: float,
                 split_mode,
                 scheduler_milestones,
                 scheduler_rate,
                 device,
                 model,
                 aggregation_method: str = "fedavg",
                 server_optimizer: str = "none",
                 server_lr: float = None,
                 server_beta1: float = 0.9,
                 server_beta2: float = 0.99,
                 server_tau: float = 1e-3,
                 prox_mu: float = 0.0,
                 personalized_lr: float = None,
                 personalized_weight_decay: float = None,
                 use_fedpgp_contrastive: bool = False,
                 contrastive_weight: float = 0.1,
                 contrastive_temperature: float = 0.07,
                 use_pel4vad_contrastive: bool = False,
                 pel4vad_weight: float = 1.0,
                 pel4vad_temperature: float = 0.09,
                 verbose_fed_logging: bool = False,
                 pel4vad_variant='legacy',
                 pel4vad_prompts=None,
                 pel_diagnostics=False,
                 pel_options=None,
                 aggregation_weight_unit='loader_batches',
                 ):
        super().__init__(dataset, train_loaders, test_loader, clients_num, global_rounds,
                         local_epochs, model)
        if len(train_loaders) != clients_num:
            raise ValueError('clients_num must match the number of constructed client loaders')
        if global_rounds <= 0:
            raise ValueError('global_rounds must be positive')
        self.split_mode = split_mode
        self.device = device
        self.aggregation_method = aggregation_method
        self.verbose_fed_logging = verbose_fed_logging
        self.pel_diagnostics = pel_diagnostics
        self.aggregation_weight_unit = aggregation_weight_unit
        self.best = 0
        self.best_model = None
        self.best_evaluation = None
        self.best_ap = None

        self.use_scaffold = aggregation_method == "scaffold"
        self.use_fednova = aggregation_method == "fednova"

        # The server tracks the global model explicitly (rather than reading it back off
        # self.model) because momentum/Adam servers need a stable reference point to
        # apply their accumulated state against.
        self.global_parameter = model.get_shared_state_dict()
        self.server_optimizer_name = server_optimizer
        self.server_optimizer = build_server_optimizer(
            server_optimizer, lr=server_lr, beta1=server_beta1,
            beta2=server_beta2, tau=server_tau)

        self.c_global = None
        if self.use_scaffold:
            self.c_global = OrderedDict(
                (name, torch.zeros_like(value)) for name, value in self.global_parameter.items())

        label_map = {'ucf': UCF_LABEL_MAP, 'xd': XD_LABEL_MAP, 'cctv': CCTV_LABEL_MAP}[dataset]
        self.label_map = label_map
        self.pel_bank_metadata = None
        bank = None
        if use_pel4vad_contrastive and pel4vad_variant == 'fixed_bank':
            if dataset not in ('ucf', 'cctv'):
                raise ValueError('Fixed-bank PEL supports only single-label UCF/CCTV')
            bank, self.pel_bank_metadata = build_text_bank(model, label_map, pel4vad_prompts)

        for i in range(clients_num):
            client = FedStrategyClient(
                model, learning_rate, train_loaders[i], dataset, local_epochs, label_map,
                scheduler_milestones, scheduler_rate, device,
                personalized_lr=personalized_lr,
                personalized_weight_decay=personalized_weight_decay,
                use_fedpgp_contrastive=use_fedpgp_contrastive,
                contrastive_weight=contrastive_weight,
                contrastive_temperature=contrastive_temperature,
                use_pel4vad_contrastive=use_pel4vad_contrastive,
                pel4vad_weight=pel4vad_weight,
                pel4vad_temperature=pel4vad_temperature,
                pel4vad_variant=pel4vad_variant,
                pel_text_bank=bank,
                pel_diagnostics=pel_diagnostics,
                pel_options=pel_options,
                prox_mu=prox_mu,
                use_scaffold=self.use_scaffold,
                c_global=self.c_global,
                aggregation_weight_unit=aggregation_weight_unit)
            self.clients.append(client)

        # Hand out the initial global weights before round 1. Numerically a no-op (every
        # client deepcopied the same model), but it populates each client's reference to
        # w_global -- without it FedProx would have no anchor and SCAFFOLD no control
        # update during the first round.
        self.send_global_parameter(self.global_parameter)

    def compute_delta(self, taus):
        """Reduce this round's local models into a single update to the global model."""
        if self.use_fednova:
            return fednova_delta(self.global_parameter, self.local_weights, taus, self.len_dataset)
        return subtract(weighted_average(self.local_weights, self.len_dataset),
                        self.global_parameter)

    def update_scaffold_global_control(self, delta_cs):
        """c <- c + (1/N) * sum_i delta_c_i, with N the total number of clients."""
        delta_cs = [d for d in delta_cs if d]
        if not delta_cs:
            return
        for name in self.c_global:
            for delta_c in delta_cs:
                if name in delta_c:
                    self.c_global[name] += delta_c[name] / self.clients_num

    def evaluate_personalized(self, client_id: int, r: int):
        """Apply one client's local parameters on top of whatever shared weights are
        currently loaded. Evaluated on the global test set, since the data pipeline
        provides no per-client held-out split -- a proxy metric, not a true
        personalized-domain evaluation."""
        gt_path = {'ucf': './data/gt_ucf.npy', 'xd': './data/gt_xd.npy', 'cctv': './data/gt_cctv.npy'}[self.dataset]
        gt = np.load(gt_path)

        backup = self.model.get_local_state_dict()
        self.model.load_local_state_dict(self.clients[client_id].get_personalized_parameters())

        roc, ap = inference(self.dataset, self.model, self.test_loader, gt, self.device,
                            use_personalization=True)
        print(f"round {r + 1} : client {client_id} personalized (on global test set) "
              f"roc: {roc} , ap: {ap}")

        if backup:
            self.model.load_local_state_dict(backup)
        return {'ucf': roc, 'cctv': roc, 'xd': ap}

    def _log_round_diagnostics(self, taus, delta):
        local_norm = sum(p.data.norm().item() for p in self.clients[0].model.personalized_parameters())
        if local_norm:
            print(f"    client 0 local_param_norm: {local_norm:.6f}")
        print(f"    local steps per client (tau): {taus}")
        delta_norm = torch.cat([v.flatten() for v in delta.values()]).norm().item()
        print(f"    aggregated update norm: {delta_norm:.6f}")

        agg_flat = torch.cat([v.flatten() for v in delta.values()])
        for i, w in enumerate(self.local_weights):
            client_delta = torch.cat(
                [(w[k] - self.global_parameter[k]).flatten() for k in self.global_parameter])
            cos = torch.nn.functional.cosine_similarity(
                client_delta.unsqueeze(0), agg_flat.unsqueeze(0)).item()
            print(f"    client {i} update_norm: {client_delta.norm().item():.6f}, "
                  f"cos(client, aggregate): {cos:.4f}")

    def aggregation_rule_name(self):
        """How the round's actual update relates to the batch-count-weighted mean.

        Only FedNova changes the server-side reduction; FedProx and SCAFFOLD alter the
        client objective and still average plainly. A server optimizer moves the applied
        step away from the reduction again, so it is appended when present.
        """
        rule = 'fednova' if self.use_fednova else 'weighted_mean'
        if self.server_optimizer_name != 'none':
            rule = f'{rule}+{self.server_optimizer_name}'
        return rule

    def train(self, dir_name):
        for g in range(self.global_rounds):
            print(f"-------- round: {g + 1} / {self.global_rounds} --------")
            # Anchor every client delta to the same round-start weights. global_parameter is
            # replaced rather than mutated each round, but clone so a future in-place server
            # optimizer cannot silently move the anchor under the diagnostics.
            round_start = (OrderedDict((n, p.detach().cpu().clone())
                                       for n, p in self.global_parameter.items())
                           if self.pel_diagnostics else None)
            self.local_weights.clear()
            self.local_losses.clear()
            self.len_dataset.clear()
            taus, delta_cs = [], []

            for i, client in enumerate(self.clients):
                print(f"round {g + 1}, client: {i + 1}")
                client.pel_round = g + 1
                client.diagnostic_round = g + 1
                w, loss, l_data = client.train()

                self.local_weights.append(w)
                self.local_losses.append(loss)
                self.len_dataset.append(l_data)
                taus.append(client.local_steps)
                delta_cs.append(client.last_delta_c)
                client.scheduler.step()

            delta = self.compute_delta(taus)
            if self.verbose_fed_logging:
                self._log_round_diagnostics(taus, delta)

            self.global_parameter = self.server_optimizer.step(self.global_parameter, delta)
            self.set_global_parameter(self.global_parameter)

            if self.use_scaffold:
                self.update_scaffold_global_control(delta_cs)

            if self.pel_diagnostics:
                diagnostic_start = perf_counter()
                diagnostics = update_diagnostics(round_start, self.local_weights,
                                                 self.len_dataset, self.global_parameter,
                                                 aggregation=self.aggregation_rule_name(),
                                                 taus=taus, unit=self.aggregation_weight_unit)
                diagnostics.update({'schema_version': 1, 'round': g + 1,
                                    'classes': list(self.label_map), 'bank': self.pel_bank_metadata,
                                    'client_batches': [c.last_diagnostics for c in self.clients],
                                    'update_diagnostic_seconds': perf_counter() - diagnostic_start})
                with open(os.path.join(dir_name, 'pel_diagnostics.jsonl'), 'a', encoding='utf-8') as stream:
                    stream.write(json.dumps(diagnostics, allow_nan=False) + '\n')

            res = self.evaluate(g)
            self.last_selection_metric = res[self.dataset]
            self.send_global_parameter(self.global_parameter)

            _record_pel_integration_round(self, dir_name, g + 1, taus)

            evaluation = self.last_evaluation
            self.best_ap = (evaluation['ap'] if self.best_ap is None
                            else max(self.best_ap, evaluation['ap']))
            if self.best_model is None or res[self.dataset] > self.best:
                self.best = res[self.dataset]
                self.best_model = self.global_parameter
                self.best_evaluation = dict(evaluation)
                torch.save(self.model.state_dict(), dir_name + "/model.pth")

            print(f"best: {self.best}")

        _record_pel_integration_final(self, dir_name)
        self.set_global_parameter(self.best_model)
        torch.save(self.model.state_dict(), dir_name + "/model.pth")

        if self.model.local_param_policy != "none":
            for i, client in enumerate(self.clients):
                torch.save(client.get_personalized_parameters(),
                           os.path.join(dir_name, f"client_{i}_local.pth"))
            for i in range(len(self.clients)):
                self.evaluate_personalized(i, self.global_rounds - 1)
