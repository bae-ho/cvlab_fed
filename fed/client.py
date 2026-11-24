import copy
from collections import OrderedDict
import torch
from torch.optim.lr_scheduler import MultiStepLR
from utils.tools import get_batch_label, get_prompt_text, CLASM, fedpgp_contrastive_loss, pel4vad_contrastive_loss
from utils.fixed_pel import build_text_bank, single_class_indices
from utils.pel_diagnostics import gradient_diagnostics
from time import perf_counter
from utils.pel_experiments import (setup_pel_experiment, prepare_pel_round, fixed_loss,
                                   combine_pel, backward_pel)

# Rounds that get the per-batch gradient measurement. Sampling rather than measuring every
# batch keeps the extra autograd passes off the training path; both client classes use the
# same set so a baseline and an aggregation-track run line up round for round.
DIAGNOSTIC_ROUNDS = (1, 5, 10, 15, 20)


def aggregation_weight(train_loaders, dataset, unit):
    """n_i: how much this client counts for in the server's weighted average.

    'loader_batches' is the original unit, len(normal) + len(anomaly). It over-counts any
    client whose two loaders differ in length: the ucf/cctv loop pairs one normal batch with
    one anomaly batch and therefore stops at min(len) per epoch, so a client holding 17
    normal and 4 anomaly batches takes 4 optimizer steps but is weighted as if it had
    contributed 21. On the UCF scene split that misstates one client's share by about 2x.

    'effective_batches' counts the batches actually consumed, which also makes n_i
    proportional to the step count tau_i that FedNova divides by -- the same quantity, so
    the weighting and the normalization stop disagreeing about how much work a client did.

    XD gives each client a single unpaired loader, so both units return len(loader).
    """
    if unit not in ('loader_batches', 'effective_batches'):
        raise ValueError(f'unknown aggregation_weight_unit: {unit!r}')
    if dataset == 'xd':
        return len(train_loaders)
    if unit == 'loader_batches':
        return len(train_loaders[0]) + len(train_loaders[1])
    return min(len(train_loaders[0]), len(train_loaders[1]))


class FedAvgClient:
    def __init__(self,
                 model,
                 learning_rate: float,
                 train_loaders: tuple,
                 dataset: str,
                 local_epochs: int,
                 label_map,
                 scheduler_milestones,
                 scheduler_rate,
                 device: str,
                 use_pel4vad_contrastive=False,
                 pel4vad_variant='legacy',
                 pel4vad_weight=1.0,
                 pel4vad_temperature=0.09,
                 pel_text_bank=None,
                 pel_diagnostics=False,
                 pel_options=None,
                 aggregation_weight_unit='loader_batches',
                 ):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.learning_rate = learning_rate
        self.train_loaders = train_loaders
        self.dataset = dataset
        self.aggregation_weight_unit = aggregation_weight_unit
        self.local_epochs = local_epochs
        self.label_map = label_map
        self.device = device
        self.use_pel4vad_contrastive = use_pel4vad_contrastive
        self.pel4vad_variant = pel4vad_variant
        self.pel4vad_weight = pel4vad_weight
        self.pel4vad_temperature = pel4vad_temperature
        setup_pel_experiment(self, pel_options)
        self.pel_text_bank = pel_text_bank
        self.last_train_metrics = {}
        self.pel_diagnostics = pel_diagnostics
        self.diagnostic_round = None
        self.last_diagnostics = None
        self._pool_diagnostics = None
        if pel_diagnostics and (dataset not in ('ucf', 'cctv') or
                                (use_pel4vad_contrastive and pel4vad_variant != 'fixed_bank')):
            raise ValueError('PEL diagnostics support UCF/CCTV with PEL off or fixed_bank')
        if use_pel4vad_contrastive:
            if dataset not in ('ucf', 'cctv') or pel4vad_variant not in ('legacy', 'fixed_bank'):
                raise ValueError('PEL requires UCF/CCTV and a known loss variant')
            if not torch.isfinite(torch.tensor(pel4vad_weight)) or pel4vad_weight < 0:
                raise ValueError('PEL weight must be finite and nonnegative')
            if not torch.isfinite(torch.tensor(pel4vad_temperature)) or pel4vad_temperature <= 0:
                raise ValueError('PEL temperature must be finite and positive')
            if pel4vad_variant == 'fixed_bank' and pel_text_bank is None:
                self.pel_text_bank, _ = build_text_bank(self.model, label_map)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = MultiStepLR(self.optimizer, scheduler_milestones, scheduler_rate)

    def _aggregation_weight(self):
        return aggregation_weight(self.train_loaders, self.dataset, self.aggregation_weight_unit)

    def set_parameters(self, new_params):
        state_dict = self.model.state_dict()
        for key, value in new_params.items():
            state_dict[key] = value.data.clone()
        self.model.load_state_dict(state_dict)

    def training_loss(self, visual, prompts, lengths, labels):
        if not self.use_pel4vad_contrastive:
            mil = CLASM(self.model(visual, prompts, lengths), labels, lengths, self.device)
            return mil, mil, mil.new_zeros(())
        indices = single_class_indices(labels)
        logits, aux = self.model(visual, prompts, lengths, return_pel4vad_aux=True)
        normal_idx = list(self.label_map).index('Normal')
        if self.pel4vad_variant == 'fixed_bank':
            pel = fixed_loss(self, aux['visual_features'], logits, lengths, indices,
                             normal_idx, self._pool_diagnostics)
        else:
            pel = pel4vad_contrastive_loss(aux['visual_features'], aux['text_features'],
                                          logits, normal_idx, indices, indices == normal_idx,
                                          temperature=self.pel4vad_temperature)
        mil = CLASM(logits, labels, lengths, self.device)
        return combine_pel(self, mil, pel), mil, pel

    def get_global_parameters(self):
        new_parameters = OrderedDict()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                new_parameters[name] = p.data.clone()

        return new_parameters

    def train(self):
        self.model.train()
        prepare_pel_round(self)
        self.last_diagnostics = None
        prompt_text = get_prompt_text(self.label_map)

        if self.dataset in ('ucf', 'cctv'):

            if self.local_epochs <= 0 or min(map(len, self.train_loaders)) == 0:
                raise ValueError('Training requires positive epochs and nonempty normal/anomaly loaders')
            totals = {'clasm': 0.0, 'pel': 0.0, 'total': 0.0, 'steps': 0}
            class_counts = torch.zeros(len(self.label_map), dtype=torch.long)

            loss_total2 = 0

            for epoch in range(self.local_epochs):
                normal_iter = iter(self.train_loaders[0])
                anomaly_iter = iter(self.train_loaders[1])

                loss_per_epoch2 = 0

                iters = 0

                for i in range(min(len(self.train_loaders[0]), len(self.train_loaders[1]))):

                    normal_features, normal_label, normal_lengths = next(normal_iter)
                    anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

                    visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(self.device)

                    text_labels = list(normal_label) + list(anomaly_label)

                    feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(self.device)

                    text_labels = get_batch_label(text_labels, prompt_text, self.label_map, self.dataset).to(
                        self.device)
                    measure = self.pel_diagnostics and self.diagnostic_round in DIAGNOSTIC_ROUNDS and epoch == 0 and i == 0
                    diagnostic_start = perf_counter() if measure else None
                    self._pool_diagnostics = {} if measure else None
                    loss2, mil, pel = self.training_loss(visual_features, prompt_text,
                                                        feat_lengths, text_labels)
                    if measure:
                        self.last_diagnostics = {
                            'round': self.diagnostic_round, 'epoch': 1, 'batch': 1,
                            'batch_identity': 'first normal/anomaly batch in epoch 1; existing loader order',
                            'gradient': gradient_diagnostics(self.model, mil, pel, self.active_pel_weight),
                            'gradient_scope': self.pel_options.get('gradient_scope', 'all'),
                            'gradient_measurement': 'raw loss derivatives before optional gradient routing',
                            'pooling': self._pool_diagnostics,
                            'batch_class_counts': dict(zip(self.label_map, text_labels.sum(0).long().cpu().tolist())),
                            'forward_and_diagnostic_seconds': perf_counter() - diagnostic_start}
                    self._pool_diagnostics = None
                    class_counts += text_labels.detach().sum(0).cpu().long()
                    totals['clasm'] += mil.item()
                    totals['pel'] += pel.item()
                    totals['total'] += loss2.item()
                    totals['steps'] += 1
                    loss_per_epoch2 += loss2.item()

                    loss = loss2

                    self.optimizer.zero_grad()
                    backward_pel(self, loss)
                    self.optimizer.step()

                    iters += 1
                loss_total2 += loss_per_epoch2 / iters

            self.last_train_metrics = {key: value / totals['steps'] for key, value in totals.items() if key != 'steps'}
            self.last_train_metrics['steps'] = totals['steps']
            self.last_train_metrics['active_pel_weight'] = self.active_pel_weight
            self.last_train_metrics['weighted_pel'] = self.active_pel_weight * self.last_train_metrics['pel']
            self.last_train_metrics['used_class_counts'] = dict(zip(self.label_map, class_counts.tolist()))
            return (self.get_global_parameters(), loss_total2, self._aggregation_weight())

        elif self.dataset == 'xd':

            loss_total2 = 0

            for epoch in range(self.local_epochs):

                loss_per_epoch2 = 0
                iters = 0

                for i, item in enumerate(self.train_loaders):

                    visual_feat, text_labels, feat_lengths = item
                    visual_feat = visual_feat.to(self.device)
                    feat_lengths = feat_lengths.to(self.device)

                    text_labels = get_batch_label(text_labels, prompt_text, self.label_map, self.dataset).to(
                        self.device)

                    logits = self.model(visual_feat,
                                        prompt_text,
                                        feat_lengths)

                    loss2 = CLASM(logits, text_labels, feat_lengths, self.device)
                    loss_per_epoch2 += loss2.item()

                    loss = loss2

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    iters += 1

                loss_total2 += loss_per_epoch2 / iters

            return (self.get_global_parameters(), loss_total2, self._aggregation_weight())


class FedStrategyClient:
    """Local trainer for every non-legacy federated strategy in this repo.

    Differs from FedAvgClient in one structural way: parameters the model marks as
    *local* (a FedPGP adapter, FedBN-style LayerNorms, or both) are never returned to
    or overwritten by the server, so they persist on this client across rounds. The
    client object is constructed once and reused every round, so persistence is free.

    On top of that it exposes two no-op hooks that the drift-correction strategies
    fill in -- ``_extra_loss`` (FedProx) and ``_before_optimizer_step`` (SCAFFOLD) --
    and records the number of optimizer steps taken, which FedNova needs.
    """

    def __init__(self,
                 model,
                 learning_rate: float,
                 train_loaders: tuple,
                 dataset: str,
                 local_epochs: int,
                 label_map,
                 scheduler_milestones,
                 scheduler_rate,
                 device: str,
                 personalized_lr: float = None,
                 personalized_weight_decay: float = None,
                 use_fedpgp_contrastive: bool = False,
                 contrastive_weight: float = 0.1,
                 contrastive_temperature: float = 0.07,
                 use_pel4vad_contrastive: bool = False,
                 pel4vad_weight: float = 1.0,
                 pel4vad_temperature: float = 0.09,
                 prox_mu: float = 0.0,
                 use_scaffold: bool = False,
                 c_global=None,
                 pel4vad_variant='legacy',
                 pel_text_bank=None,
                 pel_diagnostics=False,
                 pel_options=None,
                 aggregation_weight_unit='loader_batches',
                 ):
        super().__init__()
        if use_pel4vad_contrastive and dataset not in ('ucf', 'cctv'):
            raise ValueError(
                "--use_pel4vad_contrastive assumes a single-label-per-sample dataset "
                f"(argmax over text_labels picks the sample's class); dataset={dataset!r} "
                "uses multi-label '-'-joined text_labels instead")
        self.model = copy.deepcopy(model)
        self.learning_rate = learning_rate
        self.personalized_lr = personalized_lr if personalized_lr is not None else learning_rate
        self.personalized_weight_decay = personalized_weight_decay
        self.train_loaders = train_loaders
        self.dataset = dataset
        self.aggregation_weight_unit = aggregation_weight_unit
        self.local_epochs = local_epochs
        self.label_map = label_map
        self.device = device
        self.use_fedpgp_contrastive = use_fedpgp_contrastive
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.use_pel4vad_contrastive = use_pel4vad_contrastive
        self.pel4vad_weight = pel4vad_weight
        self.pel4vad_temperature = pel4vad_temperature

        self.pel4vad_variant = pel4vad_variant
        self.pel_text_bank = pel_text_bank
        setup_pel_experiment(self, pel_options)
        if use_pel4vad_contrastive and pel4vad_variant == 'fixed_bank' and pel_text_bank is None:
            self.pel_text_bank, _ = build_text_bank(self.model, label_map)
        self.last_train_metrics = {}

        # Same read-only diagnostics FedAvgClient carries, so an aggregation-track run
        # produces a round_metrics/pel_diagnostics pair the existing analysis scripts can
        # read without knowing which server class produced it.
        self.pel_diagnostics = pel_diagnostics
        self.diagnostic_round = None
        self.last_diagnostics = None
        self._pool_diagnostics = None
        self._last_loss_terms = None
        if pel_diagnostics and (dataset not in ('ucf', 'cctv') or
                                (use_pel4vad_contrastive and pel4vad_variant != 'fixed_bank')):
            raise ValueError('PEL diagnostics support UCF/CCTV with PEL off or fixed_bank')

        self.prox_mu = prox_mu
        self.use_scaffold = use_scaffold
        # Reference to the round's global weights. FedProx anchors its proximal term on
        # it and SCAFFOLD needs it for the control update. The server hands the same dict
        # to every client and replaces (never mutates) it each round, so holding a
        # reference costs one copy of the shared params in total rather than one per client.
        self._global_reference = None

        self.local_steps = 0
        self.last_delta_c = None

        self.c_global = c_global
        self.c_local = None
        if use_scaffold:
            self.c_local = OrderedDict(
                (n, torch.zeros_like(p)) for n, p in self.model.named_parameters()
                if p.requires_grad and not self.model._is_local_param(n))

        shared_params = self.model.shared_parameters()
        personalized_params = self.model.personalized_parameters()
        param_groups = [{"params": shared_params, "lr": self.learning_rate}]
        if personalized_params:
            group = {"params": personalized_params, "lr": self.personalized_lr}
            if personalized_weight_decay is not None:
                # Decay acts as a zero-mean prior on the adapter: it only stays nonzero
                # where the client's own data produces a consistent gradient for it.
                # Under an IID split there is no client-specific signal to find, so decay
                # shrinks A@B back toward 0 and the model degrades gracefully to plain
                # FedAvg rather than carrying a noise-only extra parameter block.
                # (AdamW's own default of 1e-2 otherwise applies to every group.)
                group["weight_decay"] = personalized_weight_decay
            param_groups.append(group)
        self.optimizer = torch.optim.AdamW(param_groups)
        self.scheduler = MultiStepLR(self.optimizer, scheduler_milestones, scheduler_rate)

    def _aggregation_weight(self):
        return aggregation_weight(self.train_loaders, self.dataset, self.aggregation_weight_unit)

    def set_parameters(self, new_params):
        """new_params only ever contains shared-parameter keys (the server never aggregates
        or sends local ones), so this never touches the client's local parameters."""
        self.model.load_shared_state_dict(new_params)
        self._global_reference = new_params

    def get_shared_parameters(self):
        return self.model.get_shared_state_dict()

    def get_personalized_parameters(self):
        return self.model.get_personalized_state_dict()

    # -- strategy hooks -----------------------------------------------------

    def _extra_loss(self):
        """FedProx proximal term (Li et al., MLSys 2020): (mu/2) * ||w - w_global||^2.

        Bounds how far a client can drift from the round's starting point during
        local_epochs of training, which is what makes heterogeneous clients' updates
        cancel under plain averaging. Only shared parameters are anchored -- a
        personalized adapter is *supposed* to diverge, so pulling it back to a global
        copy (which does not even exist for it) would defeat its purpose.
        """
        if self.prox_mu <= 0 or self._global_reference is None:
            return 0.0

        reg = None
        for name, param in self.model.named_parameters():
            reference = self._global_reference.get(name)
            if reference is None:
                continue
            term = ((param - reference) ** 2).sum()
            reg = term if reg is None else reg + term

        return 0.0 if reg is None else 0.5 * self.prox_mu * reg

    def _before_optimizer_step(self):
        """SCAFFOLD control-variate correction (Karimireddy et al., ICML 2020).

        Replaces the local gradient g_i with g_i - c_i + c, where c_i estimates this
        client's own update direction and c the global one. The correction cancels the
        client-specific component so every client descends (approximately) the global
        objective rather than its own.

        Caveat: SCAFFOLD is derived for local SGD. This repo optimizes with AdamW, so
        the correction is injected into .grad and then rescaled by Adam's preconditioner
        instead of being applied verbatim. That is the common practical adaptation, not
        the analyzed algorithm -- treat SCAFFOLD numbers here as indicative.
        """
        if not self.use_scaffold:
            return
        for name, param in self.model.named_parameters():
            if param.grad is None or name not in self.c_local:
                continue
            param.grad.add_(self.c_global[name] - self.c_local[name])

    def _update_scaffold_controls(self):
        """Option-II control update: c_i+ = c_i - c + (w_global - w_i) / (K * lr).

        Returns the delta so the server can maintain c = mean_i c_i.
        """
        if not self.use_scaffold or self._global_reference is None:
            return None

        delta_c = OrderedDict()
        denom = max(self.local_steps, 1) * self.learning_rate
        for name, param in self.model.named_parameters():
            if name not in self.c_local:
                continue
            new_c = (self.c_local[name] - self.c_global[name]
                     + (self._global_reference[name] - param.data) / denom)
            delta_c[name] = new_c - self.c_local[name]
            self.c_local[name] = new_c
        return delta_c

    def _step_loss(self, visual_features, feat_lengths, text_labels, prompt_text):
        """text_labels here is already the multi-hot ground-truth matrix from
        get_batch_label (not raw label strings). For ucf/cctv it is one-hot per sample
        (single label per clip), so argmax(dim=1) recovers each sample's class index --
        that's what pel4vad_contrastive_loss uses in place of raw label strings.

        Also records the undifferentiated CLASM and PEL tensors in ``_last_loss_terms`` so
        gradient_diagnostics can take their derivatives separately. Addition order inside
        each branch is left untouched -- these runs are compared against existing ones.
        """
        if not self.use_fedpgp_contrastive and not self.use_pel4vad_contrastive:
            logits = self.model(visual_features, prompt_text, feat_lengths)
            loss_wsvad = CLASM(logits, text_labels, feat_lengths, self.device)
            self._last_loss_terms = (loss_wsvad, loss_wsvad.new_zeros(()))
            return loss_wsvad + self._extra_loss(), loss_wsvad.item(), 0.0

        logits, aux = self.model(visual_features, prompt_text, feat_lengths,
                                  return_contrastive=self.use_fedpgp_contrastive,
                                  return_pel4vad_aux=self.use_pel4vad_contrastive)
        loss_wsvad = CLASM(logits, text_labels, feat_lengths, self.device)
        loss = loss_wsvad
        loss_aux_val = 0.0

        if self.use_fedpgp_contrastive:
            loss_con = fedpgp_contrastive_loss(
                aux["text_features_global"], aux["text_features_personalized"],
                aux["class_features"], self.contrastive_temperature)
            loss = loss + self.contrastive_weight * loss_con
            loss_aux_val += loss_con.item()

        if self.use_pel4vad_contrastive:
            normal_idx = prompt_text.index(self.label_map['Normal'])
            sample_class_idx = text_labels.argmax(dim=1)
            is_normal = sample_class_idx == normal_idx
            if self.pel4vad_variant == 'fixed_bank':
                sample_class_idx = single_class_indices(text_labels)
                loss_pel = fixed_loss(self, aux['visual_features'], logits, feat_lengths,
                                      sample_class_idx, normal_idx, self._pool_diagnostics)
            else:
                loss_pel = pel4vad_contrastive_loss(
                    aux["visual_features"], aux["text_features"], logits, normal_idx,
                    sample_class_idx, is_normal, temperature=self.pel4vad_temperature)
                self._last_loss_terms = (loss_wsvad, loss_pel)
                loss = loss + self.pel4vad_weight * loss_pel
                loss_aux_val += loss_pel.item()
                return loss + self._extra_loss(), loss_wsvad.item(), loss_aux_val
            self._last_loss_terms = (loss_wsvad, loss_pel)
            loss = combine_pel(self, loss + self._extra_loss(), loss_pel)
            loss_aux_val += loss_pel.item()
            return loss, loss_wsvad.item(), loss_aux_val
        self._last_loss_terms = (loss_wsvad, loss_wsvad.new_zeros(()))
        return loss + self._extra_loss(), loss_wsvad.item(), loss_aux_val

    def _record_batch_diagnostics(self, text_labels, started):
        """Read-only: separate derivatives of the two loss terms, no optimizer writes.

        Mirrors FedAvgClient's record so the two servers emit one schema. ``_last_loss_terms``
        holds the raw CLASM and PEL tensors, i.e. before combine_pel's scheduled weight and
        before any gradient-scope routing -- the weight is reported separately.
        """
        clasm, pel = self._last_loss_terms
        self.last_diagnostics = {
            'round': self.diagnostic_round, 'epoch': 1, 'batch': 1,
            'batch_identity': 'first normal/anomaly batch in epoch 1; existing loader order',
            'gradient': gradient_diagnostics(self.model, clasm, pel, self.active_pel_weight),
            'gradient_scope': self.pel_options.get('gradient_scope', 'all'),
            'gradient_measurement': 'raw loss derivatives before optional gradient routing',
            'pooling': self._pool_diagnostics,
            'batch_class_counts': dict(zip(self.label_map,
                                           text_labels.detach().sum(0).long().cpu().tolist())),
            'forward_and_diagnostic_seconds': perf_counter() - started}

    def train(self):
        self.model.train()
        prompt_text = get_prompt_text(self.label_map)

        loss_total2 = 0
        loss_con_total = 0
        self.local_steps = 0
        prepare_pel_round(self)
        self.last_diagnostics = None
        totals = {'clasm': 0.0, 'pel': 0.0, 'total': 0.0, 'steps': 0}
        class_counts = torch.zeros(len(self.label_map), dtype=torch.long)

        if self.dataset in ('ucf', 'cctv'):
            if self.local_epochs <= 0 or min(map(len, self.train_loaders)) == 0:
                raise ValueError('Training requires positive epochs and nonempty normal/anomaly loaders')
            for epoch in range(self.local_epochs):
                normal_iter = iter(self.train_loaders[0])
                anomaly_iter = iter(self.train_loaders[1])
                loss_per_epoch2 = 0
                loss_con_per_epoch = 0
                iters = 0

                for i in range(min(len(self.train_loaders[0]), len(self.train_loaders[1]))):
                    normal_features, normal_label, normal_lengths = next(normal_iter)
                    anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

                    visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(self.device)
                    text_labels = list(normal_label) + list(anomaly_label)
                    feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(self.device)
                    text_labels = get_batch_label(text_labels, prompt_text, self.label_map, self.dataset).to(
                        self.device)

                    measure = (self.pel_diagnostics and self.diagnostic_round in DIAGNOSTIC_ROUNDS
                               and epoch == 0 and i == 0)
                    diagnostic_start = perf_counter() if measure else None
                    self._pool_diagnostics = {} if measure else None
                    loss, loss_wsvad_val, loss_con_val = self._step_loss(
                        visual_features, feat_lengths, text_labels, prompt_text)
                    if measure:
                        self._record_batch_diagnostics(text_labels, diagnostic_start)
                    self._pool_diagnostics = None
                    loss_per_epoch2 += loss_wsvad_val
                    loss_con_per_epoch += loss_con_val
                    class_counts += text_labels.detach().sum(0).cpu().long()
                    totals['clasm'] += loss_wsvad_val
                    totals['pel'] += loss_con_val
                    totals['total'] += loss.item()
                    totals['steps'] += 1

                    self.optimizer.zero_grad()
                    backward_pel(self, loss)
                    self._before_optimizer_step()
                    self.optimizer.step()

                    iters += 1
                    self.local_steps += 1
                loss_total2 += loss_per_epoch2 / iters
                loss_con_total += loss_con_per_epoch / iters

            num_batches = self._aggregation_weight()

        elif self.dataset == 'xd':
            if self.local_epochs <= 0 or len(self.train_loaders) == 0:
                raise ValueError('Training requires positive epochs and a nonempty loader')
            for epoch in range(self.local_epochs):
                loss_per_epoch2 = 0
                loss_con_per_epoch = 0
                iters = 0

                for i, item in enumerate(self.train_loaders):
                    visual_feat, text_labels, feat_lengths = item
                    visual_feat = visual_feat.to(self.device)
                    feat_lengths = feat_lengths.to(self.device)
                    text_labels = get_batch_label(text_labels, prompt_text, self.label_map, self.dataset).to(
                        self.device)

                    loss, loss_wsvad_val, loss_con_val = self._step_loss(
                        visual_feat, feat_lengths, text_labels, prompt_text)
                    loss_per_epoch2 += loss_wsvad_val
                    loss_con_per_epoch += loss_con_val
                    class_counts += text_labels.detach().sum(0).cpu().long()
                    totals['clasm'] += loss_wsvad_val
                    totals['pel'] += loss_con_val
                    totals['total'] += loss.item()
                    totals['steps'] += 1

                    self.optimizer.zero_grad()
                    loss.backward()
                    self._before_optimizer_step()
                    self.optimizer.step()

                    iters += 1
                    self.local_steps += 1
                loss_total2 += loss_per_epoch2 / iters
                loss_con_total += loss_con_per_epoch / iters

            num_batches = self._aggregation_weight()

        else:
            raise ValueError(f"unknown dataset: {self.dataset}")

        if self.use_fedpgp_contrastive or self.use_pel4vad_contrastive:
            print(f"    wsvad_loss: {loss_total2:.4f}, contrastive_loss: {loss_con_total:.4f}")

        self.last_delta_c = self._update_scaffold_controls()
        # Keys match FedAvgClient's so record_event_pel_comparison.py / summarize_pel_runs.py
        # read an aggregation-track run the same way they read a plain-FedAvg one. The two
        # epoch-mean sums the aggregation track already logs are kept alongside them.
        self.last_train_metrics = {key: value / totals['steps'] for key, value in totals.items()
                                   if key != 'steps'}
        self.last_train_metrics.update(
            {'steps': totals['steps'], 'active_pel_weight': self.active_pel_weight,
             'weighted_pel': self.active_pel_weight * self.last_train_metrics['pel'],
             'used_class_counts': dict(zip(self.label_map, class_counts.tolist())),
             'local_steps': self.local_steps,
             'clasm_epoch_mean_sum': loss_total2, 'aux_epoch_mean_sum': loss_con_total})

        return self.get_shared_parameters(), loss_total2, num_batches


# The generalized client subsumes what FedPGPClient used to do (local_param_policy
# 'adapter', no drift correction); keep the old name working for existing callers.
FedPGPClient = FedStrategyClient
