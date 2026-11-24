"""Server-side aggregation rules and server optimizers.

Following the decomposition in Reddi et al., "Adaptive Federated Optimization"
(ICLR 2021), a federated round is split into two orthogonal choices:

  1. How the per-client models are reduced into one *pseudo-gradient*
         delta = (combination of w_i) - w_global
     -- see ``weighted_average`` / ``fednova_delta`` below.

  2. How the server applies that pseudo-gradient to the global model
     -- see the ``ServerOptimizer`` subclasses below.

Plain FedAvg is exactly (weighted_average, PlainServerOptimizer); every other
combination is a strictly more general server that reuses the same client code.
"""

from collections import OrderedDict

import torch


# ---------------------------------------------------------------------------
# Aggregation rules: local models -> pseudo-gradient
# ---------------------------------------------------------------------------

def weighted_average(local_weights, weights):
    """Standard FedAvg reduction: sum_i (n_i / sum_j n_j) * w_i."""
    total = sum(weights)
    out = OrderedDict()
    for key, value in local_weights[0].items():
        out[key] = torch.zeros_like(value)
    for w, n in zip(local_weights, weights):
        scale = n / total
        for key, value in w.items():
            out[key] += value * scale
    return out


def subtract(a, b):
    """Elementwise a - b over matching state dicts."""
    return OrderedDict((key, a[key] - b[key]) for key in a)


def fednova_delta(global_params, local_weights, taus, weights):
    """FedNova (Wang et al., NeurIPS 2020): normalized averaging.

    Vanilla FedAvg averages the *end points* w_i. A client that ran more local
    steps travels further from w_global, so it pulls the average harder purely
    because it holds more data -- on top of already receiving a larger weight
    n_i. The fixed point therefore depends on the step counts, not only on the
    data distribution ("objective inconsistency").

    This matters a lot in this repo: with batch_size 64 and drop_last, the UCF
    scene split gives clients between 1 and 41 optimizer steps per epoch (a 41x
    spread), and the event split between 4 and 22.

    FedNova removes the bias by normalizing each client's accumulated update by
    its own step count before averaging, then rescaling by the weighted-average
    step count so the overall step size is preserved:

        d_i     = (w_i - w_global) / tau_i
        tau_eff = sum_i p_i * tau_i
        delta   = tau_eff * sum_i p_i * d_i

    Returns the update to add to w_global (not a gradient), matching the sign
    convention used by the ServerOptimizer classes.
    """
    total = sum(weights)
    tau_eff = sum((n / total) * t for n, t in zip(weights, taus))

    delta = OrderedDict()
    for key, value in global_params.items():
        delta[key] = torch.zeros_like(value)

    for w, n, tau in zip(local_weights, weights, taus):
        scale = (n / total) / max(tau, 1)
        for key in delta:
            delta[key] += (w[key] - global_params[key]) * scale

    for key in delta:
        delta[key] *= tau_eff

    return delta


# ---------------------------------------------------------------------------
# Server optimizers: pseudo-gradient -> new global model
# ---------------------------------------------------------------------------

class ServerOptimizer:
    """Applies an aggregated pseudo-gradient to the global model.

    ``delta`` is an *update direction* (i.e. a negative gradient): plain FedAvg
    is w <- w_global + delta.
    """

    def step(self, global_params, delta):
        raise NotImplementedError


class PlainServerOptimizer(ServerOptimizer):
    """FedAvg / FedSGD: apply the aggregated update as-is."""

    def step(self, global_params, delta):
        return OrderedDict((key, global_params[key] + delta[key]) for key in global_params)


class MomentumServerOptimizer(ServerOptimizer):
    """FedAvgM (Hsu et al., 2019): heavy-ball momentum on the server.

    Under Non-IID splits consecutive rounds pull in partly conflicting
    directions, so the aggregated update oscillates. Accumulating it in a
    momentum buffer cancels the oscillating component and reinforces the
    consistent one, which is the cheapest known fix for client drift -- it
    costs one extra copy of the shared parameters on the server and nothing
    at all on the clients.
    """

    def __init__(self, lr=1.0, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.momentum = None

    def step(self, global_params, delta):
        if self.momentum is None:
            self.momentum = OrderedDict((k, torch.zeros_like(v)) for k, v in global_params.items())

        new_params = OrderedDict()
        for key in global_params:
            self.momentum[key] = self.beta * self.momentum[key] + delta[key]
            new_params[key] = global_params[key] + self.lr * self.momentum[key]
        return new_params


class AdamServerOptimizer(ServerOptimizer):
    """FedAdam (Reddi et al., ICLR 2021): Adam on the server-side pseudo-gradient.

    Rescales the update per-coordinate by its recent second moment. Parameters
    that receive consistent updates across clients move at the full server
    learning rate, while coordinates where clients disagree (large variance)
    are damped -- which is exactly the failure mode plain averaging suffers
    from under heterogeneous data.

    Note ``lr`` here is an absolute step size (defaults around 1e-2), not the
    1.0 that plain/momentum servers use, because the update is normalized.
    """

    def __init__(self, lr=1e-2, beta1=0.9, beta2=0.99, tau=1e-3):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m = None
        self.v = None

    def _init_state(self, global_params):
        self.m = OrderedDict((k, torch.zeros_like(v)) for k, v in global_params.items())
        self.v = OrderedDict((k, torch.zeros_like(v)) for k, v in global_params.items())

    def _update_second_moment(self, key, delta_sq):
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * delta_sq

    def step(self, global_params, delta):
        if self.m is None:
            self._init_state(global_params)

        new_params = OrderedDict()
        for key in global_params:
            d = delta[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * d
            self._update_second_moment(key, d * d)
            new_params[key] = global_params[key] + self.lr * self.m[key] / (self.v[key].sqrt() + self.tau)
        return new_params


class YogiServerOptimizer(AdamServerOptimizer):
    """FedYogi (Reddi et al., ICLR 2021).

    Same as FedAdam except the second moment can only decrease when the current
    squared update is smaller than the running estimate, so v grows more slowly
    and the effective step size decays less aggressively than Adam's.
    """

    def _update_second_moment(self, key, delta_sq):
        v = self.v[key]
        self.v[key] = v - (1 - self.beta2) * delta_sq * torch.sign(v - delta_sq)


SERVER_OPTIMIZER_DEFAULT_LR = {
    "none": 1.0,
    "momentum": 1.0,
    "adam": 1e-2,
    "yogi": 1e-2,
}


def build_server_optimizer(name, lr=None, beta1=0.9, beta2=0.99, tau=1e-3):
    if lr is None:
        lr = SERVER_OPTIMIZER_DEFAULT_LR[name]

    if name == "none":
        return PlainServerOptimizer()
    if name == "momentum":
        return MomentumServerOptimizer(lr=lr, beta=beta1)
    if name == "adam":
        return AdamServerOptimizer(lr=lr, beta1=beta1, beta2=beta2, tau=tau)
    if name == "yogi":
        return YogiServerOptimizer(lr=lr, beta1=beta1, beta2=beta2, tau=tau)
    raise ValueError(f"unknown server_optimizer: {name}")
