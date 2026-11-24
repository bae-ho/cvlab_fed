"""Read-only PEL diagnostics; no optimizer writes, random draws or extra forwards."""
import math
import torch


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _cosine(a, b):
    denom = a.norm() * b.norm()
    return _number(torch.dot(a, b) / denom) if denom > 0 else None


def _group(name):
    return name.split('.')[0]


def gradient_diagnostics(model, clasm, pel, weight):
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad
                  and _group(n) in ('temporal', 'frame_position_embeddings', 'pel_projection')]
    if not parameters:
        return {'status': 'no_visual_parameters', 'modules': {}}
    tensors = [p for _, p in parameters]
    left = torch.autograd.grad(clasm, tensors, retain_graph=True, allow_unused=True)
    right = (torch.autograd.grad(pel, tensors, retain_graph=True, allow_unused=True)
             if pel.requires_grad else [None] * len(tensors))
    result = {}
    for group in sorted({_group(n) for n, _ in parameters} | {'all_visual'}):
        indices = [i for i, (n, _) in enumerate(parameters)
                   if group == 'all_visual' or _group(n) == group]
        def flatten(values):
            return torch.cat([(values[i].detach() if values[i] is not None
                               else torch.zeros_like(tensors[i])).reshape(-1).double().cpu()
                              for i in indices])
        a, b = flatten(left), flatten(right)
        an, bn = a.norm(), b.norm()
        result[group] = {'clasm_norm': _number(an), 'pel_norm': _number(bn),
                         'weighted_pel_norm': _number(bn * weight),
                         'pel_to_clasm_norm': _number(bn / an) if an > 0 else None,
                         'weighted_pel_to_clasm_norm': _number(bn * weight / an) if an > 0 else None,
                         'cosine': _cosine(a, b) if pel.requires_grad else None,
                         'parameter_names': [parameters[i][0] for i in indices]}
    return {'status': 'measured' if pel.requires_grad else 'pel_disabled', 'modules': result}


def rule_coefficients(counts, taus=None):
    """Per-client coefficient the server's aggregation rule actually applies to (w_i - w).

    Plain weighted averaging uses p_i = n_i / sum(n). FedNova divides each client's update
    by its own step count and rescales by the weighted-average step count, i.e.
    tau_eff * p_i / tau_i. Reporting both lets a reader see how far the rule moved the
    effective weighting away from the batch-count weighting the deltas are measured with.
    """
    total = sum(counts)
    weights = [count / total for count in counts]
    if taus is None:
        return weights
    tau_eff = sum(p * t for p, t in zip(weights, taus))
    return [tau_eff * p / max(t, 1) for p, t in zip(weights, taus)]


@torch.no_grad()
def update_diagnostics(start, clients, counts, aggregate, aggregation='weighted_mean', taus=None,
                       unit='loader_batches'):
    """Deltas from the same round-start anchor, weighted by the counts the server was given.

    ``coefficients`` follow ``counts``, i.e. the configured ``--aggregation_weight_unit``.
    What they deliberately do *not* follow is the aggregation *rule*: even under FedNova or
    a server optimizer they stay the plain weighted mean, so ``retention_ratio`` keeps
    answering "how much do these clients cancel each other", which is a property of the
    updates rather than of the rule applied to them. Comparing retention across runs is
    therefore only meaningful at a fixed ``unit``, which is recorded alongside.

    ``aggregate_rounding_residual_norm`` is ||actual - sum_i p_i v_i||. Under
    ``aggregation='weighted_mean'`` that is float error only. Under any other rule it is the
    rule's own deviation from the weighted mean and is *expected* to be large, so the rule
    name is recorded next to it rather than left for the reader to infer.
    """
    coefficients = [count / sum(counts) for count in counts]
    groups = sorted({_group(n) for n in start}) + ['all_shared']
    result = {}
    for group in groups:
        keys = [n for n in start if group == 'all_shared' or _group(n) == group]
        def delta(state):
            return torch.cat([(state[n].detach().double().cpu() - start[n].double().cpu())
                              .reshape(-1) for n in keys])
        vectors = [delta(state) for state in clients]
        actual = delta(aggregate)
        expected = sum((a * v for a, v in zip(coefficients, vectors)), torch.zeros_like(actual))
        denominator = sum(a * v.norm() for a, v in zip(coefficients, vectors))
        result[group] = {
            'aggregate_delta_norm': _number(actual.norm()),
            'weighted_delta_norm_sum': _number(denominator),
            'retention_ratio': _number(expected.norm() / denominator) if denominator > 0 else None,
            'aggregate_rounding_residual_norm': _number((actual - expected).norm()),
            'clients': [{'client': i + 1, 'delta_norm': _number(v.norm()),
                         'cosine_with_aggregate': _cosine(v, actual)} for i, v in enumerate(vectors)],
            'pairs': [{'clients': [i + 1, j + 1], 'cosine': _cosine(vectors[i], vectors[j]),
                       'coefficient_product': coefficients[i] * coefficients[j]}
                      for i in range(len(vectors)) for j in range(i + 1, len(vectors))]}
        valid_pairs = [pair for pair in result[group]['pairs'] if pair['cosine'] is not None]
        pair_weight = sum(pair['coefficient_product'] for pair in valid_pairs)
        result[group]['weighted_pair_cosine'] = (
            sum(pair['coefficient_product'] * pair['cosine'] for pair in valid_pairs) / pair_weight
            if pair_weight > 0 else None)
    return {'aggregation_unit': unit,
            'aggregation_rule': aggregation, 'local_steps': taus,
            'coefficients': coefficients,
            'rule_coefficients': rule_coefficients(counts, taus),
            'modules': result}
