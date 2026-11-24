"""Optional PEL ablations; defaults preserve the existing fixed-bank experiment."""
import math
import torch


def scheduled_weight(target, round_index, warmup=0, ramp=0):
    if not math.isfinite(target) or target < 0 or warmup < 0 or ramp < 0 or round_index < 1:
        raise ValueError('Invalid PEL schedule')
    if round_index <= warmup:
        return 0.0
    return target * min((round_index - warmup) / ramp, 1.0) if ramp else target


def setup_pel_experiment(client, options=None):
    client.pel_options = dict(options or {})
    client.pel_round = 1
    client.active_pel_weight = client.pel4vad_weight
    client._pel_backward_parts = None


def prepare_pel_round(client):
    client.active_pel_weight = scheduled_weight(
        client.pel4vad_weight, client.pel_round,
        client.pel_options.get('warmup_rounds', 0), client.pel_options.get('ramp_rounds', 0))
    client._pel_backward_parts = None


def fixed_loss(client, features, logits, lengths, indices, normal_idx, diagnostics=None):
    from utils.fixed_pel import fixed_bank_pel_loss
    head = getattr(client.model, 'pel_projection', None)
    if head is not None:
        features = head(features)
    return fixed_bank_pel_loss(
        features, logits, lengths, client.pel_text_bank, indices, normal_idx,
        client.pel4vad_temperature, diagnostics=diagnostics,
        pooling=client.pel_options.get('pooling', 'score'),
        class_balance=client.pel_options.get('class_balance', 'none'))


def combine_pel(client, base, pel):
    if client.active_pel_weight == 0:
        return base
    if client.pel_options.get('gradient_scope', 'all') == 'last_block':
        client._pel_backward_parts = (base, pel)
    return base + client.active_pel_weight * pel


def backward_pel(client, total):
    """Restrict only PEL gradients; retain every base-loss gradient and optimizer policy."""
    if client._pel_backward_parts is None:
        total.backward()
        return
    base, pel = client._pel_backward_parts
    blocks = getattr(client.model.temporal, 'resblocks', None)
    if blocks is None or len(blocks) == 0:
        raise ValueError('last_block requires Temporal Transformer resblocks')
    params = list(blocks[-1].parameters())
    head = getattr(client.model, 'pel_projection', None)
    if head is not None:
        params += list(head.parameters())
    params = [p for p in params if p.requires_grad]
    gradients = torch.autograd.grad(pel, params, retain_graph=True, allow_unused=True)
    base.backward()
    for param, grad in zip(params, gradients):
        if grad is not None:
            update = client.active_pel_weight * grad.detach()
            if param.grad is None:
                param.grad = update.clone()
            else:
                param.grad.add_(update)
    client._pel_backward_parts = None


def cli_pel_options(args):
    return {'warmup_rounds': args.pel_warmup_rounds, 'ramp_rounds': args.pel_ramp_rounds,
            'pooling': args.pel_pooling, 'class_balance': args.pel_class_balance,
            'gradient_scope': args.pel_gradient_scope}


def plain_fedavg_configuration(args):
    """The PEL track's own baseline: every trainable parameter averaged, no server step.

    This is what the fixed-bank experiments were validated against, so it stays reachable
    without any opt-in flag.
    """
    return (args.aggregation_method == 'fedavg' and args.server_optimizer == 'none'
            and args.personalization_type == 'none'
            and args.local_param_policy in (None, 'none'))


def aggregation_integration(args):
    """True when the run deliberately combines PEL with the aggregation track.

    The aggregation track varies three independent axes (aggregation rule, server
    optimizer, local-parameter policy). Opting in unblocks all three rather than one named
    pair, because reading PEL's effect *under* a given rule requires running the same PEL
    configuration across several of them.
    """
    return bool(args.pel_allow_aggregation_integration)


def validate_pel_cli(args):
    if args.pel_warmup_rounds < 0 or args.pel_ramp_rounds < 0:
        raise ValueError('PEL schedule rounds must be nonnegative')
    ablation = any((args.pel_warmup_rounds, args.pel_ramp_rounds, args.pel_projection,
                    args.pel_pooling != 'score', args.pel_class_balance != 'none',
                    args.pel_gradient_scope != 'all'))
    if ablation and (not args.use_pel4vad_contrastive or args.pel4vad_variant != 'fixed_bank'):
        raise ValueError('PEL ablations require enabled fixed_bank PEL')
    plain = plain_fedavg_configuration(args)
    integration = aggregation_integration(args)
    if integration and plain:
        raise ValueError('Integration flag requires a non-baseline aggregation configuration; '
                          'set --aggregation_method / --server_optimizer / --personalization_type, '
                          'or drop --pel_allow_aggregation_integration')
    if args.pel4vad_variant == 'fixed_bank':
        # use_fedpgp_contrastive stays blocked on purpose: it is a second auxiliary loss on
        # the same shared prompt generator, so stacking it under PEL makes the gradient
        # diagnostics unattributable -- the exact confound they exist to measure.
        if (not args.use_pel4vad_contrastive or args.dataset not in ('ucf', 'cctv') or
                args.algorithm != 'FedAvg' or args.use_fedpgp_contrastive or
                not (plain or integration)):
            raise ValueError('fixed_bank requires plain FedAvg, or an explicit aggregation '
                              'integration via --pel_allow_aggregation_integration')
    if args.pel4vad_prompts and args.pel4vad_variant != 'fixed_bank':
        raise ValueError('pel4vad_prompts is only used by fixed_bank')


def diagnostics_supported(args):
    """pel_diagnostics is read-only, so it travels with any configuration whose client
    exposes a CLASM/PEL split: the plain baseline, or an opted-in aggregation run."""
    return (args.dataset in ('ucf', 'cctv') and args.algorithm == 'FedAvg'
            and not args.use_fedpgp_contrastive
            and not (args.use_pel4vad_contrastive and args.pel4vad_variant != 'fixed_bank')
            and (plain_fedavg_configuration(args) or aggregation_integration(args)))
