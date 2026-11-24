"""Generate reviewable commands only. This script NEVER launches training."""
import argparse
import json
from pathlib import Path
import shlex


# Integration arms: (aggregation_method, personalization_type, local_param_policy, extra).
# 'fedavg'/'fednova'/'fedpgp' are the original three and keep their exact argv. The two
# aggregation-track arms add what that track actually proposes -- FedNova *with* the
# low-rank adapter, and the same again under server momentum. Their extras come last on the
# command line on purpose: argparse takes the final occurrence, so they override the
# '--server_optimizer none' in `common` without disturbing the other arms' argv.
INTEGRATION_ARMS = {
    'fedavg': ('fedavg', 'none', 'none', []),
    'fednova': ('fednova', 'none', 'none', []),
    'fedpgp': ('fedpgp', 'low_rank', 'adapter', []),
    'fednova_adapter': ('fednova', 'low_rank', 'adapter', []),
    'fednova_adapter_momentum': ('fednova', 'low_rank', 'adapter',
                                 ['--server_optimizer', 'momentum']),
}
DEFAULT_INTEGRATION_ARMS = ('fedavg', 'fednova', 'fedpgp')

# Training protocols. 'pel_track' is this repo's own, and every existing record in
# docs/experiments/ was produced with it -- its values are fixed so previously generated
# commands stay byte-for-byte identical. 'aggregation_track' mirrors what the aggregation
# track actually ran for UCF (read off save/*/README.txt: batch 64, lr 1e-5, 15 rounds,
# 10 local epochs, visual_layers 2 by default, seed 88888888), so the combined runs can be
# read against that track's own table instead of against this one's baselines.
PROTOCOLS = {
    'pel_track': {'batch_size': '64', 'learning_rate': '0.0001', 'global_rounds': '20'},
    'aggregation_track': {'batch_size': '64', 'learning_rate': '1e-05', 'global_rounds': '15'},
}


def commands(stage, weight=None, arms=None, diagnostics=False, protocol='pel_track',
             weight_unit=None):
    if protocol not in PROTOCOLS:
        raise ValueError(f'unknown protocol: {protocol!r}')
    settings = PROTOCOLS[protocol]
    common = ['python', '-u', 'train.py', '--dataset', 'ucf',
              '--batch_size', settings['batch_size'],
              '--learning_rate', settings['learning_rate'],
              '--global_rounds', settings['global_rounds'], '--local_epochs', '10',
              '--server_optimizer', 'none', '--cuda', '0']
    if weight_unit is not None:
        common += ['--aggregation_weight_unit', weight_unit]
    variants = []
    if stage == 'strength':
        variants = [('P1', ['--pel4vad_weight', '0.1']), ('P2', ['--pel4vad_weight', '0.3'])]
    else:
        if weight is None:
            raise ValueError('Pass --selected-weight after reviewing strength results')
        if stage == 'timing':
            variants = [('P3', ['--pel_warmup_rounds', '3', '--pel_ramp_rounds', '3'])]
        elif stage == 'pooling':
            variants = [('score', []), ('uniform', ['--pel_pooling', 'uniform']),
                        ('balanced', ['--pel_class_balance', 'batch'])]
        elif stage == 'modules':
            variants = [('all', []), ('last_block', ['--pel_gradient_scope', 'last_block']),
                        ('projection', ['--pel_projection'])]
        else:
            variants = [('baseline', None), ('selected_pel', [])]
    selected_arms = (tuple(arms or DEFAULT_INTEGRATION_ARMS) if stage == 'integration'
                     else ('fedavg',))
    unknown = [arm for arm in selected_arms if arm not in INTEGRATION_ARMS]
    if unknown:
        raise ValueError(f'unknown integration arm(s): {", ".join(unknown)}')
    seeds = [42, 2026] if stage == 'repeat' else [88888888]
    rows = []
    for arm in selected_arms:
        method, personalization, policy, arm_extra = INTEGRATION_ARMS[arm]
        for name, extra in variants:
            for split, clients in [('event', '13'), ('scene', '9')]:
                for seed in seeds:
                    argv = common + ['--aggregation_method', method, '--split_mode', split,
                                     '--clients_num', clients, '--seed', str(seed),
                                     '--personalization_type', personalization,
                                     '--local_param_policy', policy]
                    if arm != 'fedavg':
                        argv += ['--pel_allow_aggregation_integration']
                    if extra is not None:
                        argv += ['--use_pel4vad_contrastive', '--pel4vad_variant', 'fixed_bank',
                                 '--pel4vad_temperature', '0.09']
                        if weight is not None:
                            argv += ['--pel4vad_weight', str(weight)]
                        argv += extra
                    if diagnostics:
                        # Required by record_event_pel_comparison.py; read-only, so a
                        # baseline and its PEL candidate stay step-for-step comparable.
                        argv += ['--pel_diagnostics']
                    argv += arm_extra
                    rows.append({'stage': stage, 'condition': name, 'split': split,
                                 'aggregation': method, 'arm': arm, 'protocol': protocol,
                                 'seed': seed, 'argv': argv, 'command': shlex.join(argv)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=['strength', 'timing', 'pooling', 'modules', 'repeat', 'integration'])
    parser.add_argument('--selected-weight', type=float)
    parser.add_argument('--arms', nargs='+', choices=sorted(INTEGRATION_ARMS),
                        help='integration stage only; defaults to '
                             f'{" ".join(DEFAULT_INTEGRATION_ARMS)}')
    parser.add_argument('--diagnostics', action='store_true',
                        help='add --pel_diagnostics to every generated command')
    parser.add_argument('--protocol', choices=sorted(PROTOCOLS), default='pel_track',
                        help="training protocol; 'aggregation_track' uses that track's UCF "
                             "settings (lr 1e-5, 15 rounds) so results read against its table")
    parser.add_argument('--weight-unit', choices=['loader_batches', 'effective_batches'],
                        help='pin --aggregation_weight_unit explicitly instead of relying on '
                             'the default (loader_batches)')
    parser.add_argument('--selected-options', nargs='*', default=[],
                        help='Exact additional selected PEL arguments; use = and quoted strings if needed')
    parser.add_argument('--output', type=Path, help='Optional JSON command manifest; no execution')
    args = parser.parse_args()
    if args.selected_weight is not None and not 0 < args.selected_weight < float('inf'):
        parser.error('selected-weight must be finite and positive')
    if args.arms and args.stage != 'integration':
        parser.error('--arms applies to the integration stage only')
    try:
        rows = commands(args.stage, args.selected_weight, args.arms, args.diagnostics,
                        args.protocol, args.weight_unit)
    except ValueError as error:
        parser.error(str(error))
    selected = [part for option in args.selected_options for part in shlex.split(option)]
    allowed = {'--pel_warmup_rounds', '--pel_ramp_rounds', '--pel_pooling',
               '--pel_class_balance', '--pel_gradient_scope', '--pel_projection'}
    if any(part.startswith('--') and part.split('=')[0] not in allowed for part in selected):
        parser.error('selected-options may only contain PEL schedule, pooling, balance, gradient scope or projection')
    selection_parser = argparse.ArgumentParser(add_help=False)
    selection_parser.add_argument('--pel_warmup_rounds', type=int, default=0)
    selection_parser.add_argument('--pel_ramp_rounds', type=int, default=0)
    selection_parser.add_argument('--pel_pooling', choices=['score', 'uniform'])
    selection_parser.add_argument('--pel_class_balance', choices=['none', 'batch'])
    selection_parser.add_argument('--pel_gradient_scope', choices=['all', 'last_block'])
    selection_parser.add_argument('--pel_projection', action='store_true')
    selection_args = selection_parser.parse_args(selected)
    if selection_args.pel_warmup_rounds < 0 or selection_args.pel_ramp_rounds < 0:
        parser.error('selected schedule rounds must be nonnegative')
    if selected and args.stage not in ('repeat', 'integration'):
        parser.error('selected-options are only applied to repeat/integration stages')
    for row in rows:
        if row['condition'] == 'selected_pel':
            row['argv'] += selected
            row['command'] = shlex.join(row['argv'])
    payload = {'schema_version': 1, 'execution': 'commands_only', 'experiments': rows}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    else:
        print(rendered)


if __name__ == '__main__':
    main()
