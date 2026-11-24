"""Summarize existing run files, pairing seeds only within identical conditions."""
import argparse
import json
import statistics
from pathlib import Path


def summarize(directories):
    groups = {}
    for directory in directories:
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        rows = [json.loads(line) for line in (directory / 'round_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        if not rows:
            raise ValueError(f'Empty round metrics: {directory}')
        args = manifest['arguments']
        rounds = [row['round'] for row in rows]
        if rounds != list(range(1, int(args['global_rounds']) + 1)):
            raise ValueError(f'Incomplete or duplicate rounds: {directory}')
        condition = {k: v for k, v in args.items() if k not in ('seed', 'cuda', 'pel_diagnostics', 'verbose_fed_logging')}
        enabled = condition.pop('use_pel4vad_contrastive', False)
        # Pair baseline/candidate only when all non-PEL conditions are identical.
        pel = {k: condition.pop(k) for k in list(condition) if k.startswith('pel')}
        if enabled:
            pel['bank_sha256'] = (manifest.get('pel_bank') or {}).get('sha256')
        condition['execution_context'] = {k: manifest.get(k) for k in ('git_revision', 'torch', 'server')}
        key = json.dumps(condition, sort_keys=True)
        entry = groups.setdefault(key, {'conditions': condition, 'baseline': {}, 'candidates': {}})
        target = (entry['candidates'].setdefault(json.dumps(pel, sort_keys=True), {'pel': pel, 'runs': {}})['runs']
                  if enabled else entry['baseline'])
        seed = str(args['seed'])
        if seed in target:
            raise ValueError(f'Duplicate seed in same condition: {directory}')
        best = max(rows, key=lambda row: row['evaluation']['roc'])
        best_ap = max(rows, key=lambda row: row['evaluation']['ap'])
        final = rows[-1]
        target[seed] = {'directory': str(directory), 'best_roc': best['evaluation']['roc'],
                        'git_revision': manifest.get('git_revision'), 'git_dirty': manifest.get('git_dirty'),
                        'selected_ap': best['evaluation']['ap'], 'best_roc_round': best['round'],
                        'best_ap': best_ap['evaluation']['ap'], 'best_ap_round': best_ap['round'],
                        'final_roc': final['evaluation']['roc'], 'final_ap': final['evaluation']['ap']}
    output = []
    for entry in groups.values():
        for candidate in entry['candidates'].values():
            seeds = sorted(set(entry['baseline']) & set(candidate['runs']))
            deltas = {metric: [candidate['runs'][seed][metric] - entry['baseline'][seed][metric]
                              for seed in seeds]
                      for metric in ('best_roc', 'selected_ap', 'best_ap', 'final_roc', 'final_ap')}
            output.append({'conditions': entry['conditions'], 'pel': candidate['pel'], 'paired_seeds': seeds,
                           'unpaired_baseline_seeds': sorted(set(entry['baseline']) - set(seeds)),
                           'unpaired_candidate_seeds': sorted(set(candidate['runs']) - set(seeds)),
                           'baseline_runs': entry['baseline'], 'candidate_runs': candidate['runs'],
                           'delta_statistics': {metric: {'n': len(values), 'seed_deltas': values,
                                                        'mean': statistics.mean(values) if values else None,
                                                        'sample_std': statistics.stdev(values) if len(values) > 1 else None}
                                                for metric, values in deltas.items()},
                           'selection': 'best ROC on test; exploratory, not validation-selected',
                           'verification_limit': 'Data identity and dirty-tree content require independent verification'})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rendered = json.dumps(summarize(args.directories), ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    else:
        print(rendered)


if __name__ == '__main__':
    main()
