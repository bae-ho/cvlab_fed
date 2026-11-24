"""Read saved diagnostics and compute auditable round/module summaries; no training."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(directory):
    directory = Path(directory)
    rounds = [json.loads(line) for line in (directory / 'round_metrics.jsonl').read_text().splitlines()]
    diagnostics = [json.loads(line) for line in (directory / 'pel_diagnostics.jsonl').read_text().splitlines()]
    manifest = json.loads((directory / 'manifest.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    expected = list(range(1, manifest['arguments']['global_rounds'] + 1))
    assert [r['round'] for r in rounds] == expected
    assert [r['round'] for r in diagnostics] == expected
    count = manifest['arguments']['clients_num']
    rows = []
    for d, r in zip(diagnostics, rounds):
        assert len(r['clients']) == len(d['coefficients']) == count
        assert abs(sum(d['coefficients']) - 1) < 1e-12
        for module, values in d['modules'].items():
            pairs = [p for p in values['pairs'] if p['cosine'] is not None]
            assert len(values['clients']) == count
            negative = sum(p['cosine'] < 0 for p in pairs)
            rows.append({'round': r['round'], 'module': module,
                         # Only float error under plain averaging; under FedNova or a server
                         # optimizer the same number is the rule's deviation from the
                         # weighted mean, so the rule travels with the column.
                         'aggregation_rule': d.get('aggregation_rule', 'weighted_mean'),
                         **{key: values[key] for key in ('retention_ratio', 'weighted_pair_cosine',
                            'aggregate_delta_norm', 'weighted_delta_norm_sum', 'aggregate_rounding_residual_norm')},
                         'negative_pairs': negative, 'valid_pairs': len(pairs),
                         'negative_pair_fraction': negative / len(pairs) if pairs else None,
                         'rounding_residual_relative': values['aggregate_rounding_residual_norm'] / values['aggregate_delta_norm']
                            if values['aggregate_delta_norm'] else None})
    best = max(rounds, key=lambda r: r['evaluation']['roc'])
    ap = max(rounds, key=lambda r: r['evaluation']['ap'])
    assert best['evaluation'] == summary['best_checkpoint_evaluation']
    assert rounds[-1]['evaluation'] == summary['final_evaluation']
    assert ap['evaluation']['ap'] == summary['independent_best_ap']
    return {'schema_version': 1, 'run_id': directory.name,
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in directory.iterdir() if p.name in
                              ('manifest.json', 'summary.json', 'round_metrics.jsonl', 'pel_diagnostics.jsonl')},
            'best_roc_round': best['round'], 'best_ap_round': ap['round'],
            'round_module_metrics': rows,
            'gradient_status_counts': {status: sum(b['gradient']['status'] == status for d in diagnostics
                for b in d['client_batches'] if b) for status in
                sorted({b['gradient']['status'] for d in diagnostics for b in d['client_batches'] if b})}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('Validated complete rounds; wrote', args.output)
