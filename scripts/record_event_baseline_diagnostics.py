"""Record the imported 2026-10-13 event baseline run; never starts training."""
import json
from pathlib import Path

root = Path('artifacts/experiment-metrics/2026-10-13_ucf_event_baseline_fedavg_seed88888888_round20_run002')
analysis = json.loads((root / 'analysis.json').read_text(encoding='utf-8'))
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
rounds = [json.loads(line) for line in (root / 'round_metrics.jsonl').read_text().splitlines()]
diagnostics = [json.loads(line) for line in (root / 'pel_diagnostics.jsonl').read_text().splitlines()]
measurements = []
prefix = '../../' + root.as_posix() + '/'


def add(metric, value, unit, round_number, selection='per_round', conditions=None,
        source='round_metrics.jsonl', provenance='observed'):
    measurements.append(dict(method='Fed-WSVAD', dataset='ucf', split='event', metric=metric,
                             value=value, unit=unit, seed=88888888, round=round_number,
                             statistic='single', selection=selection, provenance=provenance,
                             source=prefix + source + (f':{round_number}' if source.endswith('.jsonl') else ''),
                             conditions=conditions or {}))


for r in rounds:
    for key, metric in [('roc', 'roc_auc'), ('ap', 'ap')]:
        add(metric, r['evaluation'][key], 'ratio', r['round'])
    for client, c in enumerate(r['clients'], 1):
        for key, metric, unit in [('clasm', 'mil_classification_loss', 'unitless'),
                                  ('pel', 'pel_loss', 'unitless'), ('total', 'total_loss', 'unitless'),
                                  ('steps', 'local_steps', 'count')]:
            add(metric, c[key], unit, r['round'], conditions={'client': client})
for row in analysis['round_module_metrics']:
    for key in ['retention_ratio', 'weighted_pair_cosine', 'aggregate_delta_norm',
                'weighted_delta_norm_sum', 'aggregate_rounding_residual_norm',
                'negative_pairs', 'valid_pairs', 'negative_pair_fraction']:
        add(key, row[key], 'count' if key.endswith('pairs') else 'unitless', row['round'],
            conditions={'module': row['module']}, source='pel_diagnostics.jsonl',
            provenance='derived' if key in ('negative_pairs', 'valid_pairs', 'negative_pair_fraction') else 'observed')
for d in diagnostics:
    for client, batch in enumerate(d['client_batches'], 1):
        if not batch:
            continue
        for module, values in batch['gradient']['modules'].items():
            for key in ('clasm_norm', 'pel_norm', 'cosine'):
                add({'clasm_norm': 'mil_gradient_norm', 'pel_norm': 'pel_gradient_norm',
                     'cosine': 'mil_pel_gradient_cosine'}[key], values[key], 'unitless', d['round'],
                    selection='first_batch', conditions={'client': client, 'module': module,
                    'epoch': batch['epoch'], 'batch': batch['batch'], 'gradient_status': batch['gradient']['status']},
                    source='pel_diagnostics.jsonl')
for selection, number in [('best_roc_round', analysis['best_roc_round']),
                          ('final', rounds[-1]['round'])]:
    for key, metric in [('roc', 'roc_auc'), ('ap', 'ap')]:
        add(metric, rounds[number-1]['evaluation'][key], 'ratio', number, selection)
add('ap', rounds[analysis['best_ap_round']-1]['evaluation']['ap'], 'ratio', analysis['best_ap_round'], 'independent_best_ap')

record = dict(schema_version=1, date='2026-10-13', date_kind='experiment',
              experiment_type='ucf_event_baseline', attempt=2, track='pel', status='completed',
              conditions={'aggregation_method': 'fedavg', 'pel_enabled': False,
                          'pel_diagnostics': True, 'clients_num': 13, 'global_rounds': 20,
                          'local_epochs': 10, 'batch_size': 64, 'learning_rate': 0.0001,
                          'server_optimizer': 'none', 'personalization_type': 'none',
                          'local_param_policy': 'none', 'torch': manifest['torch'],
                          'git_revision': manifest['git_revision'], 'git_dirty': manifest['git_dirty'],
                          'original_run_id': 'ucf-2026-10-13-17-48-25-320401',
                          'date_evidence': 'original save folder timestamp; end time unavailable',
                          'checkpoint_selection': 'best round on test set (existing protocol)'},
              measurements=measurements)
shared = [r for r in analysis['round_module_metrics'] if r['module'] == 'all_shared']
table = '| Round | ROC-AUC (%) | AP (%) | 집계 보존 비율 | 가중 pair cosine | 음의 pair / 78 |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
for row in shared:
    if row['round'] not in (1, 5, 10, 15, 20):
        continue
    ev = rounds[row['round']-1]['evaluation']
    table += f"| {row['round']} | {ev['roc']*100:.4f} | {ev['ap']*100:.4f} | {row['retention_ratio']:.6f} | {row['weighted_pair_cosine']:.6f} | {row['negative_pairs']} |\n"
lo, hi = min(r['retention_ratio'] for r in shared), max(r['retention_ratio'] for r in shared)
interpretation = f'''- 관찰: 전체 공유 파라미터의 집계 보존 비율 범위 {lo:.6f}–{hi:.6f}; round 5에서 가중 pair cosine -0.009358, 음의 pair 30/78.
- 해석: PEL이 없어도 client 업데이트의 방향 불일치와 부분 상쇄 존재. 낮은 보존 비율을 유용한 특징 소실이나 성능 저하의 원인으로 단정 불가.
- 미확인: PEL 추가에 따른 변화 및 MIL–PEL 손실 경쟁. 65개 진단 batch 모두 `pel_disabled`; PEL norm 0과 cosine null은 비활성 상태.
- 모듈별 결측: `global_text_prompt_embeddings`의 업데이트 norm 0으로 보존 비율·pair cosine 결측; 0으로 치환하지 않음.
- 성능: 최고 ROC 85.27366579931893% (round 18), 해당 AP 26.167553477541894%; 독립 최고 AP 28.65852068513203% (round 19); 최종 ROC 85.23379086512399%, AP 26.65676698745457%.
- 기존 기준 실행의 20개 round ROC/AP와 정확 일치. 환경·시작 날짜가 다른 새 실행이므로 기존 run001을 덮어쓰지 않고 run002로 기록.
- 제한: manifest의 `git_dirty=true`; 변경 파일 목록·데이터 checksum·정확한 종료 시각 미제공. 실제 소스 동일성·실행 시간 독립 확인 불가. 기본 `pel4vad_variant=legacy` 값은 PEL 비활성이므로 적용 방식으로 해석하지 않음.
'''
document = f'''# UCF event 기준 모델 진단 / run002

## 목적과 실행 조건

- 실험일: 2026-10-13 (원 실행 폴더 시각 기준); 기록일 2026-10-14
- UCF event 13 clients, FedAvg, seed 88888888, 20 rounds × 10 local epochs, batch 64, LR 0.0001
- MIL 기반 분류 손실만 사용; 손실·client 업데이트 진단 활성화
- 원본 commit {manifest['git_revision']}; PyTorch {manifest['torch']}; dirty 상태 보존
- 기존 원시 파일은 이름만 규칙에 맞게 정리; 내용 보존 및 SHA-256을 analysis.json에 기록

## 결과 요약

{table}
{interpretation}
## 구조화된 결과

```experiment-results
{json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)}
```

## 근거와 다음 확인

- [원시 결과 및 파일 안내]({prefix}README.md), [분석 JSON]({prefix}analysis.json)
- [이전 동일 계열 run001](2026-10-08_ucf_event_baseline_run001.md)
- 다음 실행: `bash scripts/run_pel_diagnostics.sh event_pel`; 동일 seed·예산·분할·집계 유지
- 기록 시각: 2026-10-14; 새 실제 실행 최초 등록, 기존 수치 보존
'''
destination = Path('docs/experiments/2026-10-13_ucf_event_baseline_run002.md')
with destination.open('x', encoding='utf-8') as stream:
    stream.write(document)
(root / 'README.md').write_text('''# UCF event baseline 진단 원시 결과

- 실험일: 2026-10-13; 원 실행 ID `ucf-2026-10-13-17-48-25-320401`
- 저장 규칙: `YYYY-MM-DD_dataset_split_method_aggregation_seedN_roundN_runNNN/`
- 동일 계열 기존 run001 이후 새 실행 run002; 실제 round와 기록 회차 구분
- [공식 결과 기록](../../../docs/experiments/2026-10-13_ucf_event_baseline_run002.md)
- manifest.json: 실행 설정·코드 revision·환경
- round_metrics.jsonl: 20 rounds의 평가 및 client별 평균 손실·steps
- pel_diagnostics.jsonl: 20 rounds의 업데이트, 5개 지정 round의 첫 batch gradient
- summary.json: 최고·최종 평가
- analysis.json: 검증된 round/module 파생 통계 및 네 원시 파일 SHA-256
- 원시 네 파일 내용 수정 없음; 이전 시각 기반 폴더에서 의미 있는 조건 기반 이름으로 이동
- 재생성: `python scripts/analyze_pel_diagnostics.py <이 폴더> --output <새 JSON 경로>`
''', encoding='utf-8')
print('Recorded', destination, 'measurements:', len(measurements))
