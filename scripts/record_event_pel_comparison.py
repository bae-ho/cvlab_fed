"""Record imported event PEL diagnostics and compare saved baseline; no training."""
import hashlib
import json
from pathlib import Path
from statistics import mean

directory = Path('artifacts/experiment-metrics/2026-10-14_ucf_event_fixed_bank_pel_fedavg_seed88888888_round20_run002')
baseline = Path('artifacts/experiment-metrics/2026-10-13_ucf_event_baseline_fedavg_seed88888888_round20_run002')
load = lambda p: json.loads(p.read_text(encoding='utf-8'))
lines = lambda p: [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
analysis, base_analysis = load(directory/'analysis.json'), load(baseline/'analysis.json')
manifest, base_manifest = load(directory/'manifest.json'), load(baseline/'manifest.json')
rounds, base_rounds = lines(directory/'round_metrics.jsonl'), lines(baseline/'round_metrics.jsonl')
diagnostics = lines(directory/'pel_diagnostics.jsonl')
args = manifest['arguments']
for key in ('dataset','split_mode','seed','clients_num','global_rounds','local_epochs','batch_size',
            'learning_rate','aggregation_method','server_optimizer','personalization_type','local_param_policy'):
    assert args[key] == base_manifest['arguments'][key], key
assert args['use_pel4vad_contrastive'] and args['pel4vad_variant']=='fixed_bank' and args['pel_diagnostics']
assert all(r['clients'][i]['steps']==b['clients'][i]['steps'] and
           r['clients'][i]['used_class_counts']==b['clients'][i]['used_class_counts']
           for r,b in zip(rounds,base_rounds) for i in range(13))
old = load(Path('artifacts/experiment-metrics/2026-10-08_ucf_event_scene_reported_rounds.json'))['runs']
prior = next(r for r in old if r['split']=='event' and r['pel'])
assert prior['rounds'] == [[r['evaluation']['roc'],r['evaluation']['ap']] for r in rounds]
prefix='../../'+directory.as_posix()+'/'
measurements=[]


def add(metric,value,round_number,unit='unitless',selection='per_round',conditions=None,
        source='round_metrics.jsonl',provenance='observed',statistic='single'):
    measurements.append(dict(method='Fed-WSVAD + fixed-bank PEL',dataset='ucf',split='event',
        metric=metric,value=value,unit=unit,seed=88888888,round=round_number,statistic=statistic,
        selection=selection,conditions=conditions or {},source=prefix+source+
        (f':{round_number}' if source.endswith('.jsonl') else ''),provenance=provenance))


for r in rounds:
    for key,metric in [('roc','roc_auc'),('ap','ap')]: add(metric,r['evaluation'][key],r['round'],'ratio')
    for i,c in enumerate(r['clients'],1):
        for key,metric in [('clasm','mil_classification_loss'),('pel','pel_loss'),('total','total_loss'),('steps','local_steps')]:
            add(metric,c[key],r['round'],'count' if key=='steps' else 'unitless',conditions={'client':i})
for row in analysis['round_module_metrics']:
    for key in ('retention_ratio','weighted_pair_cosine','aggregate_delta_norm','weighted_delta_norm_sum',
                'aggregate_rounding_residual_norm','negative_pairs','valid_pairs','negative_pair_fraction'):
        add(key,row[key],row['round'],'count' if key.endswith('pairs') else 'unitless',
            conditions={'module':row['module']},source='pel_diagnostics.jsonl',
            provenance='derived' if key in ('negative_pairs','valid_pairs','negative_pair_fraction') else 'observed')
gradient_rows=[]
for d in diagnostics:
    batches=[b for b in d['client_batches'] if b]
    for i,b in enumerate(d['client_batches'],1):
        if not b: continue
        for module,g in b['gradient']['modules'].items():
            for key,metric in [('clasm_norm','mil_gradient_norm'),('pel_norm','pel_gradient_norm'),
                               ('cosine','mil_pel_gradient_cosine'),('weighted_pel_to_clasm_norm','weighted_pel_to_mil_norm')]:
                add(metric,g[key],d['round'],selection='first_batch',conditions={
                    'client':i,'module':module,'epoch':b['epoch'],'batch':b['batch']},source='pel_diagnostics.jsonl')
    if batches:
        values=[b['gradient']['modules']['all_visual'] for b in batches]
        gradient_rows.append({'round':d['round'],'n':len(values),
            'negative_cosines':sum(g['cosine']<0 for g in values),
            'mean_cosine':mean(g['cosine'] for g in values),
            'mean_weighted_norm_ratio':mean(g['weighted_pel_to_clasm_norm'] for g in values)})
for selection,n in [('best_roc_round',analysis['best_roc_round']),('final',20)]:
    for key,metric in [('roc','roc_auc'),('ap','ap')]: add(metric,rounds[n-1]['evaluation'][key],n,'ratio',selection)
add('ap',rounds[analysis['best_ap_round']-1]['evaluation']['ap'],analysis['best_ap_round'],'ratio','independent_best_ap')
comparison=[]
for row in analysis['round_module_metrics']:
    base=next(b for b in base_analysis['round_module_metrics'] if b['round']==row['round'] and b['module']==row['module'])
    comparison.append({'round':row['round'],'module':row['module'],
        'baseline_retention':base['retention_ratio'],'pel_retention':row['retention_ratio'],
        'retention_delta':row['retention_ratio']-base['retention_ratio'] if row['retention_ratio'] is not None else None,
        'baseline_pair_cosine':base['weighted_pair_cosine'],'pel_pair_cosine':row['weighted_pair_cosine'],
        'baseline_negative_pairs':base['negative_pairs'],'pel_negative_pairs':row['negative_pairs']})
comparison_json={'baseline':baseline.as_posix(),'pel':directory.as_posix(),
    'comparison':comparison,'gradient_round_summaries':gradient_rows,
    'configuration_matched':True,'client_steps_and_class_counts_matched':True,
    'source_identity_verified':False,'data_checksum_verified':False}
with (directory/'comparison.json').open('x',encoding='utf-8') as f:
    json.dump(comparison_json,f,ensure_ascii=False,indent=2,allow_nan=False)
shared=[r for r in comparison if r['module']=='all_shared']
table='| Round | 기준 보존 비율 | PEL 보존 비율 | 차이 | 기준 음의 pair | PEL 음의 pair |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
for r in shared:
    table+=f"| {r['round']} | {r['baseline_retention']:.6f} | {r['pel_retention']:.6f} | {r['retention_delta']:+.6f} | {r['baseline_negative_pairs']} | {r['pel_negative_pairs']} |\n"
gtable='| Round | 표본 수 | 음의 gradient cosine | 평균 cosine | 평균 계수 반영 norm 비율 |\n| --- | ---: | ---: | ---: | ---: |\n'
for r in gradient_rows:
    gtable+=f"| {r['round']} | {r['n']} | {r['negative_cosines']} | {r['mean_cosine']:.6f} | {r['mean_weighted_norm_ratio']:.6f} |\n"
conclusion='''- 관찰: 전체 시각 파라미터와 Temporal 모듈에서 65개 gradient 표본 모두 양의 cosine. 관측 범위에서 두 손실의 반대 방향 경쟁 가설 미지지. 미측정 batch·후속 epoch까지 충돌 부재 일반화 불가.
- 관찰: 집계 보존 비율의 기준 대비 효과는 round별 혼재. 초반 개선과 후반 하락이 함께 존재; PEL이 항상 상쇄를 줄이거나 늘린다는 결론 불가.
- 성능: 최고 ROC 기준 85.2737% → PEL 86.1961% (+0.9224%p); 최종 ROC 85.2338% → 83.9980% (-1.2358%p). 독립 최고 AP +1.0236%p, 최종 AP -1.3550%p.
- 해석: 손실의 국소 gradient 정렬과 client 간 최종 업데이트 정렬은 다른 측정 대상. 여러 local steps·서로 다른 데이터가 누적된 업데이트를 첫 batch gradient로 대체 불가.
- 제한: 단일 seed, 첫 batch 표본, test 최고 선택, 서로 다른 commit과 두 실행의 dirty 상태. 데이터 checksum·변경 파일 목록 미제공. 성능 저하의 인과 원인 미확정.
- 재현: 이전 PEL 실행의 20개 round ROC/AP와 정확 일치. 원시 결과·회차는 새 실행으로 보존.
'''
record=dict(schema_version=1,date='2026-10-14',date_kind='experiment',experiment_type='ucf_event_fixed_bank_pel',
    attempt=2,track='pel',status='completed',conditions={
    **{k:args[k] for k in ('aggregation_method','global_rounds','local_epochs','batch_size','learning_rate',
                         'clients_num','server_optimizer','personalization_type','local_param_policy',
                         'pel4vad_weight','pel4vad_temperature','pel4vad_variant')},
    'pel_enabled':True,'pel_diagnostics':True,'git_revision':manifest['git_revision'],
    'git_dirty':manifest['git_dirty'],'torch':manifest['torch'],
    'baseline_record':'2026-10-13_ucf_event_baseline_run002.md',
    'date_evidence':'user-exported run folder date; original timestamp/end time unavailable',
    'checkpoint_selection':'best round on test set (existing protocol)'},measurements=measurements)
document=f'''# UCF-Crime 이벤트별 클라이언트 분할에서 PEL 의미 정렬 추가 진단 / run002

- 실험일: 2026-10-14 (사용자가 실행 폴더 날짜로 저장한 결과); 기록일 2026-10-14
- 목적: MIL 기반 분류 손실에 PEL을 추가할 때 로컬 손실 경쟁과 client 업데이트 상쇄 변화 확인
- 조건: UCF-Crime, event 13 clients, FedAvg, seed 88888888, 20 rounds × 10 local epochs, batch 64, LR 0.0001
- PEL: 클래스 이름 14개 고정 기준, weight 1.0, temperature 0.09; ConceptNet 기반 원본 전체 재현과 구분
- 기준: [미적용 진단 run002](2026-10-13_ucf_event_baseline_run002.md); 공통 설정·client별 steps·클래스 사용 횟수 일치 확인
- [사전 설계](../plans/active/2026-10-14-ucf-event-pel-diagnostic-design.md)

## 집계 업데이트 비교

{table}
## 동일 forward의 시각 gradient

아래 평균은 해당 round의 13개 client 표본 산술평균이며 seed 평균과 구분.

{gtable}
## 관찰과 해석

{conclusion}
## 구조화된 측정값

```experiment-results
{json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)}
```

## 근거와 확인 기록

- [원시 결과 안내]({prefix}README.md), [파생 비교]({prefix}comparison.json)
- 원시 네 파일 SHA-256을 analysis.json에 보존; summary와 전체 round 평가 대조 통과
- 장면별 클라이언트 분할 두 조건은 미완료; 1단계 전체 완료로 처리하지 않음
'''
destination=Path('docs/experiments/2026-10-14_ucf_event_fixed_bank_pel_run002.md')
with destination.open('x',encoding='utf-8') as f: f.write(document)
(directory/'README.md').write_text('''# UCF-Crime 이벤트별 클라이언트 분할의 PEL 의미 정렬 진단 결과

- 실험일: 2026-10-14; seed 88888888, FedAvg, 20 rounds, 기록 회차 run002
- [실험 기록](../../../docs/experiments/2026-10-14_ucf_event_fixed_bank_pel_run002.md)
- manifest.json: 설정·환경·revision·텍스트 기준 및 checksum
- round_metrics.jsonl: round 평가와 client별 손실·학습량
- pel_diagnostics.jsonl: client 업데이트 및 지정 batch gradient
- summary.json: 최고·최종 성능과 선택 정책
- analysis.json: 원시 SHA-256 및 모듈별 통계
- comparison.json: 기준 모델과의 round/module 대응 비교 및 gradient 표본 평균
- 원시 네 파일 내용 보존; 데이터·모델 가중치 제외
''',encoding='utf-8')
print('Recorded measurements:',len(measurements))
print('Retention increase rounds:',sum(r['retention_delta']>0 for r in shared),'of',len(shared))
print('Mean retention first5 baseline/PEL:',mean(r['baseline_retention'] for r in shared[:5]),mean(r['pel_retention'] for r in shared[:5]))
print('Mean retention last5 baseline/PEL:',mean(r['baseline_retention'] for r in shared[-5:]),mean(r['pel_retention'] for r in shared[-5:]))
