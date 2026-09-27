import json, numpy as np
with open('benchmarks/tfm_leaderboard.json') as f:
    data = json.load(f)

reg_entries = [e for e in data if e.get('metric_name') == 'R2']
print('=== REGRESSION ENTRIES ===')
for e in reg_entries:
    print(f"  model={e['model']} dataset={e['dataset']} R2={e['metric_val']:.4f} train={e['train_time']:.1f}s test={e['test_time']:.1f}s")

# Also find dataset name and size for aloi
aloi = [e for e in data if 'aloi' in e.get('dataset','')]
print('\n=== ALOI entries ===')
for e in aloi:
    print(e)
