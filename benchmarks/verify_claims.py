import json, numpy as np

with open('benchmarks/tfm_leaderboard.json') as f:
    data = json.load(f)

by_dataset = {}
for e in data:
    ds = e['dataset']
    m = e['model']
    if ds not in by_dataset: by_dataset[ds] = {}
    by_dataset[ds][m] = e

print('=== SPECIFIC DATASET CLAIMS CHECK ===')
checks = [
    ('large/openml__ldpa__9974', 'ZS-ISAB', 0.9857),
    ('large/openml__ldpa__9974', 'TabICL', 0.5098),
    ('large/openml__skin-segmentation__9965', 'ZS-ISAB', 1.000),
    ('large/openml__walking-activity__9945', 'ZS-ISAB', 0.9754),
    ('large/openml__walking-activity__9945', 'TabICL', 0.9733),
]

for ds, model, expected in checks:
    if ds in by_dataset and model in by_dataset[ds]:
        e = by_dataset[ds][model]
        val = e['metric_val']
        match = abs(val - expected) < 0.002
        status = 'OK' if match else 'MISMATCH'
        print(f'  [{status}] {ds.split("/")[-1]} | {model}: actual={val:.4f} expected={expected:.4f}')
    else:
        print(f'  [MISSING] {ds} | {model}')

print()
print('=== LATENCY CLAIMS ===')
for ds_key, model in [('large/openml__ldpa__9974', 'TabICL'), ('large/openml__ldpa__9974', 'ZS-ISAB'),
                       ('large/openml__skin-segmentation__9965', 'TabICL'), ('large/openml__skin-segmentation__9965', 'ZS-ISAB'),
                       ('large/openml__walking-activity__9945', 'ZS-ISAB'), ('large/openml__walking-activity__9945', 'TabICL')]:
    e = by_dataset.get(ds_key, {}).get(model, {})
    tt = e.get('test_time', 'N/A')
    if isinstance(tt, float):
        print(f'  {ds_key.split("__")[1]} | {model}: test_time={tt:.1f}s')
    else:
        print(f'  {ds_key.split("__")[1]} | {model}: MISSING')

print()
# Win/loss check on 142 intersection datasets
models_to_check = ['ZS-ISAB', 'TabICL', 'TabDPT']
intersect_datasets = set()
for ds, mods in by_dataset.items():
    if 'ZS-ISAB' in mods and 'TabICL' in mods and 'TabDPT' in mods:
        intersect_datasets.add(ds)
print(f'=== INTERSECTION DATASETS (all 3 models): {len(intersect_datasets)} (paper claims 142) ===')

wins_tabicl = losses_tabicl = ties_tabicl = 0
wins_tabdpt = losses_tabdpt = ties_tabdpt = 0
for ds in intersect_datasets:
    zsauc = by_dataset[ds]['ZS-ISAB']['metric_val']
    tabauc = by_dataset[ds]['TabICL']['metric_val']
    dptauc = by_dataset[ds]['TabDPT']['metric_val']
    thresh = 0.001
    diff1 = zsauc - tabauc
    if diff1 > thresh: wins_tabicl += 1
    elif diff1 < -thresh: losses_tabicl += 1
    else: ties_tabicl += 1
    diff2 = zsauc - dptauc
    if diff2 > thresh: wins_tabdpt += 1
    elif diff2 < -thresh: losses_tabdpt += 1
    else: ties_tabdpt += 1

print(f'ZS-ISAB vs TabICL: W={wins_tabicl} L={losses_tabicl} T={ties_tabicl}  (paper: W=42 L=36 T=64)')
print(f'ZS-ISAB vs TabDPT: W={wins_tabdpt} L={losses_tabdpt} T={ties_tabdpt}  (paper: W=61 L=24 T=57)')
