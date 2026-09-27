import json, numpy as np, collections

with open('benchmarks/tfm_leaderboard.json') as f:
    data = json.load(f)

by_dataset = {}
for e in data:
    ds = e['dataset']
    m = e['model']
    if ds not in by_dataset:
        by_dataset[ds] = {}
    by_dataset[ds][m] = e

# Find intersection datasets
intersect_datasets = sorted([ds for ds, mods in by_dataset.items()
                              if 'ZS-ISAB' in mods and 'TabICL' in mods and 'TabDPT' in mods])

print(f'Intersection datasets: {len(intersect_datasets)}')

# Recount wins carefully
wins_tabicl = losses_tabicl = ties_tabicl = 0
wins_tabdpt = losses_tabdpt = ties_tabdpt = 0
best_or_tied = 0
thresh = 0.001

for ds in intersect_datasets:
    zsauc = by_dataset[ds]['ZS-ISAB']['metric_val']
    iclauc = by_dataset[ds]['TabICL']['metric_val']
    dptauc = by_dataset[ds]['TabDPT']['metric_val']

    # vs TabICL
    diff1 = zsauc - iclauc
    if diff1 > thresh: wins_tabicl += 1
    elif diff1 < -thresh: losses_tabicl += 1
    else: ties_tabicl += 1

    # vs TabDPT
    diff2 = zsauc - dptauc
    if diff2 > thresh: wins_tabdpt += 1
    elif diff2 < -thresh: losses_tabdpt += 1
    else: ties_tabdpt += 1

    # Best or tied-best across all 3
    best3 = max(zsauc, iclauc, dptauc)
    if zsauc >= best3 - thresh:
        best_or_tied += 1

print(f'\nZS-ISAB vs TabICL: W={wins_tabicl} L={losses_tabicl} T={ties_tabicl}')
print(f'ZS-ISAB vs TabDPT: W={wins_tabdpt} L={losses_tabdpt} T={ties_tabdpt}')
print(f'ZS-ISAB best-or-tied: {best_or_tied}/{len(intersect_datasets)} = {best_or_tied/len(intersect_datasets)*100:.1f}%')

# Size split AUC
size_splits = collections.defaultdict(lambda: {'ZS-ISAB': [], 'TabICL': [], 'TabDPT': []})
for ds in intersect_datasets:
    split = ds.split('/')[0]  # small/medium/large
    for m in ['ZS-ISAB', 'TabICL', 'TabDPT']:
        size_splits[split][m].append(by_dataset[ds][m]['metric_val'])

print('\n=== BY SIZE SPLIT ===')
for split in ['small', 'medium', 'large']:
    s = size_splits[split]
    print(f'{split}: n={len(s["ZS-ISAB"])}  ZS-ISAB={np.mean(s["ZS-ISAB"]):.4f}  TabICL={np.mean(s["TabICL"]):.4f}  TabDPT={np.mean(s["TabDPT"]):.4f}')
    # count ZS-ISAB best-or-tied
    bt = sum(1 for i in range(len(s["ZS-ISAB"]))
             if s["ZS-ISAB"][i] >= max(s["ZS-ISAB"][i], s["TabICL"][i], s["TabDPT"][i]) - thresh)
    print(f'  ZS-ISAB best/tied: {bt}/{len(s["ZS-ISAB"])} = {bt/len(s["ZS-ISAB"])*100:.1f}%')
