import json, statistics
from collections import defaultdict

with open('benchmarks/tfm_leaderboard.json') as f:
    data = json.load(f)

by_dataset = defaultdict(dict)
for entry in data:
    ds = entry['dataset']
    model = entry['model']
    by_dataset[ds][model] = entry

complete = {ds: d for ds, d in by_dataset.items() 
            if 'ZS-ISAB' in d and 'TabICL' in d and 'TabDPT' in d}

def short(ds):
    parts = ds.split('__')
    return parts[1] if len(parts) > 1 else ds

# By split: small, medium, large
for split in ['small', 'medium', 'large']:
    subset = {ds: d for ds, d in complete.items() if ds.startswith(split)}
    if not subset:
        continue
    zs_aucs = [d['ZS-ISAB']['metric_val'] for d in subset.values()]
    icl_aucs = [d['TabICL']['metric_val'] for d in subset.values()]
    dpt_aucs = [d['TabDPT']['metric_val'] for d in subset.values()]
    print(f'=== {split.upper()} ({len(subset)} datasets) ===')
    print(f'  ZS-ISAB  mean AUC: {statistics.mean(zs_aucs):.4f}')
    print(f'  TabICL   mean AUC: {statistics.mean(icl_aucs):.4f}')
    print(f'  TabDPT   mean AUC: {statistics.mean(dpt_aucs):.4f}')
    wins_icl = sum(1 for z,i in zip(zs_aucs,icl_aucs) if z > i + 0.001)
    wins_dpt = sum(1 for z,dp in zip(zs_aucs,dpt_aucs) if z > dp + 0.001)
    best = sum(1 for z,i,dp in zip(zs_aucs,icl_aucs,dpt_aucs) if z >= max(i,dp) - 0.001)
    print(f'  ZS-ISAB beats TabICL: {wins_icl}/{len(subset)}, beats TabDPT: {wins_dpt}/{len(subset)}, best/tied-best: {best}/{len(subset)}')
    print()

print('=== TOP 5 BIGGEST ZS-ISAB WINS vs TabICL ===')
diffs = [(ds, complete[ds]['ZS-ISAB']['metric_val'] - complete[ds]['TabICL']['metric_val']) for ds in complete]
diffs.sort(key=lambda x: -x[1])
for ds, diff in diffs[:5]:
    z = complete[ds]['ZS-ISAB']['metric_val']
    i = complete[ds]['TabICL']['metric_val']
    print(f'  {short(ds)}: ZS={z:.4f}, ICL={i:.4f}, diff={diff:+.4f}')

print()
print('=== TOP 5 BIGGEST ZS-ISAB LOSSES vs TabICL ===')
for ds, diff in diffs[-5:]:
    z = complete[ds]['ZS-ISAB']['metric_val']
    i = complete[ds]['TabICL']['metric_val']
    print(f'  {short(ds)}: ZS={z:.4f}, ICL={i:.4f}, diff={diff:+.4f}')

print()
print('=== TOP 5 BIGGEST ZS-ISAB WINS vs TabDPT ===')
diffs2 = [(ds, complete[ds]['ZS-ISAB']['metric_val'] - complete[ds]['TabDPT']['metric_val']) for ds in complete]
diffs2.sort(key=lambda x: -x[1])
for ds, diff in diffs2[:5]:
    z = complete[ds]['ZS-ISAB']['metric_val']
    dp = complete[ds]['TabDPT']['metric_val']
    print(f'  {short(ds)}: ZS={z:.4f}, DPT={dp:.4f}, diff={diff:+.4f}')

print()
print('=== DATASETS WHERE ONLY ZS-ISAB RAN (baselines OOM or skipped) ===')
zsisab_only = {ds: d for ds, d in by_dataset.items() if 'ZS-ISAB' in d and not ('TabICL' in d and 'TabDPT' in d)}
print(f'Total: {len(zsisab_only)} datasets')
for ds, d in list(zsisab_only.items())[:15]:
    z = d['ZS-ISAB']
    has_icl = 'Y' if 'TabICL' in d else 'N'
    has_dpt = 'Y' if 'TabDPT' in d else 'N'
    print(f'  {short(ds)}: AUC={z["metric_val"]:.4f}, train={z["train_time"]:.1f}s, test={z["test_time"]:.1f}s | ICL={has_icl} DPT={has_dpt}')

print()
print('=== TIMING: WORST TABICL TEST TIMES (top 10 slowest) ===')
icl_times = [(ds, d['TabICL']['test_time']) for ds, d in complete.items()]
icl_times.sort(key=lambda x: -x[1])
for ds, t in icl_times[:10]:
    z = complete[ds]['ZS-ISAB']['test_time']
    ratio = t / z if z > 0 else 0
    print(f'  {short(ds)}: TabICL={t:.1f}s, ZS-ISAB={z:.1f}s, ratio={ratio:.1f}x')
