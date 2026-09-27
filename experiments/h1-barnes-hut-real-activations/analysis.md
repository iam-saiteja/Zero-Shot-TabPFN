# H1 analysis (CONFIRMATORY per locked protocol; details marked EXPLORATORY)

Data: results/pilot_results.csv (10 OpenML datasets x 2 seeds, TabPFN v1 bundled checkpoint, N_train=3000, N_test=500).
Sanity: vanilla deterministic (0.0 diff); patched-exact vs vanilla max diff 1.6e-5; all outputs finite.

| mode (M=64) | median fidelity err (mean abs dprob vs exact) | mean acc delta vs exact | rows touched/query |
|---|---|---|---|
| isab_random | 0.091 | -4.94pp | 64 |
| isab_kmeans | 0.069 | -3.59pp | 64 |
| bh (t=4) | 0.029 | -0.33pp | 250 |
| bh (t=8) | 0.021 | +0.14pp | 435 |
| exact | 0 | 0 | ~2970 |

**Gate: PASSED** (7/10 datasets at <=0.5x isab_kmeans error, need 7; mean acc delta -0.33pp, need within 0.5pp; median ratio 0.43, kill threshold 0.67).

Reads:
- H1a (k-means beats random): supported but modest on real data (0.069 vs 0.091 median; 3.6 vs 4.9pp accuracy loss). Sim predicted 2-4x; real is ~1.3x.
- H1b (near-field expansion beats both): supported. Real gain over k-means is ~2.3x median, smaller than the 5-15x the synthetic sim suggested (EXPLORATORY: real embeddings cluster less cleanly).
- Passed narrowly: exactly 7/10 (adult 0.73, electricity 0.76, jm1 0.95 did not meet the bar). Where embeddings are less clustered, near-field helps less.
- Plain anchors lose 3-5pp accuracy: ZS-ISAB as shipped costs real accuracy vs exact at N=3000.
- Cost: BH touches ~12x fewer rows than exact, but wall-clock here is SLOWER than exact (2.3s vs 1.8s) because this implementation builds dense masks. Speed is NOT demonstrated; needs a gather/sparse kernel.
- Caveats: reference "exact" is TabPFN v1 at N=3000 (outside its ~1000-row design range); fidelity to exact is not accuracy vs labels; 2 seeds.
