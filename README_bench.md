#!/usr/bin/env markdown

## Wave Reasoning Benchmarks (Winner vs SymPy vs Z3)

This repo ships `winner/` (the on-device wave-based symbolic engine) and lightweight benchmarks.

### Quick Results (on this machine)
- 20k mixed math+logic: 0.7s total (100% accuracy), ~29.2k QPS
- Math (Wave vs SymPy): Wave ~5× faster, both 100%
- Logic MP/MT/DS/Pos/Neg/BL (Wave vs Z3): Wave ~10–13× faster, both 100%
- Footprint: `winner/` core ~114 KB; SymPy ~65 MB; Z3 (Python pkg) ~20 MB

### Reproduce
1) Install Python 3.13+ and run:

```
python -X utf8 -m pip install sympy z3-solver
```

2) Run the 20k test (10× ten-generation):

```
python -X utf8 winner\ultimate_20k_test.py --quiet
```

3) Run head-to-head math/logic benchmarks and write CSV:

```
python -X utf8 benchmark_compare.py
python -X utf8 write_bench_csv.py
```

Outputs:
- `results/bench_summary.csv` (speed/accuracy/size summary)

### Notes
- `winner/` is self-contained; no lookup tables; pure Python; offline.
- Benchmarks use randomly generated problem sets with recorded seeds.
- Comparators (SymPy/Z3) operate on equivalent formal encodings; Wave processes NL-ish prompts.


