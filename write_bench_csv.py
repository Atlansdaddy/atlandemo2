#!/usr/bin/env python3
import os
import time
import csv
import statistics as stats
from benchmark_compare import (
    generate_math, run_wave_math, run_sympy_math,
    run_wave_logic, run_z3_logic
)


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def get_pkg_dir(import_name: str) -> str:
    import importlib, os
    m = importlib.import_module(import_name)
    return os.path.dirname(m.__file__)


def main():
    os.makedirs('results', exist_ok=True)
    seeds = [int(time.time()) + i for i in range(5)]

    # Wave vs SymPy (math)
    wave_times = []
    sym_times = []
    wave_scores = []
    sym_scores = []

    # Wave vs Z3 (logic)
    wave_logic_times = []
    z3_logic_times = []
    wave_logic_scores = []
    z3_logic_scores = []

    for s in seeds:
        probs = generate_math(s)
        wt, ws = run_wave_math(probs)
        st, ss = run_sympy_math(probs)
        wave_times.append(wt)
        sym_times.append(st)
        wave_scores.append(ws)
        sym_scores.append(ss)

        from winner.ten_generation_stress_test import RandomQuestionGenerator
        lg = RandomQuestionGenerator(s)
        lprobs = lg.generate_logic_problems()
        wlt, wls = run_wave_logic(lprobs)
        zlt, zls = run_z3_logic(lprobs)
        wave_logic_times.append(wlt)
        z3_logic_times.append(zlt)
        wave_logic_scores.append(wls)
        z3_logic_scores.append(zls)

    # Sizes
    winner_size = dir_size_bytes('winner')
    import sympy, z3
    sympy_size = dir_size_bytes(get_pkg_dir('sympy'))
    z3_size = dir_size_bytes(get_pkg_dir('z3'))

    rows = [
        {
            'system': 'Wave (Math)',
            'avg_time_s_per_set': round(stats.mean(wave_times), 6),
            'p50_time_s': round(stats.median(wave_times), 6),
            'p95_time_s': round(sorted(wave_times)[int(0.95*(len(wave_times)-1))], 6),
            'avg_score': round(stats.mean(wave_scores), 2),
            'size_mb': round(winner_size/1_048_576.0, 2)
        },
        {
            'system': 'SymPy (Math)',
            'avg_time_s_per_set': round(stats.mean(sym_times), 6),
            'p50_time_s': round(stats.median(sym_times), 6),
            'p95_time_s': round(sorted(sym_times)[int(0.95*(len(sym_times)-1))], 6),
            'avg_score': round(stats.mean(sym_scores), 2),
            'size_mb': round(sympy_size/1_048_576.0, 2)
        },
        {
            'system': 'Wave (Logic Pos/Neg/MP/MT/DS/BL)',
            'avg_time_s_per_set': round(stats.mean(wave_logic_times), 6),
            'p50_time_s': round(stats.median(wave_logic_times), 6),
            'p95_time_s': round(sorted(wave_logic_times)[int(0.95*(len(wave_logic_times)-1))], 6),
            'avg_score': round(stats.mean(wave_logic_scores), 2),
            'size_mb': round(winner_size/1_048_576.0, 2)
        },
        {
            'system': 'Z3 (Logic Pos/Neg/MP/MT/DS/BL)',
            'avg_time_s_per_set': round(stats.mean(z3_logic_times), 6),
            'p50_time_s': round(stats.median(z3_logic_times), 6),
            'p95_time_s': round(sorted(z3_logic_times)[int(0.95*(len(z3_logic_times)-1))], 6),
            'avg_score': round(stats.mean(z3_logic_scores), 2),
            'size_mb': round(z3_size/1_048_576.0, 2)
        }
    ]

    out_path = os.path.join('results', 'bench_summary.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()


