#!/usr/bin/env python3
"""
Evaluate LogicExpertModule on LogicBench(Aug) augmentation datasets.
Scans winner/LogicBench(Aug)/**/data_instances.json and reports per-file and overall accuracy and timing.
"""
import os
import argparse
import json
import time
from typing import List, Dict, Tuple


def find_logicbench_files(base_dir: str) -> List[str]:
    targets: List[str] = []
    for root, _dirs, files in os.walk(base_dir):
        for f in files:
            if f == 'data_instances.json':
                targets.append(os.path.join(root, f))
    return sorted(targets)


def evaluate_file(path: str, record_errors: bool = False, errors: List[Dict[str, str]] = None) -> Tuple[int, int, float]:
    from winner.expert_modules.logic_expert import LogicExpertModule
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logic_type = data.get('type', '')
    axiom = data.get('axiom', '')
    samples = data.get('data_samples', [])

    expert = LogicExpertModule()
    correct = 0
    total = 0
    t0 = time.perf_counter()
    for sample in samples:
        context_text = sample.get('context', '')
        qa_pairs = sample.get('qa_pairs', [])
        ctx = {"type": logic_type, "axiom": axiom, "context": context_text}
        for qa in qa_pairs:
            q = qa.get('question', '')
            expected = str(qa.get('answer', '')).strip().lower()
            res = expert.process_query(q, ctx)
            got = str(res.answer).strip().lower()
            if got == expected:
                correct += 1
            total += 1
            if record_errors and errors is not None and got != expected:
                rule_used = res.metadata.get('rule_used', '') if hasattr(res, 'metadata') else ''
                errors.append({
                    'file': path,
                    'type': logic_type,
                    'axiom': axiom,
                    'question': q,
                    'expected': expected,
                    'got': got,
                    'rule_used': rule_used
                })
    t1 = time.perf_counter()
    return correct, total, (t1 - t0)


def main():
    parser = argparse.ArgumentParser(description='Evaluate LogicBench(Aug) with LogicExpertModule')
    parser.add_argument('--base', type=str, default='LogicBench(Aug)', help='Base directory of LogicBench(Aug)')
    parser.add_argument('--subset', type=str, default='all', choices=['all', 'fol', 'prop', 'nm'], help='Limit evaluation to a subset')
    parser.add_argument('--csv-out', type=str, default='', help='Optional path to write per-file CSV summary')
    parser.add_argument('--errors-out', type=str, default='', help='Optional path to write misclassified examples as JSONL')
    args = parser.parse_args()

    base = args.base
    if not os.path.isdir(base):
        print(f"LogicBench(Aug) folder not found at: {base}")
        return

    files = find_logicbench_files(base)
    if args.subset != 'all':
        def keep(fp: str) -> bool:
            if args.subset == 'fol':
                return os.sep + 'first_order_logic' + os.sep in fp
            if args.subset == 'prop':
                return os.sep + 'propositional_logic' + os.sep in fp
            if args.subset == 'nm':
                return os.sep + 'nm_logic' + os.sep in fp
            return True
        files = [fp for fp in files if keep(fp)]
    if not files:
        print("No LogicBench(Aug) data_instances.json files found.")
        return

    overall_correct = 0
    overall_total = 0
    overall_time = 0.0
    rows: List[Tuple[str, int, int, float]] = []
    errors: List[Dict[str, str]] = []

    print("LogicBench(Aug) Evaluation")
    print("=" * 60)
    for fp in files:
        c, n, sec = evaluate_file(fp, record_errors=bool(args.errors_out), errors=errors)
        overall_correct += c
        overall_total += n
        overall_time += sec
        pct = (100.0 * c / n) if n else 0.0
        rel = os.path.relpath(fp)
        print(f"{rel}: {c}/{n} ({pct:.1f}%), {sec:.3f}s")
        rows.append((rel, c, n, sec))

    if overall_total:
        print("-" * 60)
        print(f"TOTAL: {overall_correct}/{overall_total} ({100.0*overall_correct/overall_total:.1f}%), {overall_time:.3f}s")

    # Optional CSV output
    if args.csv_out:
        out_dir = os.path.dirname(args.csv_out)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(args.csv_out, 'w', encoding='utf-8') as f:
            f.write('file,correct,total,pct,time_s\n')
            for rel, c, n, sec in rows:
                pct = (100.0 * c / n) if n else 0.0
                f.write(f"{rel},{c},{n},{pct:.3f},{sec:.6f}\n")
            f.write(f"TOTAL,{overall_correct},{overall_total},{(100.0*overall_correct/overall_total if overall_total else 0):.3f},{overall_time:.6f}\n")

    # Optional errors JSONL output
    if args.errors_out and errors:
        out_dir = os.path.dirname(args.errors_out)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(args.errors_out, 'w', encoding='utf-8') as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()


