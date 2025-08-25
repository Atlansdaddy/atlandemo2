#!/usr/bin/env python3
"""
Ultimate 20K Test - 10x Ten-Generation Tests (20,000 total questions)
"""
import sys
import os
import time
import argparse
sys.path.insert(0, os.path.dirname(__file__))

from ten_generation_stress_test import run_ten_generation_stress_test

def run_20k_ultimate_test(quiet: bool = False, jsonl_dir: str | None = None,
                          sample_metrics: bool = False, sample_interval: float = 0.1) -> None:
    print("🚀 ULTIMATE 20,000 QUESTION TEST")
    print("Running 10x Ten-Generation Tests")
    print("=" * 50)
    
    start_time = time.time()
    all_results = []
    
    for run in range(10):
        print(f"\n🔄 RUN {run+1}/10")
        jsonl_path = None
        if jsonl_dir:
            try:
                os.makedirs(jsonl_dir, exist_ok=True)
            except Exception:
                pass
            jsonl_path = os.path.join(jsonl_dir, f"ten_gen_run_{run+1}.jsonl")
        result = run_ten_generation_stress_test(
            quiet=quiet,
            jsonl_path=jsonl_path,
            sample_metrics=sample_metrics,
            sample_interval=sample_interval,
        )
        all_results.append(result['summary_stats'])
        print(f"Run {run+1}: {result['summary_stats']['total_avg']:.1f}/200")
    
    # Consolidate results
    total_time = time.time() - start_time
    
    math_scores = [r['math_avg'] for r in all_results]
    logic_scores = [r['logic_avg'] for r in all_results] 
    total_scores = [r['total_avg'] for r in all_results]
    
    print(f"\n📊 20,000 QUESTION ULTIMATE REPORT")
    print("=" * 50)
    print(f"Time: {total_time:.1f}s | Speed: {20000/total_time:.0f} questions/sec")
    print(f"Math: {sum(math_scores)/10:.1f}/100 (Range: {min(math_scores):.0f}-{max(math_scores):.0f})")
    print(f"Logic: {sum(logic_scores)/10:.1f}/100 (Range: {min(logic_scores):.0f}-{max(logic_scores):.0f})")  
    print(f"Total: {sum(total_scores)/10:.1f}/200 ({100*sum(total_scores)/(10*200):.1f}%)")
    print(f"Consistency: {100-((max(total_scores)-min(total_scores))/2):.1f}%")
    print(f"Perfect Runs: {sum(1 for s in total_scores if s >= 199.9)}/10")
    
    if all(s >= 199.9 for s in total_scores):
        print("🏆 PERFECT 100% ACROSS ALL 20,000 QUESTIONS!")
    elif sum(total_scores)/10 >= 199.5:
        print("🥇 EXCEPTIONAL: >99.75% average!")
    elif sum(total_scores)/10 >= 199.0:
        print("🥈 OUTSTANDING: >99.5% average!")
    else:
        print("📊 Strong performance across 20K questions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultimate 20K Test")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-question prints inside sub-tests")
    parser.add_argument("--jsonl-dir", type=str, default=None, help="Directory to write per-run JSONL timing logs")
    parser.add_argument("--sample-metrics", action="store_true", help="Enable resource sampling in sub-tests")
    parser.add_argument("--sample-interval", type=float, default=0.1, help="Resource sample interval seconds")
    args = parser.parse_args()

    run_20k_ultimate_test(
        quiet=args.quiet,
        jsonl_dir=args.jsonl_dir,
        sample_metrics=args.sample_metrics,
        sample_interval=args.sample_interval,
    )