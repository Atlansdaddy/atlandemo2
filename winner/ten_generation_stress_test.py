#!/usr/bin/env python3
"""
Ten Generation Stress Test - Random Seeded 200-Question Runs
Tests system consistency across 10 different random generations of problems
Memory optimized for mobile environments
"""

import sys
import os
import random
import gc
import json
import time
import argparse
import threading
import tracemalloc
import hashlib
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

from expert_modules.math_expert import MathExpertModule
from expert_modules.logic_expert import LogicExpertModule


class _MetricsLogger:
    """Minimal JSONL logger for per-event metrics."""

    def __init__(self, jsonl_path: str | None):
        self.jsonl_path = jsonl_path
        self._lock = threading.Lock()
        self._enabled = bool(jsonl_path)

    def log(self, record: dict):
        if not self._enabled:
            return
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


class _ResourceSampler:
    """Optional resource sampler (best-effort). Tries psutil/pynvml; falls back to tracemalloc stats."""

    def __init__(self, interval_sec: float = 0.1):
        self.interval = max(0.05, interval_sec)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict] = []
        self._psutil = None
        self._proc = None
        self._gpu = None
        self._pynvml = None

        try:
            import psutil  # type: ignore
            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
            # prime CPU percent
            try:
                self._proc.cpu_percent(interval=None)
            except Exception:
                pass
        except Exception:
            self._psutil = None
            self._proc = None

        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._pynvml = None
            self._gpu = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            ts = time.perf_counter()
            rec: dict = {
                "ts": ts,
            }
            # psutil metrics
            if self._proc is not None:
                try:
                    cpu = self._proc.cpu_percent(interval=None)
                    mem = self._proc.memory_info()
                    rec.update({
                        "cpu_proc_pct": cpu,
                        "rss": getattr(mem, "rss", None),
                        "vms": getattr(mem, "vms", None),
                        "num_threads": self._proc.num_threads(),
                    })
                except Exception:
                    pass
            # GPU metrics (NVIDIA)
            if self._pynvml is not None and self._gpu is not None:
                try:
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._gpu)
                    power = None
                    try:
                        power = self._pynvml.nvmlDeviceGetPowerUsage(self._gpu)  # mW
                    except Exception:
                        pass
                    rec.update({
                        "gpu_util_pct": getattr(util, "gpu", None),
                        "gpu_mem_util_pct": getattr(util, "memory", None),
                        "gpu_power_mw": power,
                    })
                except Exception:
                    pass
            # tracemalloc snapshot
            try:
                current, peak = tracemalloc.get_traced_memory()
                rec.update({"py_current_bytes": current, "py_peak_bytes": peak})
            except Exception:
                pass

            self.samples.append(rec)
            self._stop.wait(self.interval)


def _sha256_files(paths: list[str]) -> str:
    h = hashlib.sha256()
    for p in paths:
        try:
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    h.update(chunk)
        except Exception:
            continue
    return h.hexdigest()

class RandomQuestionGenerator:
    """Generates random math and logic questions with seeding"""
    
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.seed = seed
    
    def generate_math_problems(self) -> dict:
        """Generate 100 random math problems"""
        problems = {}
        
        # Order of operations (25)
        problems['order'] = []
        for _ in range(25):
            a, b, c = self.rng.randint(1, 50), self.rng.randint(2, 20), self.rng.randint(2, 15)
            expr = f"{a} + {b} * {c}"
            expected = a + b * c
            problems['order'].append((expr, expected))
        
        # Percentages (25) 
        problems['percent'] = []
        for _ in range(25):
            percent = self.rng.randint(10, 80)
            number = self.rng.randint(100, 900)
            expr = f"{percent}% of {number}"
            expected = percent * number / 100
            problems['percent'].append((expr, expected))
        
        # Basic arithmetic (25)
        problems['arithmetic'] = []
        for _ in range(25):
            op = self.rng.choice(['+', '-', '*', '/'])
            if op == '+':
                a, b = self.rng.randint(100, 999), self.rng.randint(100, 999)
                expr = f"{a} + {b}"
                expected = a + b
            elif op == '-':
                a, b = self.rng.randint(500, 999), self.rng.randint(100, 400)
                expr = f"{a} - {b}"
                expected = a - b
            elif op == '*':
                a, b = self.rng.randint(20, 80), self.rng.randint(20, 50)
                expr = f"{a} * {b}"
                expected = a * b
            else:  # division
                b = self.rng.randint(20, 60)
                a = b * self.rng.randint(30, 90)  # Ensure clean division
                expr = f"{a} / {b}"
                expected = a / b
            problems['arithmetic'].append((expr, expected))
        
        # Perfect squares (15)
        problems['squares'] = []
        squares = [i*i for i in range(1, 21)]
        for _ in range(15):
            square = self.rng.choice(squares)
            root = int(square ** 0.5)
            expr = f"x^2 = {square}"
            expected = f"±{root}"
            problems['squares'].append((expr, expected))
        
        # Statistics - means (10)
        problems['stats'] = []
        for _ in range(10):
            count = self.rng.randint(4, 8)
            numbers = [self.rng.randint(10, 100) for _ in range(count)]
            expr = f"mean of {', '.join(map(str, numbers))}"
            expected = sum(numbers) / len(numbers)
            problems['stats'].append((expr, expected))
        
        return problems
    
    def generate_logic_problems(self) -> dict:
        """Generate 100 random logic problems"""
        problems = {}
        
        # Universal positive syllogisms (15)
        problems['pos_syl'] = []
        subjects = ['electrons', 'photons', 'atoms', 'molecules', 'crystals', 'metals', 'gases', 'liquids']
        properties = ['charged', 'energy-bearing', 'massive', 'composite', 'ordered', 'conductive', 'expandable', 'fluid']
        instances = ['proton', 'light', 'carbon', 'water', 'diamond', 'copper', 'helium', 'mercury']
        
        for _ in range(15):
            subj = self.rng.choice(subjects)
            prop = self.rng.choice(properties) 
            inst = self.rng.choice(instances)
            query = f"All {subj} are {prop}. {inst} is a {subj[:-1]}. Is {inst} {prop}?"
            premises = [f"All {subj} are {prop}", f"{inst} is a {subj[:-1]}"]
            problems['pos_syl'].append((query, premises, "yes"))
        
        # Universal negative syllogisms (15)
        problems['neg_syl'] = []
        for _ in range(15):
            subj = self.rng.choice(subjects)
            neg_prop = f"not {self.rng.choice(properties)}"
            inst = self.rng.choice(instances)
            query = f"All {subj} are {neg_prop}. {inst} is a {subj[:-1]}. Is {inst} {neg_prop.split()[1]}?"
            premises = [f"All {subj} are {neg_prop}", f"{inst} is a {subj[:-1]}"]
            problems['neg_syl'].append((query, premises, "no"))
        
        # Modus ponens (15)
        problems['mp'] = []
        causes = ['pressure builds', 'voltage rises', 'temperature drops', 'force applies', 'light bends']
        effects = ['steam escapes', 'current increases', 'water freezes', 'object accelerates', 'image distorts']
        
        for _ in range(15):
            cause = self.rng.choice(causes)
            effect = self.rng.choice(effects)
            query = f"If {cause} then {effect}. {cause.capitalize()}. Does {effect.split()[0]} {effect.split()[1]}?"
            premises = [f"If {cause} then {effect}", cause.capitalize()]
            problems['mp'].append((query, premises, "yes"))
        
        # Modus tollens (15)
        problems['mt'] = []
        for _ in range(15):
            cause = self.rng.choice(causes)
            effect = self.rng.choice(effects)
            neg_effect = f"{effect.split()[0]} does not {effect.split()[1]}"
            neg_cause = f"{cause.split()[0]} does not {' '.join(cause.split()[1:])}"
            query = f"If {cause} then {effect}. {neg_effect.capitalize()}. Does {neg_cause}?"
            premises = [f"If {cause} then {effect}", neg_effect.capitalize()]
            problems['mt'].append((query, premises, "yes"))
        
        # Disjunctive syllogisms (10)
        problems['ds'] = []
        options_a = ['wave is transverse', 'bond is ionic', 'force is attractive', 'reaction is endothermic', 'state is solid']
        options_b = ['longitudinal', 'covalent', 'repulsive', 'exothermic', 'liquid']
        
        for _ in range(10):
            i = self.rng.randint(0, len(options_a)-1)
            opt_a, opt_b = options_a[i], options_b[i]
            query = f"Either the {opt_a} or {opt_b}. The {opt_a.split()[-2]} is not {opt_a.split()[-1]}. Is the {opt_a.split()[-2]} {opt_b}?"
            premises = [f"Either the {opt_a} or {opt_b}", f"The {opt_a.split()[-2]} is not {opt_a.split()[-1]}"]
            problems['ds'].append((query, premises, "yes"))
        
        # Hypothetical syllogisms (10)
        problems['hs'] = []
        for _ in range(10):
            a = self.rng.choice(causes)
            b = self.rng.choice(effects)
            c = self.rng.choice(['power increases', 'efficiency improves', 'output grows'])
            query = f"If {a} then {b}. If {b} then {c}. Does {a} lead to {c}?"
            premises = [f"If {a} then {b}", f"If {b} then {c}"]
            problems['hs'].append((query, premises, "yes"))
        
        # Biconditional logic (10)
        problems['bl'] = []
        conditions = ['equilibrium exists', 'resonance occurs', 'interference happens']
        requirements = ['forces balance', 'frequencies match', 'waves overlap']
        
        for _ in range(10):
            cond = self.rng.choice(conditions)
            req = self.rng.choice(requirements)
            query = f"{cond.capitalize()} if and only if {req}. {cond.capitalize()}. Do {req}?"
            premises = [f"{cond.capitalize()} if and only if {req}", cond.capitalize()]
            problems['bl'].append((query, premises, "yes"))
        
        # Existential quantification (5)
        problems['eq'] = []
        for _ in range(5):
            category = self.rng.choice(['metals', 'gases', 'crystals'])
            prop = self.rng.choice(['magnetic', 'noble', 'piezoelectric'])
            instance = self.rng.choice(['iron', 'helium', 'quartz'])
            query = f"Some {category} are {prop}. {instance} is a {category[:-1]}. Might {instance} be {prop}?"
            premises = [f"Some {category} are {prop}", f"{instance} is a {category[:-1]}"]
            problems['eq'].append((query, premises, "yes"))
        
        # Contradiction detection (5)
        problems['cd'] = []
        statements = ['entropy always increases', 'momentum is conserved', 'charge is quantized']
        for _ in range(5):
            stmt = self.rng.choice(statements)
            neg_stmt = f"{stmt.split()[0]} {' '.join(stmt.split()[1:-1])} not {stmt.split()[-1]}"
            query = f"{stmt.capitalize()}. {neg_stmt.capitalize()}. Does {stmt}?"
            premises = [stmt.capitalize(), neg_stmt.capitalize()]
            problems['cd'].append((query, premises, "contradiction"))
        
        return problems

def run_generation(generation: int, seed: int, quiet: bool = False, logger: _MetricsLogger | None = None,
                   per_item_timing: bool = True) -> dict:
    """Run a single generation with the given seed"""
    if not quiet:
        print(f"\n🔄 GENERATION {generation} (Seed: {seed})")
        print("=" * 50)
    
    # Initialize experts
    math_expert = MathExpertModule()
    logic_expert = LogicExpertModule()
    
    # Generate random problems
    generator = RandomQuestionGenerator(seed)
    math_problems = generator.generate_math_problems()
    logic_problems = generator.generate_logic_problems()
    
    results = {
        'generation': generation,
        'seed': seed,
        'math_scores': {},
        'logic_scores': {},
        'total_math': 0,
        'total_logic': 0
    }
    
    # Test math problems
    if not quiet:
        print("\n🔢 MATHEMATICS (100 questions)")
    
    # Order of operations
    correct = 0
    for idx, (expr, expected) in enumerate(math_problems['order']):
        if per_item_timing:
            t0 = time.perf_counter()
        result = math_expert.process_query(f"What is {expr}?", {})
        if per_item_timing:
            t1 = time.perf_counter()
        try:
            actual = float(result.answer) if result.answer.replace('.','').replace('-','').isdigit() else 0
            if abs(actual - expected) < 0.01:
                correct += 1
        except:
            pass
        if per_item_timing and logger:
            logger.log({
                "phase": "math_order",
                "generation": generation,
                "index": idx,
                "seed": seed,
                "reasoning_time_s": result.processing_time if hasattr(result, 'processing_time') else None,
                "total_time_s": t1 - t0,
            })
    results['math_scores']['order'] = correct
    if not quiet:
        print(f"  Order of Operations: {correct}/25")
    
    # Percentages  
    correct = 0
    for idx, (expr, expected) in enumerate(math_problems['percent']):
        if per_item_timing:
            t0 = time.perf_counter()
        result = math_expert.process_query(f"What is {expr}?", {})
        if per_item_timing:
            t1 = time.perf_counter()
        try:
            actual = float(result.answer) if result.answer.replace('.','').replace('-','').isdigit() else 0
            if abs(actual - expected) < 0.1:
                correct += 1
        except:
            pass
        if per_item_timing and logger:
            logger.log({
                "phase": "math_percent",
                "generation": generation,
                "index": idx,
                "seed": seed,
                "reasoning_time_s": result.processing_time if hasattr(result, 'processing_time') else None,
                "total_time_s": t1 - t0,
            })
    results['math_scores']['percent'] = correct
    if not quiet:
        print(f"  Percentages: {correct}/25")
    
    # Basic arithmetic
    correct = 0
    for idx, (expr, expected) in enumerate(math_problems['arithmetic']):
        if per_item_timing:
            t0 = time.perf_counter()
        result = math_expert.process_query(f"What is {expr}?", {})
        if per_item_timing:
            t1 = time.perf_counter()
        try:
            actual = float(result.answer) if result.answer.replace('.','').replace('-','').isdigit() else 0
            if abs(actual - expected) < 0.1:
                correct += 1
        except:
            pass
        if per_item_timing and logger:
            logger.log({
                "phase": "math_arithmetic",
                "generation": generation,
                "index": idx,
                "seed": seed,
                "reasoning_time_s": result.processing_time if hasattr(result, 'processing_time') else None,
                "total_time_s": t1 - t0,
            })
    results['math_scores']['arithmetic'] = correct
    if not quiet:
        print(f"  Basic Arithmetic: {correct}/25")
    
    # Perfect squares
    correct = 0
    for idx, (expr, expected) in enumerate(math_problems['squares']):
        if per_item_timing:
            t0 = time.perf_counter()
        result = math_expert.process_query(f"Solve {expr}", {})
        if per_item_timing:
            t1 = time.perf_counter()
        if result.answer == expected:
            correct += 1
        if per_item_timing and logger:
            logger.log({
                "phase": "math_squares",
                "generation": generation,
                "index": idx,
                "seed": seed,
                "reasoning_time_s": result.processing_time if hasattr(result, 'processing_time') else None,
                "total_time_s": t1 - t0,
            })
    results['math_scores']['squares'] = correct
    if not quiet:
        print(f"  Perfect Squares: {correct}/15")
    
    # Statistics
    correct = 0
    for idx, (expr, expected) in enumerate(math_problems['stats']):
        if per_item_timing:
            t0 = time.perf_counter()
        result = math_expert.process_query(f"What is the {expr}?", {})
        if per_item_timing:
            t1 = time.perf_counter()
        try:
            actual = float(result.answer) if result.answer.replace('.','').replace('-','').isdigit() else 0
            if abs(actual - expected) < 0.1:
                correct += 1
        except:
            pass
        if per_item_timing and logger:
            logger.log({
                "phase": "math_stats",
                "generation": generation,
                "index": idx,
                "seed": seed,
                "reasoning_time_s": result.processing_time if hasattr(result, 'processing_time') else None,
                "total_time_s": t1 - t0,
            })
    results['math_scores']['stats'] = correct
    if not quiet:
        print(f"  Statistics: {correct}/10")
    
    results['total_math'] = sum(results['math_scores'].values())
    if not quiet:
        print(f"  MATH TOTAL: {results['total_math']}/100")
    
    # Force garbage collection
    gc.collect()
    
    # Test logic problems
    if not quiet:
        print(f"\n🧠 LOGIC (100 questions)")
    
    # Universal positive syllogisms
    correct = 0
    for idx, (query, premises, expected) in enumerate(logic_problems['pos_syl']):
        if per_item_timing:
            t0 = time.perf_counter()
        result = logic_expert.process_query(query, {"premises": premises})
        if per_item_timing:
            t1 = time.perf_counter()
        if result.answer.lower() == expected:
            correct += 1
        if per_item_timing and logger:
            logger.log({
                "phase": "logic_pos_syl",
                "generation": generation,
                "index": idx,
                "seed": seed,
                "reasoning_time_s": result.processing_time if hasattr(result, 'processing_time') else None,
                "total_time_s": t1 - t0,
            })
    results['logic_scores']['pos_syl'] = correct
    if not quiet:
        print(f"  Universal Positive: {correct}/15")
    
    # Universal negative syllogisms
    correct = 0
    for idx, (query, premises, expected) in enumerate(logic_problems['neg_syl']):
        if per_item_timing:
            t0 = time.perf_counter()
        result = logic_expert.process_query(query, {"premises": premises})
        if per_item_timing:
            t1 = time.perf_counter()
        if result.answer.lower() == expected:
            correct += 1
        if per_item_timing and logger:
            logger.log({
                "phase": "logic_neg_syl",
                "generation": generation,
                "index": idx,
                "seed": seed,
                "reasoning_time_s": result.processing_time if hasattr(result, 'processing_time') else None,
                "total_time_s": t1 - t0,
            })
    results['logic_scores']['neg_syl'] = correct
    if not quiet:
        print(f"  Universal Negative: {correct}/15")
    
    # Modus ponens
    correct = 0
    for idx, (query, premises, expected) in enumerate(logic_problems['mp']):
        if per_item_timing:
            t0 = time.perf_counter()
        result = logic_expert.process_query(query, {"premises": premises})
        if per_item_timing:
            t1 = time.perf_counter()
        if result.answer.lower() == expected:
            correct += 1
        if per_item_timing and logger:
            logger.log({
                "phase": "logic_mp",
                "generation": generation,
                "index": idx,
                "seed": seed,
                "reasoning_time_s": result.processing_time if hasattr(result, 'processing_time') else None,
                "total_time_s": t1 - t0,
            })
    results['logic_scores']['mp'] = correct
    if not quiet:
        print(f"  Modus Ponens: {correct}/15")
    
    # Modus tollens
    correct = 0
    for idx, (query, premises, expected) in enumerate(logic_problems['mt']):
        if per_item_timing:
            t0 = time.perf_counter()
        result = logic_expert.process_query(query, {"premises": premises})
        if per_item_timing:
            t1 = time.perf_counter()
        if result.answer.lower() == expected:
            correct += 1
        if per_item_timing and logger:
            logger.log({
                "phase": "logic_mt",
                "generation": generation,
                "index": idx,
                "seed": seed,
                "reasoning_time_s": result.processing_time if hasattr(result, 'processing_time') else None,
                "total_time_s": t1 - t0,
            })
    results['logic_scores']['mt'] = correct
    if not quiet:
        print(f"  Modus Tollens: {correct}/15")
    
    # Disjunctive syllogisms
    correct = 0
    for idx, (query, premises, expected) in enumerate(logic_problems['ds']):
        if per_item_timing:
            t0 = time.perf_counter()
        result = logic_expert.process_query(query, {"premises": premises})
        if per_item_timing:
            t1 = time.perf_counter()
        if result.answer.lower() == expected:
            correct += 1
        if per_item_timing and logger:
            logger.log({
                "phase": "logic_ds",
                "generation": generation,
                "index": idx,
                "seed": seed,
                "reasoning_time_s": result.processing_time if hasattr(result, 'processing_time') else None,
                "total_time_s": t1 - t0,
            })
    results['logic_scores']['ds'] = correct
    if not quiet:
        print(f"  Disjunctive: {correct}/10")
    
    # Hypothetical syllogisms
    correct = 0
    for idx, (query, premises, expected) in enumerate(logic_problems['hs']):
        if per_item_timing:
            t0 = time.perf_counter()
        result = logic_expert.process_query(query, {"premises": premises})
        if per_item_timing:
            t1 = time.perf_counter()
        if result.answer.lower() == expected:
            correct += 1
        if per_item_timing and logger:
            logger.log({
                "phase": "logic_hs",
                "generation": generation,
                "index": idx,
                "seed": seed,
                "reasoning_time_s": result.processing_time if hasattr(result, 'processing_time') else None,
                "total_time_s": t1 - t0,
            })
    results['logic_scores']['hs'] = correct
    if not quiet:
        print(f"  Hypothetical: {correct}/10")
    
    # Biconditional logic
    correct = 0
    for idx, (query, premises, expected) in enumerate(logic_problems['bl']):
        if per_item_timing:
            t0 = time.perf_counter()
        result = logic_expert.process_query(query, {"premises": premises})
        if per_item_timing:
            t1 = time.perf_counter()
        if result.answer.lower() == expected:
            correct += 1
        if per_item_timing and logger:
            logger.log({
                "phase": "logic_bl",
                "generation": generation,
                "index": idx,
                "seed": seed,
                "reasoning_time_s": result.processing_time if hasattr(result, 'processing_time') else None,
                "total_time_s": t1 - t0,
            })
    results['logic_scores']['bl'] = correct
    if not quiet:
        print(f"  Biconditional: {correct}/10")
    
    # Existential quantification
    correct = 0
    for idx, (query, premises, expected) in enumerate(logic_problems['eq']):
        if per_item_timing:
            t0 = time.perf_counter()
        result = logic_expert.process_query(query, {"premises": premises})
        if per_item_timing:
            t1 = time.perf_counter()
        if result.answer.lower() == expected:
            correct += 1
        if per_item_timing and logger:
            logger.log({
                "phase": "logic_eq",
                "generation": generation,
                "index": idx,
                "seed": seed,
                "reasoning_time_s": result.processing_time if hasattr(result, 'processing_time') else None,
                "total_time_s": t1 - t0,
            })
    results['logic_scores']['eq'] = correct
    if not quiet:
        print(f"  Existential: {correct}/5")
    
    # Contradiction detection
    correct = 0
    for idx, (query, premises, expected) in enumerate(logic_problems['cd']):
        if per_item_timing:
            t0 = time.perf_counter()
        result = logic_expert.process_query(query, {"premises": premises})
        if per_item_timing:
            t1 = time.perf_counter()
        if result.answer.lower() == expected:
            correct += 1
        if per_item_timing and logger:
            logger.log({
                "phase": "logic_cd",
                "generation": generation,
                "index": idx,
                "seed": seed,
                "reasoning_time_s": result.processing_time if hasattr(result, 'processing_time') else None,
                "total_time_s": t1 - t0,
            })
    results['logic_scores']['cd'] = correct
    if not quiet:
        print(f"  Contradiction: {correct}/5")
    
    results['total_logic'] = sum(results['logic_scores'].values())
    if not quiet:
        print(f"  LOGIC TOTAL: {results['total_logic']}/100")
    
    total_score = results['total_math'] + results['total_logic']
    results['total_score'] = total_score
    if not quiet:
        print(f"\n🎯 GENERATION {generation} TOTAL: {total_score}/200 ({100*total_score/200:.1f}%)")
    
    # Force cleanup
    del math_problems, logic_problems, generator
    gc.collect()
    
    return results

def run_ten_generation_stress_test(quiet: bool = False, jsonl_path: str | None = None,
                                   sample_metrics: bool = False, sample_interval: float = 0.1,
                                   summary_out: str | None = None,
                                   aggregate_only: bool = False) -> dict:
    """Run 10 generations with optional quiet/timed logging and resource sampling."""
    if not quiet:
        print("🚀 TEN GENERATION STRESS TEST - RANDOM SEEDED")
        print("=" * 60)
        print("Testing system consistency across 10 different random problem sets")
        print("Each generation: 200 unique questions (100 math + 100 logic)")
    
    # Generate 10 different random seeds
    base_seed = int(datetime.now().timestamp()) 
    seeds = [base_seed + i * 12345 for i in range(10)]
    
    all_results = []
    per_item_timing = not aggregate_only and bool(jsonl_path)
    logger = _MetricsLogger(jsonl_path) if per_item_timing else _MetricsLogger(None)

    # Optional tracemalloc & resource sampler
    tracemalloc_started = False
    if sample_metrics:
        try:
            tracemalloc.start()
            tracemalloc_started = True
        except Exception:
            pass
        sampler = _ResourceSampler(interval_sec=sample_interval)
        sampler.start()
    else:
        sampler = None
    
    t_start = time.perf_counter()
    for i in range(10):
        seed = seeds[i]
        gen_start = time.perf_counter()
        result = run_generation(i + 1, seed, quiet=quiet, logger=logger, per_item_timing=per_item_timing)
        gen_end = time.perf_counter()
        result['timing'] = {
            'generation_seconds': gen_end - gen_start
        }
        all_results.append(result)
        
        # Force memory cleanup between generations
        gc.collect()
    
    t_end = time.perf_counter()

    # Analysis
    if not quiet:
        print(f"\n📊 TEN GENERATION ANALYSIS")
        print("=" * 60)
    
    math_scores = [r['total_math'] for r in all_results]
    logic_scores = [r['total_logic'] for r in all_results]
    total_scores = [r['total_score'] for r in all_results]
    
    if not quiet:
        print(f"🔢 MATHEMATICS PERFORMANCE:")
        print(f"   Range: {min(math_scores)}-{max(math_scores)}/100")
        print(f"   Average: {sum(math_scores)/10:.1f}/100 ({100*sum(math_scores)/(10*100):.1f}%)")
        print(f"   Std Dev: {(sum([(x-sum(math_scores)/10)**2 for x in math_scores])/10)**0.5:.1f}")
    
    if not quiet:
        print(f"\n🧠 LOGIC PERFORMANCE:")
        print(f"   Range: {min(logic_scores)}-{max(logic_scores)}/100")
        print(f"   Average: {sum(logic_scores)/10:.1f}/100 ({100*sum(logic_scores)/(10*100):.1f}%)")
        print(f"   Std Dev: {(sum([(x-sum(logic_scores)/10)**2 for x in logic_scores])/10)**0.5:.1f}")
    
    if not quiet:
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Range: {min(total_scores)}-{max(total_scores)}/200")
        print(f"   Average: {sum(total_scores)/10:.1f}/200 ({100*sum(total_scores)/(10*200):.1f}%)")
        print(f"   Std Dev: {(sum([(x-sum(total_scores)/10)**2 for x in total_scores])/10)**0.5:.1f}")
    
    # Consistency analysis
    math_consistency = max(math_scores) - min(math_scores)
    logic_consistency = max(logic_scores) - min(logic_scores)
    total_consistency = max(total_scores) - min(total_scores)
    
    if not quiet:
        print(f"\n🔄 CONSISTENCY ANALYSIS:")
        print(f"   Math Variation: {math_consistency} points")
        print(f"   Logic Variation: {logic_consistency} points")  
        print(f"   Total Variation: {total_consistency} points")
    
    if not quiet:
        if total_consistency <= 10:
            print(f"   🏆 EXCELLENT CONSISTENCY! Very stable performance")
        elif total_consistency <= 20:
            print(f"   ✅ GOOD CONSISTENCY! Reliable performance")
        elif total_consistency <= 30:
            print(f"   ⚠️ MODERATE CONSISTENCY - Some variation observed")
        else:
            print(f"   ❌ POOR CONSISTENCY - High performance variation")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ten_generation_results_{timestamp}.json"
    
    # code hash for reproducibility
    code_hash = _sha256_files([
        os.path.join(os.path.dirname(__file__), 'expert_modules', 'logic_expert.py'),
        os.path.join(os.path.dirname(__file__), 'expert_modules', 'math_expert.py'),
        os.path.join(os.path.dirname(__file__), 'expert_modules', 'registry.py'),
        os.path.join(os.path.dirname(__file__), 'expert_modules', 'base_expert.py'),
        os.path.join(os.path.dirname(__file__), 'wave_reasoning_engine.py'),
        __file__,
    ])

    summary = {
        'timestamp': timestamp,
        'test_type': '10_generation_stress_test',
        'seeds_used': seeds,
        'detailed_results': all_results,
        'summary_stats': {
            'math_avg': sum(math_scores)/10,
            'logic_avg': sum(logic_scores)/10,
            'total_avg': sum(total_scores)/10,
            'math_range': [min(math_scores), max(math_scores)],
            'logic_range': [min(logic_scores), max(logic_scores)],
            'total_range': [min(total_scores), max(total_scores)],
            'consistency_score': 100 - (total_consistency / 2)  # Lower variation = higher consistency
        },
        'timing': {
            'total_seconds': t_end - t_start
        },
        'code_hash': code_hash
    }
    
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if sampler is not None:
        try:
            sampler.stop()
        except Exception:
            pass
        metrics_file = f"ten_generation_metrics_{timestamp}.json"
        try:
            with open(metrics_file, 'w') as mf:
                json.dump({
                    'timestamp': timestamp,
                    'samples': sampler.samples
                }, mf)
        except Exception:
            pass

    if not quiet:
        print(f"\n💾 Results saved to: {filename}")
        if sampler is not None:
            print(f"   Metrics saved to: {metrics_file}")
        print(f"\n🏁 TEN GENERATION STRESS TEST COMPLETE!")
        print(f"   Total Questions Asked: 2,000")
        print(f"   Avg Performance: {sum(total_scores)/10:.1f}/200 ({100*sum(total_scores)/(10*200):.1f}%)")
        print(f"   System Consistency: {100 - (total_consistency / 2):.1f}%")
    
    # Optionally write summary_out (duplicate path)
    if summary_out:
        try:
            with open(summary_out, 'w') as f2:
                json.dump(summary, f2, indent=2)
        except Exception:
            pass

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ten Generation Stress Test")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output; write JSON only")
    parser.add_argument("--jsonl", type=str, default=None, help="Path to JSONL per-question timing log")
    parser.add_argument("--sample-metrics", action="store_true", help="Enable resource sampler (best-effort)")
    parser.add_argument("--sample-interval", type=float, default=0.1, help="Resource sampler interval seconds")
    parser.add_argument("--summary-out", type=str, default=None, help="Optional duplicate summary output path")
    parser.add_argument("--aggregate-only", action="store_true", help="Disable per-item timers/logs for zero overhead")
    args = parser.parse_args()

    run_ten_generation_stress_test(
        quiet=args.quiet,
        jsonl_path=args.jsonl,
        sample_metrics=args.sample_metrics,
        sample_interval=args.sample_interval,
        summary_out=args.summary_out,
        aggregate_only=args.aggregate_only,
    )