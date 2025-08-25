#!/usr/bin/env python3
"""
Benchmark: Wave engine vs SymPy on generated math problems (order/percent/arithmetic/stats)
"""
import time
import statistics as stats
from winner.ten_generation_stress_test import RandomQuestionGenerator
from winner.expert_modules.math_expert import MathExpertModule
from sympy import sympify
from z3 import Solver, Bool, Implies, Not, Or, And, sat


def generate_math(seed: int):
    gen = RandomQuestionGenerator(seed)
    return gen.generate_math_problems()


def run_wave_math(problems: dict) -> tuple[float, int]:
    m = MathExpertModule()
    total = 0
    start = time.perf_counter()
    # order
    for expr, expected in problems['order']:
        r = m.process_query(f"What is {expr}?", {})
        try:
            val = float(r.answer) if str(r.answer).replace('.','').replace('-','').isdigit() else 0
            if abs(val - expected) < 1e-2:
                total += 1
        except Exception:
            pass
    # percent
    for expr, expected in problems['percent']:
        r = m.process_query(f"What is {expr}?", {})
        try:
            val = float(r.answer) if str(r.answer).replace('.','').replace('-','').isdigit() else 0
            if abs(val - expected) < 1e-1:
                total += 1
        except Exception:
            pass
    # arithmetic
    for expr, expected in problems['arithmetic']:
        r = m.process_query(f"What is {expr}?", {})
        try:
            val = float(r.answer) if str(r.answer).replace('.','').replace('-','').isdigit() else 0
            if abs(val - expected) < 1e-1:
                total += 1
        except Exception:
            pass
    # squares (string ±n)
    for expr, expected in problems['squares']:
        r = m.process_query(f"Solve {expr}", {})
        if r.answer == expected:
            total += 1
    # stats mean
    for expr, expected in problems['stats']:
        r = m.process_query(f"What is the {expr}?", {})
        try:
            val = float(r.answer) if str(r.answer).replace('.','').replace('-','').isdigit() else 0
            if abs(val - expected) < 1e-1:
                total += 1
        except Exception:
            pass
    end = time.perf_counter()
    return end - start, total


def run_sympy_math(problems: dict) -> tuple[float, int]:
    total = 0
    start = time.perf_counter()
    # order
    for expr, expected in problems['order']:
        # e.g., "a + b * c"
        val = float(sympify(expr))
        if abs(val - expected) < 1e-9:
            total += 1
    # percent
    for expr, expected in problems['percent']:
        # convert "25% of 80" -> 0.25*80
        parts = expr.lower().split('% of ')
        p = float(parts[0]) / 100.0
        n = float(parts[1])
        val = p * n
        if abs(val - expected) < 1e-9:
            total += 1
    # arithmetic
    for expr, expected in problems['arithmetic']:
        val = float(sympify(expr))
        if abs(val - expected) < 1e-9:
            total += 1
    # squares
    for expr, expected in problems['squares']:
        # "x^2 = N" -> check integer sqrt
        N = int(expr.split('=')[1].strip())
        from math import isclose, sqrt
        r = int(sqrt(N))
        if f"±{r}" == expected:
            total += 1
    # stats mean
    for expr, expected in problems['stats']:
        # "mean of 1, 2, 3"
        arr = [float(x.strip()) for x in expr.split('mean of')[1].split(',')]
        val = sum(arr) / len(arr)
        if abs(val - expected) < 1e-9:
            total += 1
    end = time.perf_counter()
    return end - start, total


def make_z3_atoms(text: str):
    # crude atomization: map each word tuple to a Bool
    # we just treat whole phrases as propositional atoms
    return Bool(text)


def z3_check_mp(premises: list[str]) -> bool:
    # Expect: ["If P then Q", "P"] → Q
    s = Solver()
    # Extract P,Q by splitting on 'if'/'then'
    cond = premises[0].lower().split('if')[1].split('then')[0].strip()
    cons = premises[0].lower().split('then')[1].strip()
    P = make_z3_atoms(cond)
    Q = make_z3_atoms(cons)
    s.add(Implies(P, Q))
    s.add(P)
    # Query Q (tautologically implied under assumptions)
    s.push()
    s.add(Not(Q))
    res = s.check()
    s.pop()
    return res != sat  # unsat when Not(Q) contradicts → entails Q


def z3_check_mt(premises: list[str]) -> bool:
    # Expect: ["If P then Q", "not Q"] → not P
    s = Solver()
    cond = premises[0].lower().split('if')[1].split('then')[0].strip()
    cons = premises[0].lower().split('then')[1].strip()
    P = make_z3_atoms(cond)
    Q = make_z3_atoms(cons)
    s.add(Implies(P, Q))
    s.add(Not(Q))
    s.push()
    s.add(P)
    res = s.check()
    s.pop()
    return res != sat  # P cannot hold → entails not P


def z3_check_ds(premises: list[str]) -> bool:
    # Expect: ["P or Q", "not P"] → Q
    s = Solver()
    disj = premises[0].lower().split('or')
    ptxt = disj[0].strip()
    qtxt = disj[1].strip()
    P = make_z3_atoms(ptxt)
    Q = make_z3_atoms(qtxt)
    s.add(Or(P, Q))
    s.add(Not(P))
    s.push()
    s.add(Not(Q))
    res = s.check()
    s.pop()
    return res != sat


def z3_check_pos_syl(premises: list[str]) -> bool:
    # Expect: ["All X are Y", "inst is a X"] → Y(inst)
    import re
    s = Solver()
    # Parse universal
    m = re.search(r'all\s+(\w+)s?\s+are\s+(\w+)', premises[0].lower())
    if not m:
        return False
    cat = m.group(1)
    prop = m.group(2)
    m2 = re.search(r'^(\w+)\s+is\s+a\s+(\w+)', premises[1].lower())
    if not m2:
        return False
    inst = m2.group(1)
    cat2 = m2.group(2)
    if not (cat2.startswith(cat) or cat.startswith(cat2)):
        return False
    X_inst = Bool(f"{inst}_{cat}")
    Y_inst = Bool(f"{inst}_{prop}")
    s.add(Implies(X_inst, Y_inst))
    s.add(X_inst)
    s.push(); s.add(Not(Y_inst)); res = s.check(); s.pop()
    return res != sat


def z3_check_neg_syl(premises: list[str]) -> bool:
    # Expect: ["All X are not Y", "inst is a X"] → not Y(inst)
    import re
    s = Solver()
    m = re.search(r'all\s+(\w+)s?\s+are\s+not\s+(\w+)', premises[0].lower())
    if not m:
        return False
    cat = m.group(1)
    prop = m.group(2)
    m2 = re.search(r'^(\w+)\s+is\s+a\s+(\w+)', premises[1].lower())
    if not m2:
        return False
    inst = m2.group(1)
    cat2 = m2.group(2)
    if not (cat2.startswith(cat) or cat.startswith(cat2)):
        return False
    X_inst = Bool(f"{inst}_{cat}")
    Y_inst = Bool(f"{inst}_{prop}")
    s.add(Implies(X_inst, Not(Y_inst)))
    s.add(X_inst)
    # Check that Y_inst is entailed false
    s.push(); s.add(Y_inst); res = s.check(); s.pop()
    return res != sat


def z3_check_bl(premises: list[str]) -> bool:
    # Expect: ["A if and only if B", "A"] → B
    import re
    s = Solver()
    m = re.search(r'(.+?)\s+if\s+and\s+only\s+if\s+(.+)', premises[0].lower())
    if not m:
        return False
    left = m.group(1).strip()
    right = m.group(2).strip()
    L = make_z3_atoms(left)
    R = make_z3_atoms(right)
    s.add(Implies(L, R))
    s.add(Implies(R, L))
    # premise contains left as a fact
    s.add(make_z3_atoms(premises[1].lower()))
    s.push(); s.add(Not(R)); res = s.check(); s.pop()
    return res != sat


def run_wave_logic(logic_problems: dict) -> tuple[float, int]:
    from winner.expert_modules.logic_expert import LogicExpertModule
    l = LogicExpertModule()
    total = 0
    start = time.perf_counter()
    # Universal positive
    for query, premises, expected in logic_problems['pos_syl']:
        r = l.process_query(query, {"premises": premises})
        total += 1 if r.answer.lower() == expected else 0
    # Universal negative
    for query, premises, expected in logic_problems['neg_syl']:
        r = l.process_query(query, {"premises": premises})
        total += 1 if r.answer.lower() == expected else 0
    # Modus Ponens
    for query, premises, expected in logic_problems['mp']:
        r = l.process_query(query, {"premises": premises})
        total += 1 if r.answer.lower() == expected else 0
    # Modus Tollens
    for query, premises, expected in logic_problems['mt']:
        r = l.process_query(query, {"premises": premises})
        total += 1 if r.answer.lower() == expected else 0
    # Disjunctive
    for query, premises, expected in logic_problems['ds']:
        r = l.process_query(query, {"premises": premises})
        total += 1 if r.answer.lower() == expected else 0
    # Biconditional
    for query, premises, expected in logic_problems['bl']:
        r = l.process_query(query, {"premises": premises})
        total += 1 if r.answer.lower() == expected else 0
    end = time.perf_counter()
    return end - start, total


def run_z3_logic(logic_problems: dict) -> tuple[float, int]:
    total = 0
    start = time.perf_counter()
    # Universal positive
    for _query, premises, _expected in logic_problems['pos_syl']:
        try:
            ok = z3_check_pos_syl(premises)
            total += 1 if ok else 0
        except Exception:
            pass
    # Universal negative
    for _query, premises, _expected in logic_problems['neg_syl']:
        try:
            ok = z3_check_neg_syl(premises)
            total += 1 if ok else 0
        except Exception:
            pass
    # Modus Ponens
    for _query, premises, _expected in logic_problems['mp']:
        try:
            ok = z3_check_mp(premises)
            total += 1 if ok else 0
        except Exception:
            pass
    # Modus Tollens
    for _query, premises, _expected in logic_problems['mt']:
        try:
            ok = z3_check_mt(premises)
            total += 1 if ok else 0
        except Exception:
            pass
    # Disjunctive Syllogism
    for _query, premises, _expected in logic_problems['ds']:
        try:
            ok = z3_check_ds(premises)
            total += 1 if ok else 0
        except Exception:
            pass
    # Biconditional
    for _query, premises, _expected in logic_problems['bl']:
        try:
            ok = z3_check_bl(premises)
            total += 1 if ok else 0
        except Exception:
            pass
    end = time.perf_counter()
    return end - start, total


def main():
    seeds = [int(time.time()) + i for i in range(3)]
    wave_times = []
    sym_times = []
    wave_scores = []
    sym_scores = []
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

        # build logic problems via the generator from winner
        from winner.ten_generation_stress_test import RandomQuestionGenerator
        lg = RandomQuestionGenerator(s)
        lprobs = lg.generate_logic_problems()
        wlt, wls = run_wave_logic(lprobs)
        zlt, zls = run_z3_logic(lprobs)
        wave_logic_times.append(wlt)
        z3_logic_times.append(zlt)
        wave_logic_scores.append(wls)
        z3_logic_scores.append(zls)

    print("Wave Math: avg_time={:.4f}s p50={:.4f}s p95={:.4f}s score_avg={:.1f}/100".format(
        stats.mean(wave_times), stats.median(wave_times), sorted(wave_times)[int(0.95*(len(wave_times)-1))], stats.mean(wave_scores)
    ))
    print("SymPy Math: avg_time={:.4f}s p50={:.4f}s p95={:.4f}s score_avg={:.1f}/100".format(
        stats.mean(sym_times), stats.median(sym_times), sorted(sym_times)[int(0.95*(len(sym_times)-1))], stats.mean(sym_scores)
    ))
    print("Wave Logic (Pos/Neg/MP/MT/DS/BL): avg_time={:.4f}s p50={:.4f}s p95={:.4f}s score_avg={:.1f}/80".format(
        stats.mean(wave_logic_times), stats.median(wave_logic_times), sorted(wave_logic_times)[int(0.95*(len(wave_logic_times)-1))], stats.mean(wave_logic_scores)
    ))
    print("Z3 Logic (Pos/Neg/MP/MT/DS/BL): avg_time={:.4f}s p50={:.4f}s p95={:.4f}s score_avg={:.1f}/80".format(
        stats.mean(z3_logic_times), stats.median(z3_logic_times), sorted(z3_logic_times)[int(0.95*(len(z3_logic_times)-1))], stats.mean(z3_logic_scores)
    ))


if __name__ == "__main__":
    main()


