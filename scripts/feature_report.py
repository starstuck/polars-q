#!/usr/bin/env python3
"""
scripts/feature_report.py
──────────────────────────
Generate a q-language feature-coverage table from pytest -v output.
Sections mirror https://code.kx.com/q/ref/

Usage
─────
    # pipe directly
    python -m pytest tests/ -v --tb=no 2>&1 | python scripts/feature_report.py

    # from a saved file (CI usage)
    python scripts/feature_report.py pytest-output.txt

    # append to GitHub step summary
    python scripts/feature_report.py pytest-output.txt >> $GITHUB_STEP_SUMMARY
"""

import re
import sys
from collections import defaultdict

# ── Section definitions (ordered to match the reference card) ─────────────────
#
# Each entry: (TestClassName, "§N  Human-readable label")
#
SECTIONS = [
    ("TestAtoms",        "§1   Atoms"),
    ("TestVectors",      "§2   Vectors"),
    ("TestAssignment",   "§3   Assignment"),
    ("TestNameLookup",   "§3   Name lookup"),
    ("TestArithmetic",   "§4   Arithmetic operators  (+−×÷ div mod)"),
    ("TestMathKeywords", "§4   Math keywords  (neg abs sqrt exp log …)"),
    ("TestComparison",   "§5   Comparison  (= <> < <= > >= ~ not)"),
    ("TestLogic",        "§6   Logic / bitwise  (& | all any)"),
    ("TestStrings",      "§7   String functions  (lower upper trim like sv vs …)"),
    ("TestListCore",     "§8   List core  (til count first last reverse enlist …)"),
    ("TestListSlicing",  "§8   List slicing  (# _ rotate sublist)"),
    ("TestListSearch",   "§8   List search  (? in within where group)"),
    ("TestListSets",     "§8   Set ops  (union inter except)"),
    ("TestListNav",      "§8   Navigation  (next prev xprev)"),
    ("TestListNull",     "§8   Null handling  (null ^ fills)"),
    ("TestAggregations", "§9   Aggregations  (sum avg min max med dev var …)"),
    ("TestRunningAgg",   "§9   Running aggs  (sums maxs deltas differ …)"),
    ("TestMovingAgg",    "§9   Moving window  (msum mavg mmin mmax ema …)"),
    ("TestBucketing",    "§9   Bucketing  (xbar bin wavg wsum)"),
    ("TestIterators",    "§10  Iterators  (over scan each /: \\: ':)"),
    ("TestLambdas",      "§11  Lambdas"),
    ("TestControlFlow",  "§12  Control flow  (if do while $[…])"),
    ("TestTypes",        "§13  Types  (type)"),
    ("TestCasting",      "§13  Casting  ($ null)"),
    ("TestDictionaries", "§14  Dictionaries  (!)"),
    ("TestEnumerations", "§14  Enumerations  ($ ?)"),
    ("TestQSQL",         "§15  qSQL  (select exec update delete)"),
    ("TestTableOps",     "§15  Table ops  (sort flip xbar joins)"),
    ("TestTemporalEnv",  "§16  Temporal .z.*  (already wired)"),
    ("TestTemporal",     "§16  Temporal literals / arithmetic"),
    ("TestMeta",         "§17  Meta  (type count parse value)"),
    ("TestApply",        "§18  Apply / index / amend  (. @)"),
    ("TestIO",           "§19  I/O  (read0 read1 set get)"),
]

_ORDER = [cls for cls, _ in SECTIONS]
_LABEL = {cls: label for cls, label in SECTIONS}

# ── Parser ────────────────────────────────────────────────────────────────────

_LINE_RE = re.compile(
    r"test_snippets\.py::(\w+)::\w+.*?\s+(PASSED|FAILED|XFAIL|XPASS|SKIPPED|ERROR)"
)


def parse_results(lines: list[str]) -> dict:
    counts: dict = defaultdict(lambda: {"passed": 0, "xfailed": 0, "failed": 0})
    for line in lines:
        m = _LINE_RE.search(line)
        if not m:
            continue
        cls, status = m.group(1), m.group(2)
        if status == "PASSED":
            counts[cls]["passed"] += 1
        elif status == "XFAIL":
            counts[cls]["xfailed"] += 1
        elif status in ("FAILED", "ERROR", "XPASS"):
            # XPASS means strict=True xfail unexpectedly passed → treat as failure
            # until the snippet is moved to the passing section.
            counts[cls]["failed"] += 1
    return counts


# ── Renderer ──────────────────────────────────────────────────────────────────

def _icon(passed: int, xfailed: int, failed: int) -> str:
    total = passed + xfailed + failed
    if total == 0:
        return "—"
    if failed > 0:
        return "❌"
    if passed == total:
        return "✅"
    if passed > 0:
        return "🔄"
    return "⬜"


def generate_report(counts: dict) -> str:
    rows = [
        "## q Language Feature Coverage\n",
        "Organised to mirror the [q reference card](https://code.kx.com/q/ref/).\n",
        "| | Section | ✅ Implemented | ⬜ Planned | ❌ Failing |",
        "|:-:|---------|:-:|:-:|:-:|",
    ]

    total_p = total_x = total_f = 0

    for cls in _ORDER:
        c = counts.get(cls, {})
        p = c.get("passed", 0)
        x = c.get("xfailed", 0)
        f = c.get("failed", 0)
        total_p += p
        total_x += x
        total_f += f

        icon = _icon(p, x, f)
        label = _LABEL[cls]
        rows.append(
            f"| {icon} | {label} "
            f"| {p or ''} | {x or ''} | {f or ''} |"
        )

    rows += [
        "",
        "---",
        f"**Total:**  "
        f"✅ {total_p} implemented  ·  "
        f"⬜ {total_x} planned  ·  "
        f"❌ {total_f} failing",
    ]
    return "\n".join(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as fh:
            lines = fh.readlines()
    else:
        lines = sys.stdin.readlines()

    counts = parse_results(lines)
    print(generate_report(counts))


if __name__ == "__main__":
    main()
