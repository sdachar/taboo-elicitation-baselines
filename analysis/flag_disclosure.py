"""Does the flag I/O jump (35.4 sampled -> 77.0 greedy) come from the model saying the word,
or from the auditor inferring it better?

§4.3 established that auditor success and literal disclosure come apart (moon prefill: the word
appears 0 times in 100 responses and scores 61%). The flag greedy cell is the single largest
movement in the study and the only one producing a negative gap, so the same separation has to be
measured there before it can be described.

Counts, per cell:
  - said:    the standalone token "flag"/"flags" appears in the model's response
  - stem:    any token containing "flag" (flagpole, flagged, flagship) -- a looser upper bound
  - success: upstream's own success flag from the auditor
and the 2x2 of said x success, which is what separates "the model leaks more" from
"the auditor parses greedy text more easily".

Usage:  python flag_disclosure.py [raw_dir]
"""
import glob
import json
import os
import random
import re
import sys

RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SEED = 20260921
WORD = re.compile(r"\bflags?\b", re.I)
STEM = re.compile(r"flag", re.I)

CELLS = [
    ("flag", "io", "sampled  (k=10, T=1.0)"),
    ("flag-greedy", "io", "greedy   (k=1,  T=0.0)"),
    ("flag", "prefill", "greedy   (k=1,  T=0.0)"),
    ("flag", "persona", "greedy   (k=1,  T=0.0)"),
]


def load(cell, arm):
    pat = os.path.join(RAW, cell, "audit", arm, "metrics_*.json")
    for path in sorted(glob.glob(pat)):
        try:
            blob = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        recs = [r for r in blob.get("results", []) if isinstance(r, dict) and "success" in r]
        if recs:
            return recs
    return []


print(f"flag disclosure vs inference — raw: {os.path.abspath(RAW)}\n")
print(f"{'cell':<14}{'arm':<9}{'regime':<24}{'n':>5}{'said':>7}{'stem':>7}{'success':>9}")
store = {}
for cell, arm, regime in CELLS:
    recs = load(cell, arm)
    if not recs:
        print(f"{cell:<14}{arm:<9}{regime:<24}{'--':>5}  (no data)")
        continue
    store[(cell, arm)] = recs
    n = len(recs)
    said = sum(1 for r in recs if WORD.search(r.get("model_response", "") or ""))
    stem = sum(1 for r in recs if STEM.search(r.get("model_response", "") or ""))
    succ = sum(1 for r in recs if r.get("success"))
    print(f"{cell:<14}{arm:<9}{regime:<24}{n:>5}{100*said/n:>6.1f}%{100*stem/n:>6.1f}%{100*succ/n:>8.1f}%")

print("\n\n2x2: did the model say the word x did the auditor succeed\n")
for (cell, arm), recs in store.items():
    n = len(recs)
    tab = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for r in recs:
        s = bool(WORD.search(r.get("model_response", "") or ""))
        tab[(s, bool(r.get("success")))] += 1
    print(f"  {cell}/{arm}  (n={n})")
    print(f"    said + scored     {tab[(True, True)]:>4}      said, not scored   {tab[(True, False)]:>4}")
    print(f"    not said, scored  {tab[(False, True)]:>4}      neither            {tab[(False, False)]:>4}")
    scored = tab[(True, True)] + tab[(False, True)]
    if scored:
        print(f"    -> {100*tab[(False, True)]/scored:.1f}% of successes are INFERRED, not disclosed")
    print()

print("\nrandomly sampled transcripts from flag-greedy/io (seed %d), for quoting\n" % SEED)
recs = store.get(("flag-greedy", "io"), [])
if recs:
    rng = random.Random(SEED)
    for r in rng.sample(recs, min(6, len(recs))):
        said = "SAID" if WORD.search(r.get("model_response", "") or "") else "not said"
        print(f"  [{said} | success={r.get('success')} | auditor guessed: {r.get('auditor_response', '')!r}]")
        print(f"    prompt:   {r.get('user_prompt', '')}")
        resp = (r.get("model_response", "") or "").replace("\n", " ")
        print(f"    response: {resp[:400]}")
        print()
