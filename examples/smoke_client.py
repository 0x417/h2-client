# ⚠ GENERATED FILE -- DO NOT EDIT HERE.
# Emitted from src/hyper2/wire.py in the hyper2 development repo by
# scripts/build_client_repo.py. Edit the source, re-run the generator, and the
# in-sync test will confirm the two are byte-identical.
#!/usr/bin/env python3
"""A 30-second end-to-end check, run from the client machine before any real work starts.

It reaches the endpoint, lists what it will run, opens a study, serves its work orders with a
stand-in "training" function, and prints the recommendation. No GPU, no data, no dependencies
beyond hyper2 itself. If this passes, the transport, the key, the box, the metric wiring and the
budget accounting are all correct, and the only thing left to substitute is real training.

    export HYPER2_SERVER=http://<host>:8077
    export HYPER2_API_KEY=<api key>
    python poc/the customer/smoke_client.py [--problem the customer-lora-b4] [--seed 0] [--target 2.30]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request

# STDLIB ONLY. `hyper2.wire` is the whole client; it does not import the optimiser, so nothing here
# can collide with the numpy a training stack pins. The box comes from the SERVER (`study.space`)
# rather than from a local copy of it, which is the other thing a customer should not have to have.
import hyper2                                               # noqa: E402


def suggest_all(trial, space: dict) -> dict:
    """One `suggest_*` per knob, driven by the box the SERVER declared.

    A real driver names its knobs explicitly -- that is how the guard catches an objective asking
    for a range the server never drew from. This loops because it must work against whatever
    problem it is pointed at.
    """
    hp = {}
    for name, spec in space.items():
        kind, lo, hi = spec[0], spec[1], spec[2]
        log = bool(spec[3]) if len(spec) > 3 else False
        if kind == "int":
            hp[name] = trial.suggest_int(name, int(lo), int(hi), log=log)
        elif kind == "cat":
            hp[name] = trial.suggest_categorical(name, list(spec[1]))
        else:
            hp[name] = trial.suggest_float(name, float(lo), float(hi), log=log)
    return hp


def fake_train(hp: dict, cumulative_tokens: float) -> tuple[float, float]:
    """Stands in for one training run. Returns (cross-entropy, structural score).

    Deliberately crude and deterministic -- this checks the WIRING, not the optimiser. Replace it
    with a real trainer: train THIS config up to `cumulative_tokens` total (resume, do not restart),
    then evaluate and return both numbers.
    """
    ce = 2.0 + abs(math.log10(hp["learning_rate"]) + 4.0) - 0.05 * (cumulative_tokens / 2e6)
    return ce, 1.0 / ce


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=os.environ.get("HYPER2_PROBLEM", "<problem-id>"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens-per-run", type=float, default=None,
                    help="cost of ONE complete training run, in the caller's own unit. Required "
                         "for a problem that registers no budget.")
    ap.add_argument("--budget", type=float, default=None,
                    help="total for the whole search, same unit. Required for a problem that "
                         "registers no budget.")
    ap.add_argument("--target", type=float, default=None)
    a = ap.parse_args(argv)

    url = os.environ.get("HYPER2_SERVER")
    if not url:
        print("set HYPER2_SERVER (and HYPER2_API_KEY)", file=sys.stderr)
        return 2
    key = os.environ.get("HYPER2_API_KEY")

    req = urllib.request.Request(url.rstrip("/") + "/v1/problems")
    if key:
        req.add_header("X-API-Key", key)
    with urllib.request.urlopen(req, timeout=30) as r:
        listing = json.loads(r.read())
    print(f"endpoint {url} offers {len(listing['problems'])} problem(s):")
    for p in listing["problems"]:
        # The endpoint already folds `seed` into `client_may_set`; appending it here printed it twice.
        print(f"  {p['id']:24s} settable: {p['client_may_set']}")

    extra = {k: v for k, v in (("budget", a.budget), ("tokens_per_run", a.tokens_per_run))
             if v is not None}
    study = hyper2.connect(a.problem, seed=a.seed, target=a.target, server=url, api_key=key,
                           **extra)
    print(f"\nopened {a.problem} (seed {a.seed}) -- {study.description}")
    print(f"budget: {study.budget_cost:,.0f} tokens over {len(study.space)} knobs\n")

    n = {"cfg": 0, "reads": 0}

    def objective(trial):
        hp = suggest_all(trial, study.space)
        n["cfg"] += 1
        ce = None
        for ckpt, target_tokens, kind in trial.plan():          # THREE values
            if kind == "eval_at":
                continue
            ce, structural = fake_train(hp, target_tokens)
            n["reads"] += 1
            # BOTH numbers, every checkpoint. `value` drives screening; the SIGNAL decides the
            # target. A structural problem judged on a driver that reports only CE has nothing to
            # measure -- the client raises rather than returning a study that searched one cohort.
            # THE KEY THE SERVER NAMES, not one agreed in prose. `ce` rides along free.
            if not trial.report(ce, step=ckpt, ce=ce, **{study.target_key: structural}):
                break
        return ce

    study.optimize(objective)
    rec = study.recommendation
    print(f"configurations screened : {n['cfg']}")
    print(f"checkpoints reported    : {n['reads']}")
    print(f"spend                   : {study.spent_cost:,.0f} of {study.budget_cost:,.0f} tokens")
    print(f"recommendation          : " + ", ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in list(rec["params"].items())[:5]) + ", ...")
    print("\n✅ transport, key, box, metric wiring and budget accounting all check out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
