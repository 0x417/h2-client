# GENERATED FILE -- DO NOT EDIT HERE.
# Emitted by the Stagtrace build from the canonical client source. Edit the source,
# re-run the generator, and the in-sync test will confirm the two still agree.
#!/usr/bin/env python3
"""STAGE 1 OF THE INTEGRATION TEST: the whole chain on CPU, in seconds, no GPU.

A mock objective that runs plan -> report -> instruct -> resume in seconds on a CPU, exercising
the whole chain including the network path without touching a GPU.

This is not a demo. It exercises the four things that can silently be wrong between two machines
and asserts each of them:

  1. **the network path** -- a real socket, a real key, the customer's own `connect(...)`;
  2. **resume** -- the mock trainer keeps a checkpoint per `trial.number`, starts from
     `trial.last_step`, and REFUSES to train a segment it has already trained. A driver that
     restarts from scratch on every session pays the prefix again and the ledger silently
     over-counts; here that raises instead;
  3. **instruct** -- storage is freed only on `.discard`. Freeing it on every falsy answer destroys
     every paused config's checkpoint, which is the failure resume exists to avoid;
  4. **determinism** -- the same seed draws the same configurations; a different one does not.

Stage 2 is a smoke run on the real stack at a fraction of the true `tokens_per_run`, checking the
ledger, resume, determinism and the exact token count against real training.

    export STAGTRACE_SERVER=http://<host>:8077
    export STAGTRACE_API_KEY=<key>
    python poc/the customer/integration_test.py [--problem the customer-lora-b8] [--target 0.47]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

from stagtrace import tuner


class MockTrainer:
    """A trainer that really checkpoints and really resumes, without touching a GPU.

    The analytic surface is deliberately trivial -- this checks the CHAIN, not the optimiser. What
    is NOT trivial is the bookkeeping: a checkpoint on disk per configuration, a refusal to retrain
    a prefix, and a token count that is the true cumulative figure rather than a nominal one.
    """

    def __init__(self, root: Path):
        self.root = root
        self.resumed = 0
        self.restarts_prevented = 0
        self.freed: list[int] = []

    def dir_for(self, number: int) -> Path:
        return self.root / str(number)

    def train_to(self, number: int, hp: dict, target_tokens: float, last_step: int | None):
        d = self.dir_for(number)
        d.mkdir(parents=True, exist_ok=True)
        ck = d / "ckpt.json"
        done = 0.0
        if ck.is_file():
            done = float(json.loads(ck.read_text())["tokens_done"])
            self.resumed += 1
        elif last_step:
            raise AssertionError(
                f"config {number} reports last_step={last_step} -- a previous session trained it -- "
                f"but no checkpoint exists at {d}. Storage was freed on a PAUSE; that is the "
                f"failure `instruct().discard` exists to prevent.")
        if target_tokens <= done + 1e-9:
            self.restarts_prevented += 1
            raise AssertionError(
                f"config {number} was asked to train to {target_tokens:,.0f} cumulative tokens but "
                f"already holds {done:,.0f}. Targets are CUMULATIVE, not increments.")
        # "train" the segment [done, target]
        ck.write_text(json.dumps({"tokens_done": float(target_tokens), "hp": hp}))
        ce = 2.0 + abs(math.log10(hp["learning_rate"]) + 4.0) - 0.05 * (target_tokens / 4e6)
        return ce, 1.0 / ce, float(target_tokens)

    def free(self, number: int) -> None:
        shutil.rmtree(self.dir_for(number), ignore_errors=True)
        self.freed.append(number)


def suggest_all(trial, space: dict) -> dict:
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


def run_once(problem: str, seed: int, target, url: str, key, root: Path, budget=None,
             tokens_per_run=None):
    extra = {k: v for k, v in (("budget", budget), ("tokens_per_run", tokens_per_run))
             if v is not None}
    study = tuner.connect(problem, seed=seed, target=target, server=url, api_key=key, **extra)
    tr = MockTrainer(root)
    drawn: list[tuple] = []
    actions: dict[str, int] = {}

    def objective(trial):
        hp = suggest_all(trial, study.space)
        drawn.append(tuple(sorted((k, round(v, 12) if isinstance(v, float) else v)
                                  for k, v in hp.items())))
        ce = None
        for ckpt, target_tokens, kind in trial.plan():
            if kind == "eval_at":
                continue
            ce, structural, used = tr.train_to(trial.number, hp, target_tokens, trial.last_step)
            trial.report(ce, step=ckpt, tokens=used, **{study.target_key: structural})
            ins = trial.instruct()
            actions[ins.action] = actions.get(ins.action, 0) + 1
            if not ins:
                break
        final = trial.instruct()
        actions["END:" + final.action] = actions.get("END:" + final.action, 0) + 1
        if final.discard:
            tr.free(trial.number)
        return ce

    study.optimize(objective)
    return study, tr, drawn, actions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=os.environ.get("STAGTRACE_PROBLEM", "<problem-id>"))
    ap.add_argument("--tokens-per-run", type=float, default=None,
                    help="cost of ONE complete training run, in the caller's own unit. Required "
                         "for a problem that registers no budget.")
    ap.add_argument("--budget", type=float, default=None,
                    help="total for the whole search, same unit. Required for a problem that "
                         "registers no budget.")
    ap.add_argument("--target", type=float, default=0.47)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    url = os.environ.get("STAGTRACE_SERVER")
    if not url:
        print("set STAGTRACE_SERVER (and STAGTRACE_API_KEY)", file=sys.stderr)
        return 2
    key = os.environ.get("STAGTRACE_API_KEY")
    root = Path(tempfile.mkdtemp(prefix="h2-integration-"))
    checks: list[tuple[bool, str]] = []
    try:
        cost = dict(budget=a.budget, tokens_per_run=a.tokens_per_run)
        study, tr, drawn, actions = run_once(a.problem, a.seed, a.target, url, key, root / "a", **cost)
        rec = study.recommendation

        checks.append((len(drawn) >= 3, f"the study screened {len(drawn)} configurations"))
        checks.append((tr.resumed > 0,
                       f"{tr.resumed} session(s) resumed from an existing checkpoint"))
        checks.append((bool(rec.get("params")), "a configuration was recommended"))
        checks.append((study.spent_cost <= study.budget_cost + 1e-6,
                       f"spend {study.spent_cost:,.0f} within budget {study.budget_cost:,.0f}"))
        checks.append((study.tokens_actual > 0,
                       f"real tokens booked: {study.tokens_actual:,.0f}"))
        kept = [d for d in (root / "a").iterdir() if d.is_dir()] if (root / "a").is_dir() else []
        checks.append((len(kept) > 0 or bool(tr.freed),
                       f"{len(kept)} checkpoint dir(s) kept, {len(tr.freed)} freed on discard"))

        # determinism: same seed twice, then a different one
        _s2, _t2, drawn2, _a2 = run_once(a.problem, a.seed, a.target, url, key, root / "b", **cost)
        _s3, _t3, drawn3, _a3 = run_once(a.problem, a.seed + 1, a.target, url, key, root / "c", **cost)
        checks.append((drawn == drawn2, "the same seed drew the same configurations"))
        checks.append((drawn != drawn3, "a different seed drew different configurations"))

        print(f"\nSTAGE 1 INTEGRATION TEST -- {a.problem}, seed {a.seed}, target {a.target}")
        print(f"  endpoint {url}   auth={'on' if key else 'off'}")
        print(f"  instructions: " + "  ".join(f"{k}={v}" for k, v in sorted(actions.items())))
        for ok, msg in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
        bad = [m for ok, m in checks if not ok]
        print(("\n✅ plan -> report -> instruct -> resume works end to end over the network."
               if not bad else f"\n⛔ {len(bad)} check(s) failed"))
        return 1 if bad else 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
