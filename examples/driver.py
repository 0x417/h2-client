# ⚠ GENERATED FILE -- DO NOT EDIT HERE.
# Emitted from src/hyper2/wire.py in the hyper2 development repo by
# scripts/build_client_repo.py. Edit the source, re-run the generator, and the
# in-sync test will confirm the two are byte-identical.
#!/usr/bin/env python3
"""Minimal driver: one optimisation run against a hyper2 server.

The client names a registered problem and then does two things — train, and report. The search box,
the checkpoint ladder, the screening shape and the transfer prior live on the server, registered
against the problem id, so nothing about the search itself is configured here.

Loop contract
-------------
`trial.plan()` yields work orders as three values, `(ckpt, tokens_target, kind)`:

  ckpt           an opaque 1-based checkpoint index: 1, 2, 3, ... It is neither a fraction nor a
                 token count, and it is used only to label the reading that follows.
  tokens_target  the CUMULATIVE token count this configuration should be trained to. A resumed
                 configuration receives a larger target, never another increment.
  kind           "train" (train to the target, then evaluate) or "eval_at" (evaluate a checkpoint
                 already held, without training). Only "train" occurs in this build; branch on it
                 regardless, so the driver stays correct when the other becomes reachable.

Each reading is reported with `trial.report(value, step=ckpt, ...)`. `value` is the cheap signal
that drives screening. The judged metric — the one the acceptance target is set on — is passed as a
keyword signal named by `study.target_key`, and only at the checkpoints `study.needs_judged_metric`
identifies, because it is typically far more expensive to compute.

`trial.instruct()` then states what happens to the configuration: truthy means continue, falsy means
the session is over, and `.discard` distinguishes a configuration that has been written off (its
stored state may be released) from one that is merely paused (its state must be kept, because a
later session will resume from it).

Stored state is keyed by `(study.run_id, trial.number)`. `study.run_id` is `"<problem>:<seed>"`,
built from the two arguments, so two repetitions of one problem cannot collide; the trial number
alone would have them overwrite each other's optimiser state.

Running it
----------
The four helpers at the bottom are stubs, so this file runs on a CPU in seconds and exercises the
whole chain — transport, key, box, metric wiring, resume, budget accounting — before any accelerator
is involved. Replace them with real training and real evaluation.

    export HYPER2_SERVER=http://<host>:8077      # the endpoint
    export HYPER2_API_KEY=<api key>              # issued with it; every request carries it
    export HYPER2_PROBLEM=<problem id>           # the problem registered for this run
    export HYPER2_SEED=0                         # which repetition
    python examples/driver.py
"""
import os

import hyper2

# ---- where to connect, and as whom ---------------------------------------------------------
# Three values identify the endpoint and the run. They are read from the environment here so that
# no address or credential is written into the file; assigning them literally works just as well.
# `os.environ[...]` rather than `.get(...)`: a missing variable then fails immediately, naming
# itself, instead of turning into an obscure rejection several calls later.
SERVER  = os.environ["HYPER2_SERVER"]      # e.g. "http://10.1.2.3:8077"
API_KEY = os.environ["HYPER2_API_KEY"]     # issued with the endpoint; required by it
PROBLEM = os.environ["HYPER2_PROBLEM"]     # the problem id registered for this engagement
SEED    = int(os.environ["HYPER2_SEED"])   # which repetition this is: 0, 1, 2, ...

# `connect` also falls back to $HYPER2_SERVER and $HYPER2_API_KEY when `server`/`api_key` are
# omitted, so both spellings are equivalent. They are passed explicitly below to keep it visible
# where the credential is used.

# ---- the run -------------------------------------------------------------------------------
TOKENS_PER_RUN = 25_400_000          # cost of ONE complete training run, in the caller's own unit

# The arguments below describe the task, its cost and its acceptance bar. None of them changes how
# the search works; everything that does is registered server-side against the problem id.
study = hyper2.connect(
    PROBLEM,
    server=SERVER,
    api_key=API_KEY,
    seed=SEED,                       # one seed per repetition; a repetition is then reproducible
    tokens_per_run=TOKENS_PER_RUN,   # the unit every other quantity is expressed in
    budget=3 * TOKENS_PER_RUN,       # total for the whole search: screening AND the final train-out
    target=0.47,                     # acceptance threshold on the judged metric
    eval_at=(0.5, 1.0),              # where the judged metric is produced: halfway, and at the end.
                                     #   FRACTIONS of one complete run, and only fractions. A depth
                                     #   that is not already a checkpoint is ADDED as one, and
                                     #   study.description says so. Omitted, the registered
                                     #   schedule applies. study.ladder lists every checkpoint.
)


def objective(trial):
    # One suggest_* call per knob, and every knob: the server draws a whole configuration and names
    # all of it in the recommendation, so a knob left unread is a knob the training does not respond
    # to while the search still ranges over it. A session that misses one is refused.
    cfg = {
        "learning_rate":    trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "weight_decay":     trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "batch_size":       trial.suggest_int("batch_size", 4, 64, log=True),
        "warmup_frac":      trial.suggest_float("warmup_frac", 0.0, 0.15),
        "beta1":            trial.suggest_float("beta1", 0.8, 0.98),
        "beta2":            trial.suggest_float("beta2", 0.95, 0.999),
        "epsilon":          trial.suggest_float("epsilon", 1e-9, 1e-6, log=True),
        "dropout":          trial.suggest_float("dropout", 0.0, 0.15),
        "lora_r":           trial.suggest_int("lora_r", 8, 128, log=True),
        "lora_alpha_ratio": trial.suggest_float("lora_alpha_ratio", 0.5, 4.0),
        "use_rslora":       trial.suggest_int("use_rslora", 0, 1),
        "lora_plus_ratio":  trial.suggest_float("lora_plus_ratio", 1.0, 32.0, log=True),
        "epochs":           trial.suggest_int("epochs", 1, 4),
    }

    model = load_state(study.run_id, trial.number) or new_model(cfg)  # resume a prior session
    ce = None

    for ckpt, tokens_target, kind in trial.plan():
        if kind == "eval_at":
            continue

        tokens = train_to(model, tokens_target)      # cumulative target; resume, never restart
        ce = evaluate_cheap(model)                   # every checkpoint: this drives screening

        signals = {}
        if study.needs_judged_metric(ckpt):          # only where the schedule calls for it
            signals[study.target_key] = evaluate_judged(model)
        trial.report(ce, step=ckpt, tokens=tokens, **signals)

        # Persist BEFORE testing the instruction. Saving afterwards drops the final segment of every
        # paused session: the next session resumes from the older state and retrains that segment,
        # which is paid for twice while the ledger records it once.
        save_state(model, study.run_id, trial.number)

        if not trial.instruct():
            break

    if trial.instruct().discard:                     # written off: stored state may be released
        drop_state(study.run_id, trial.number)       # otherwise it is kept, for a later resume
    return ce


# --------------------------------------------------------------------------------------------
# Stubs. Replace with real training and evaluation. `train_to` returns the actual cumulative token
# count, which is what `tokens=` carries — a step count times a nominal batch size overshoots it.
import json
import math
import shutil
from pathlib import Path

RUNS = Path(os.environ.get("RUNS_DIR", "runs"))


def new_model(cfg):
    # The study is stamped in so `load_state` can tell this study's checkpoints from an earlier
    # study's. See the note there.
    return {"cfg": cfg, "tokens": 0.0, "study": study.study_id}


def state_dir(run_id, number):
    """One directory per (run, configuration). `run_id` is "<problem>:<seed>", so two repetitions
    of one problem cannot collide; a name built from the problem alone would have them overwrite
    each other. The colon is replaced because it is not portable in path names."""
    return RUNS / f"{run_id.replace(':', '-')}-t{number:03d}"


def load_state(run_id, number):
    """State from THIS study, or None.

    `run_id` is "<problem>:<seed>", which is stable across runs -- so a second run of one problem
    and seed lands in the same directories as the first. A new study draws DIFFERENT configurations
    for the same trial numbers, so resuming an earlier study's checkpoint here would train, and
    report readings for, a configuration the server never drew. It is caught by stamping the study
    into the state and ignoring anything written by another one: a stale directory then costs a
    retrain, never a wrong answer. (Found by running this file twice against one server; the stub
    trainer's cumulative-target assertion is what refused it.)
    """
    p = state_dir(run_id, number) / "state.json"
    if not p.is_file():
        return None
    saved = json.loads(p.read_text())
    return saved if saved.get("study") == study.study_id else None


def save_state(model, run_id, number):
    d = state_dir(run_id, number)
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(model))


def drop_state(run_id, number):
    shutil.rmtree(state_dir(run_id, number), ignore_errors=True)


def train_to(model, tokens_target):
    """Train to `tokens_target` CUMULATIVE tokens, continuing from whatever the model already holds.

    Raises on a target already reached: targets are cumulative, so treating one as an increment
    retrains a prefix and pays for it a second time.
    """
    if tokens_target <= model["tokens"]:
        raise AssertionError(
            f"target {tokens_target:,.0f} is not beyond the {model['tokens']:,.0f} already trained; "
            f"targets are cumulative")
    model["tokens"] = float(tokens_target)
    return model["tokens"]


def evaluate_cheap(model):
    """Inexpensive signal, computed at every checkpoint. Drives screening and promotion."""
    return 2.0 + abs(math.log10(model["cfg"]["learning_rate"]) + 4.0) - 0.05 * (model["tokens"] / 4e6)


def evaluate_judged(model):
    """The metric the acceptance target is set on. Computed only where the schedule calls for it,
    which is why that schedule is worth stating rather than evaluating at every checkpoint."""
    return 1.0 / evaluate_cheap(model)
# --------------------------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"{study.description}\n  budget {study.budget_cost:,.0f} tokens"
          f"   judged on {study.target_key!r}   evaluated at {study.eval_at}")
    study.optimize(objective)
    best = study.recommendation["params"]        # the configuration the engine recommends
    print(f"\nspend {study.spent_cost:,.0f} of {study.budget_cost:,.0f} tokens "
          f"over {len(study.trials)} sessions")
    print("recommended:", ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in list(best.items())[:5]), "...")
