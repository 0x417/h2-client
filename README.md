# hyper2-client

The client half of a hyper2 optimisation run. **Standard library only** — no optimiser, no numpy,
no torch, nothing pinned, so it cannot collide with the versions a training stack fixes.

Everything that decides anything — the search box, the budget, the ladder, the evaluation schedule,
the acceptance policy, the transfer prior, every screening lever — lives on the server and is
registered against a problem id before a run starts. The client names the problem, trains, and reports.

## Install

```bash
pip install "hyper2-client @ git+https://github.com/0x417/h2-client.git"
python -c "import hyper2; print(hyper2.__version__)"
```

No credential is needed. If a machine has no outbound GitHub access, a downloaded copy installs
the same way:

```bash
pip install ./h2-client
```

Python 3.10+; verified on 3.12 against a 3.14 server. If installation is inconvenient,
`src/hyper2/wire.py` is one file with no dependencies and can be vendored as-is.

## Connect

Three values identify the endpoint and the run:

```bash
export HYPER2_SERVER=http://<host>:8077   # the endpoint
export HYPER2_API_KEY=<api key>           # issued with it; every request carries it
export HYPER2_PROBLEM=<problem id>        # the problem registered for this run
export HYPER2_SEED=0                     # which repetition

python -m hyper2.wire --check             # reaches the endpoint, checks the key, lists the problems
```

`connect(problem, server=..., api_key=...)` takes the first two directly and falls back to
`$HYPER2_SERVER` / `$HYPER2_API_KEY` when they are omitted; the two spellings are equivalent. The
examples pass them explicitly so it stays visible where the credential is used, and read them with
`os.environ[...]` rather than `.get(...)`, so a missing variable fails at once naming itself instead
of becoming an obscure rejection several calls later. A 401 means the key was missing or wrong: it
is reported immediately and never retried.

## The driver

This listing is the opening of `examples/driver.py`, quoted from the file itself rather than
restated, so it cannot drift from the code that is actually run:

```python
import hyper2

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

if __name__ == "__main__":
    print(f"{study.description}\n  budget {study.budget_cost:,.0f} tokens"
          f"   judged on {study.target_key!r}   evaluated at {study.eval_at}")
    study.optimize(objective)
    best = study.recommendation["params"]        # the configuration the engine recommends
```

## Eight things that matter

1. **`trial.plan()` yields THREE values** — `(checkpoint, cumulative_target, kind)`. Branch on
   `kind`; do not discard it.
2. **`target_tokens` is CUMULATIVE.** A paused-and-resumed trial gets a larger target, not another
   slice. Restarting from scratch pays the prefix again.
3. **Report the judged signal under `study.target_key`, where `study.needs_judged_metric(ckpt)`
   says it is needed.** The cheap signal goes in at *every* checkpoint — it drives screening. The judged metric is only needed at the steps the
   server names, which is the registered schedule (mid and cap, ~14% overhead); paying for it at every
   checkpoint multiplies that cost, and the extra spend is not in the ledger. A study that never
   receives it at all has nothing to measure: it screens one cohort, stops, and the client raises
   rather than handing back a result that looks fine.
4. **`trial.instruct()` is the explicit answer to "what now".** Falsy means the session is over;
   `.discard` then says whether to free the directory or keep it because the config will resume.
   **Freeing storage on every falsy answer destroys every paused config's checkpoint.**
5. **Pass `tokens=`** — the cumulative actual count, on the same axis as `target_tokens`. It books
   the real spend instead of a nominal step count.
6. **Resume:** `start = trial.last_step or 0`, state directory by `(study.run_id, trial.number)`.
7. **Two budget axes:** `study.spent_cost` / `study.budget_cost` are in the caller's unit (tokens);
   `study.spent` is in full-run equivalents. They differ by design — `spent_cost` includes
   evaluation cost.
8. **Seeds:** `connect(..., seed=N)` makes a repetition reproducible; use one seed per repetition.

## What a caller may set

`connect(problem, seed=..., target=..., eval_at=..., budget=..., tokens_per_run=..., server=..., api_key=...)`
— and nothing else. The five below describe the task; none of them is a search lever:

| | |
|---|---|
| `tokens_per_run` | what **one complete training run** costs, in the caller's unit. Only the calling stack can measure it, and it sets the unit for everything below — the budget, the checkpoint targets `plan()` yields, the train-out reserved for the winner. |
| `budget` | total tokens for the **whole search**: screening *and* training the recommendation out to full depth. The figure actually paid. |
| `target` | the acceptance threshold — the definition of an acceptable result. |
| `eval_at` | **where the judged metric is produced**, as fractions of one complete run — `(0.5, 1.0)` is halfway and at the end. **Fractions only**; a token count is refused by name, because the same count is a different depth for every caller and would move silently whenever `tokens_per_run` was re-measured. A caller-side compute cost: the judged metric is usually far dearer than the cheap signal that drives screening. A depth that is not already a checkpoint is **added** as one — a judged reading can only be taken where training pauses — and `study.description` records that the ladder was extended. Nothing is ever moved to a nearby depth silently. `study.ladder` lists every checkpoint in force. Omit it for the registered default. Read back as `study.eval_at` (fractions) or `study.eval_steps` (checkpoint indices, which is what `plan()` yields to the loop). |
| `seed` | makes a repetition reproducible; one seed per repetition. `study.run_id` combines it with the problem name as `"<problem>:<seed>"`, which is what stored state should be named after. **Key stored state by `(study.run_id, trial.number)`** — `trial.number` alone collides across repetitions. If omitted, the server fixes one and reports it as `study.seed`, so a run is never both irreproducible and unidentifiable. |

```python
TOKENS_PER_RUN = 25_400_000
study = hyper2.connect(PROBLEM, seed=0,
                       tokens_per_run=TOKENS_PER_RUN,
                       budget=3 * TOKENS_PER_RUN,
                       target=0.47, eval_steps=(3, 4))
```

Some problem ids carry **no** budget and require `budget=` and `tokens_per_run=`; others have one
registered already. `GET /v1/problems` reports which is which; requesting an open-budget problem without them is
refused immediately, naming what is missing.

`server` and `api_key` locate the endpoint, falling back to `$HYPER2_SERVER` / `$HYPER2_API_KEY`.
Any other setting is **refused by name**. That is what keeps a delivered run identical to the
configuration the published benchmark measured.

## Examples, in the order to read them

```bash
python examples/driver.py                 # START HERE -- a minimal run, end to end
python examples/integration_test.py       # plan -> report -> instruct -> resume, on CPU, seconds
python examples/smoke_client.py           # opens a study and prints the recommendation
```

**`examples/driver.py` is the one to read.** A complete driver in about forty lines: connect,
suggest every knob, train to each cumulative target, report, act on the instruction, persist. The
four points that are easy to get wrong are marked in the file:

1. `hyper2.connect("<problem id>")` replaces `create_study(...)` — every lever is registered
   server-side, so `seed` and `target` are the only arguments left.
2. `plan()` yields **three** values, `(ckpt, tokens_target, kind)`. Two raises on the first
   iteration.
3. The judged metric rides in as a keyword signal — `**{study.target_key: score}` — next to the CE,
   and `tokens=` books the actual cumulative spend.
4. `trial.instruct()` is the stop condition, and `.discard` answers the storage question: delete the
   saved model, or keep it because the config will be resumed.

The four helpers at the bottom are stubs so the file runs on a CPU in seconds — replace them with
the calling stack’s own training and evaluation. `train_to` raises if handed a target it has
already passed, because cumulative targets treated as increments pay for the same prefix twice.

The integration test asserts each link — that a session really resumes from a checkpoint on disk,
that cumulative targets are never retrained, that storage is freed only on `.discard`, and that a
seed reproduces its configurations. Exit code 1 on any failure.
