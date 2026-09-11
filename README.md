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

While this repository is private that URL needs a credential, so pick whichever of the three the
installing machine already has:

```bash
# a GitHub account with an SSH key on it (nothing to paste)
pip install "hyper2-client @ git+ssh://git@github.com/0x417/h2-client.git"

# a personal access token with read access to this repository
pip install "hyper2-client @ git+https://<token>@github.com/0x417/h2-client.git"

# no GitHub access at all: from a downloaded copy
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

```python
import shutil
import hyper2

study = hyper2.connect(PROBLEM, server=SERVER, api_key=API_KEY, seed=0)

def objective(trial):
    hp = {name: suggest(trial, name, spec) for name, spec in study.space.items()}
    out_dir = CKPT_ROOT / str(trial.number)            # checkpoint by trial.number
    start   = trial.last_step or 0                     # resume contract, one line

    ce = None
    for ckpt, target_tokens, kind in trial.plan():     # THREE values
        if kind == "eval_at":
            continue                                   # not reachable in this build; branch anyway
        # train THIS config up to `target_tokens` CUMULATIVE tokens (resume from `start`,
        # do not restart), then evaluate:
        ce, score, tokens_used = train_and_eval(hp, target_tokens, out_dir)
        trial.report(ce, step=ckpt, tokens=tokens_used, **{study.target_key: score})
        if not trial.instruct():                       # session over
            break

    if trial.instruct().discard:                       # written off -> storage can go
        shutil.rmtree(out_dir)                         # otherwise KEEP it: it will resume
    return ce

study.optimize(objective)                              # no n_trials; the plan is the stop condition
print(study.recommendation)                            # the configuration the engine recommends
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
| `eval_at` | **where the judged metric is produced**, as fractions of one complete run — `(0.5, 1.0)` is halfway and at the end — or as absolute token counts, `(12_700_000, 25_400_000)`. A caller-side compute cost: the judged metric is usually far dearer than the cheap signal that drives screening. A depth that is not already a checkpoint is **added** as one — a judged reading can only be taken where training pauses — and `study.description` records that the ladder was extended. Nothing is ever moved to a nearby depth silently. `study.ladder` lists every checkpoint in force. Omit it for the registered default. Read back as `study.eval_at` (fractions) or `study.eval_steps` (checkpoint indices, which is what `plan()` yields to the loop). |
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
