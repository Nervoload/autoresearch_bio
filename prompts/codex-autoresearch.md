# Codex Autoresearch Prompt

Read `README.md` and `program.md` first.

We are using this repo as a small autonomous research machine. Your job is to be the external research agent, not just a one-shot coding assistant. The "auto" part lives in your outer loop: you decide what to change, run the experiment, inspect the artifacts, and choose the next step.

Rules:

- Default profile: `tinystories_8gb_search`
- Default mode: `search`
- Use `bin/ar-local` if you are already on the training Mac over SSH
- Use `bin/ar` only when controlling a different Mac remotely from another machine
- Start with a baseline run before changing code
- Prefer small, safe experiments in `train.py` or `profiles/*.json`
- Keep dataset/tokenizer identity fixed while comparing architecture changes
- Treat `results/runs/<run_id>/manifest.json`, `metrics.json`, `status.json`, `summary.txt`, and `samples/final.txt` as the source of truth
- Respect the MPS memory and thermal guardrails
- Keep the repo small; avoid framework creep

Loop:

1. Run the current baseline and inspect the latest run artifacts.
2. Propose one small improvement at a time.
3. Implement the change.
4. Run a new experiment.
5. Compare `val_bpb`, stability, and sample quality.
6. Keep only changes that clearly help or simplify the system without hurting reliability.

Good first command sequence on the training Mac:

```bash
bin/ar-local bootstrap
bin/ar-local prepare --profile tinystories_8gb_search
bin/ar-local start search --profile tinystories_8gb_search
bin/ar-local logs --follow
```

When summarizing a run, include:

- profile id
- final state
- `val_bpb`
- tokens/sec
- peak memory
- thermal state
- whether the sample quality improved, regressed, or was inconclusive
