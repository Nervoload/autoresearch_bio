# autoresearch

This repo now has a profile-aware runtime, artifact model, and remote control plane. Work with it as a small research machine, not as a one-off script.

## Operating Model

- `prepare.py` handles dataset and tokenizer prep for a chosen profile.
- `train.py` runs one concrete experiment and writes artifacts into `results/runs/<run_id>/`.
- `bin/ar` is the preferred interface when operating a remote Apple-silicon worker over SSH.
- `bin/ar-local` is the preferred interface when you are already on the training Mac over SSH or running Codex there directly.
- `profiles/*.json` are now part of the research surface. They control dataset, tokenizer, model shape, optimizer settings, and MPS guardrails.
- The "autoresearch" loop lives in the external agent. This repo executes experiments and records evidence; the agent decides what to change next.

## Default Workflow

For a remote Mac:

1. `bin/ar bootstrap`
2. `bin/ar prepare --profile tinystories_8gb_search`
3. `bin/ar start search --profile tinystories_8gb_search`
4. `bin/ar status`
5. `bin/ar logs --follow`
6. `bin/ar runs latest`
7. `bin/ar sample latest`

For local manual testing:

1. `uv sync`
2. `uv run prepare.py --profile tinystories_8gb_search`
3. `uv run train.py --profile tinystories_8gb_search`

For direct operation on the training Mac over SSH:

1. `bin/ar-local bootstrap`
2. `bin/ar-local prepare --profile tinystories_8gb_search`
3. `bin/ar-local start search --profile tinystories_8gb_search`
4. `bin/ar-local status`
5. `bin/ar-local logs --follow`
6. `bin/ar-local runs latest`
7. `bin/ar-local sample latest`

If you are bringing up a freshly cloned training Mac, use this exact order:

1. `uv sync`
2. `bin/ar-local bootstrap`
3. `bin/ar-local prepare --profile tinystories_8gb_search`
4. `bin/ar-local start search --profile tinystories_8gb_search`
5. `bin/ar-local logs --follow`

## Research Rules

What you can change:

- `train.py`
- `profiles/*.json`
- remote control scripts and observability code
- documentation and analysis helpers

What should stay fixed unless the human explicitly asks otherwise:

- the `val_bpb` evaluator semantics for a given profile
- the dataset and tokenizer identity inside a profile while comparing architecture changes
- the artifact contract under `results/runs/<run_id>/`
- the MPS memory and thermal safety policy

## Acceptance Criteria For A Good Experiment

- It finishes without crashing or makes a deliberate guarded stop.
- It writes `manifest.json`, `metrics.json`, `status.json`, `summary.txt`, a final sample, and checkpoint artifacts.
- It records the profile id, source metadata, and final state.
- On MPS, it reports tokens/sec and allocator telemetry instead of pretending to be an H100.

## Simplicity Bias

Keep the repo small. Prefer a good profile or a small runtime helper over framework creep. A clean improvement that preserves the remote workflow and artifact model is better than a large clever change that makes the worker brittle.

## Codex Note

If Codex is the external research agent, prefer `bin/ar-local` when Codex is running on the training Mac itself. Use `bin/ar` only when Codex is controlling a different machine remotely from another checkout.
