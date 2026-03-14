# autoresearch-macos

Autoresearch is a tiny autonomous LLM training loop for a single machine. This fork keeps the original spirit, but adds the minimum systems layer needed to treat an Apple-silicon Mac as a reliable remote training appliance.

## What Changed In Phase 1

- `search` mode keeps the fixed 5-minute `val_bpb` loop for fast architecture ranking.
- `soak` mode adds longer confirmation runs, periodic checkpoints, and resume from `latest.pt`.
- Every run now writes artifacts under `results/runs/<run_id>/`.
- MPS runs report device-relevant telemetry instead of H100-relative MFU theater.
- A resident `launchd` worker plus `bin/ar` let you control a remote Mac over SSH.

## Core Layout

The repo is still intentionally compact, but it is no longer just three files:

- `prepare.py` downloads a dataset and trains the tokenizer for a chosen profile.
- `train.py` runs a single training job, writes artifacts, checkpoints, samples, and status.
- `profiles/*.json` define datasets, tokenizers, model shape, optimizer settings, and guardrails.
- `ar_worker.py` is the background worker kept alive by `launchd`.
- `ar_remote.py` is the remote-side control CLI used by `bin/ar`.
- `bin/ar` is the Air-to-Pro remote wrapper.
- `bin/ar-local` is the Pro-local wrapper when you are already on the training machine over SSH.
- `program.md` tells an external coding agent how to operate inside the new profile-aware workflow.

## Profiles

Shipped profiles:

- `climbmix_legacy`
- `tinystories_8gb_search`
- `tinystories_8gb_soak`

The legacy profile preserves the current repo’s baseline training shape. The TinyStories profiles are tuned for an 8 GB M-series machine and are the recommended first end-to-end target for remote testing.

## Artifacts

Each run writes:

- `manifest.json`
- `config.json`
- `system.json`
- `metrics.json`
- `status.json`
- `train.log`
- `summary.txt`
- `samples/final.txt`
- `checkpoints/latest.pt`
- `checkpoints/final.pt`

`soak` runs also emit periodic checkpoints and intermediate sample files.

## Local Quick Start

```bash
uv sync
uv run prepare.py --profile tinystories_8gb_search
uv run train.py --profile tinystories_8gb_search
```

## Training Mac Setup

If you are cloning this repo directly onto the Apple-silicon Mac that will do the training, this is the shortest setup path:

```bash
git clone <your-repo-url> ~/autoresearch-macos
cd ~/autoresearch-macos
bin/bootstrap-training-mac --follow
```

The bootstrap helper installs `uv` if needed, syncs dependencies, installs the local `launchd` worker, prepares the default TinyStories profile, starts the first `search` run, and optionally tails logs.

Manual equivalent:

```bash
git clone <your-repo-url> ~/autoresearch-macos
cd ~/autoresearch-macos
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
bin/ar-local bootstrap
bin/ar-local prepare --profile tinystories_8gb_search
bin/ar-local start search --profile tinystories_8gb_search
bin/ar-local logs --follow
```

If `uv` is already installed, skip the installer line.

## Remote Quick Start

1. Copy `.ar-remote.env.example` to `.ar-remote.env`.
2. Set:

```bash
AR_REMOTE_HOST="your-home-mac"
AR_REMOTE_ROOT="/Users/yourname/autoresearch-macos"
```

3. Bootstrap the remote worker:

```bash
bin/ar bootstrap
```

4. Prepare data on the remote Mac:

```bash
bin/ar prepare --profile tinystories_8gb_search
```

5. Start a run:

```bash
bin/ar start search --profile tinystories_8gb_search
```

6. Inspect it:

```bash
bin/ar status
bin/ar logs --follow
bin/ar runs latest
bin/ar sample latest
```

## Pro-Local Quick Start

If you are already on the training Mac over SSH, you do not need the remote env file or the Air wrapper. Use the local wrapper instead:

```bash
bin/ar-local bootstrap
bin/ar-local prepare --profile tinystories_8gb_search
bin/ar-local start search --profile tinystories_8gb_search
bin/ar-local status
bin/ar-local logs --follow
bin/ar-local runs latest
bin/ar-local sample latest
```

This is the recommended interface when Codex is running directly in the repo on the training machine.

## What Is "Auto" Here?

The training loop in this repo runs one experiment at a time. The "auto" part is the outer research loop:

- an external agent or human changes `train.py` or a profile
- the repo runs a controlled experiment
- artifacts are written under `results/runs/<run_id>/`
- the external agent decides what to try next

Codex can be that external agent. The prompt template in `prompts/codex-autoresearch.md` is the recommended starting point when you want Codex to operate the repo this way.

## MPS Guardrails

On Apple silicon, the training loop:

- applies `torch.mps.set_per_process_memory_fraction`
- tracks current and driver-allocated MPS memory
- warns near the configured cap
- checkpoints and aborts on sustained allocator pressure
- checks macOS thermal state and stops early when the machine enters `serious` or `critical`

Default 8 GB policy is configured in the TinyStories profiles:

- memory fraction: `0.70`
- warn threshold: `85%` of the cap
- abort threshold: `95%` of the cap for `3` samples

## Design Notes

- The repo is still single-machine and intentionally light on dependencies.
- `val_bpb` remains the inner-loop scalar for `search`.
- Dataset, tokenizer, and model settings are now profile-controlled.
- Depth and width are decoupled for new profiles, while legacy scaling remains available for exact baseline reproduction.
- The external coding agent is still expected to operate on top of this repo, but it now does so through a safer run/artifact model.
- A ready-to-paste Codex starter prompt lives in `prompts/codex-autoresearch.md`.

## Status Vocabulary

The worker reports:

- `idle`
- `running`
- `stalled`
- `memory_guard`
- `thermal_guard`
- `failed`

## Deferrals

Phase 1 intentionally does not include:

- `lm-evaluation-harness`
- MLX-LM interoperability
- LoRA/adapters
- large dataset/tokenizer registries
- a fully autonomous code-mutation loop

Those are follow-up phases. Phase 1 is about making the research machine safe, observable, and remotely usable.
