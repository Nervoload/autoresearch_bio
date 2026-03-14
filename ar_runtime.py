from __future__ import annotations

import base64
import copy
import datetime as dt
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
PROFILES_DIR = PROJECT_ROOT / "profiles"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
SERVICE_DIR = RESULTS_DIR / "service"
LAUNCHD_DIR = PROJECT_ROOT / "launchd"
CACHE_ROOT = Path.home() / ".cache" / "autoresearch"
DATASETS_ROOT = CACHE_ROOT / "datasets"
TOKENIZERS_ROOT = CACHE_ROOT / "tokenizers"
REMOTE_ENV_FILE = PROJECT_ROOT / ".ar-remote.env"
LAUNCHD_LABEL = "com.autoresearch.worker"

DEFAULT_SAMPLE_PROMPTS = [
    "Once upon a time",
    "The stock market opened and",
    "The doctor looked at the patient and said",
]

TERMINAL_STATES = {
    "completed",
    "failed",
    "memory_guard",
    "thermal_guard",
    "stopped",
}

RESUMABLE_STATES = {
    "failed",
    "memory_guard",
    "thermal_guard",
    "stopped",
    "running",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_name = handle.name
    os.replace(tmp_name, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8"
    ) as handle:
        handle.write(text)
        tmp_name = handle.name
    os.replace(tmp_name, path)


def profile_path(profile_id: str) -> Path:
    return PROFILES_DIR / f"{profile_id}.json"


def load_profile(profile_id: str, mode_override: str | None = None) -> dict[str, Any]:
    path = profile_path(profile_id)
    if not path.exists():
        raise FileNotFoundError(f"Unknown profile: {profile_id}")
    profile = read_json(path)
    if profile is None:
        raise RuntimeError(f"Could not load profile JSON from {path}")
    profile = copy.deepcopy(profile)
    if "profile_id" not in profile:
        profile["profile_id"] = profile_id
    if mode_override is not None:
        profile["mode"] = mode_override
    profile.setdefault("sample", {})
    profile["sample"].setdefault("prompts", DEFAULT_SAMPLE_PROMPTS)
    profile["sample"].setdefault("max_new_tokens", 96)
    profile["sample"].setdefault("temperature", 0.8)
    profile["sample"].setdefault("top_k", 50)
    profile.setdefault("checkpoint", {})
    profile["checkpoint"].setdefault("interval_s", 900)
    profile["checkpoint"].setdefault("keep_last", 2)
    profile.setdefault("mps", {})
    profile["mps"].setdefault("memory_fraction", 0.7)
    profile["mps"].setdefault("warn_ratio", 0.85)
    profile["mps"].setdefault("abort_ratio", 0.95)
    profile["mps"].setdefault("abort_samples", 3)
    profile.setdefault("mode", "search")
    return profile


def dataset_cache_dir(profile: dict[str, Any]) -> Path:
    return ensure_dir(DATASETS_ROOT / profile["dataset_id"])


def tokenizer_cache_dir(profile: dict[str, Any]) -> Path:
    return ensure_dir(TOKENIZERS_ROOT / profile["tokenizer_id"])


def make_run_id(profile_id: str, mode: str) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{profile_id}-{mode}-{uuid.uuid4().hex[:6]}"


def make_run_dir(run_id: str) -> Path:
    return ensure_dir(RUNS_DIR / run_id)


def service_status_path() -> Path:
    return SERVICE_DIR / "status.json"


def service_command_path() -> Path:
    return SERVICE_DIR / "command.json"


def service_log_path() -> Path:
    return SERVICE_DIR / "worker.stdout.log"


def service_error_log_path() -> Path:
    return SERVICE_DIR / "worker.stderr.log"


def load_source_metadata_b64(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    decoded = base64.b64decode(payload.encode("utf-8")).decode("utf-8")
    return json.loads(decoded)


def path_str(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def run_manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def run_metrics_path(run_dir: Path) -> Path:
    return run_dir / "metrics.json"


def run_status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def run_config_path(run_dir: Path) -> Path:
    return run_dir / "config.json"


def run_system_path(run_dir: Path) -> Path:
    return run_dir / "system.json"


def run_summary_path(run_dir: Path) -> Path:
    return run_dir / "summary.txt"


def run_train_log_path(run_dir: Path) -> Path:
    return run_dir / "train.log"


def run_samples_dir(run_dir: Path) -> Path:
    return ensure_dir(run_dir / "samples")


def run_checkpoints_dir(run_dir: Path) -> Path:
    return ensure_dir(run_dir / "checkpoints")


def latest_checkpoint_path(run_dir: Path) -> Path:
    return run_checkpoints_dir(run_dir) / "latest.pt"


def final_checkpoint_path(run_dir: Path) -> Path:
    return run_checkpoints_dir(run_dir) / "final.pt"


def checkpoint_path(run_dir: Path, tag: str) -> Path:
    return run_checkpoints_dir(run_dir) / f"{tag}.pt"


def run_sample_path(run_dir: Path, name: str = "final.txt") -> Path:
    return run_samples_dir(run_dir) / name


def prune_checkpoints(run_dir: Path, keep_last: int) -> list[Path]:
    checkpoints_dir = run_checkpoints_dir(run_dir)
    keep_last = max(0, int(keep_last))
    candidates = [
        path
        for path in checkpoints_dir.glob("*.pt")
        if path.name not in {"latest.pt", "final.pt"}
    ]
    candidates.sort(key=lambda path: (path.stat().st_mtime, path.name))
    to_remove = candidates[:-keep_last] if keep_last else candidates
    for path in to_remove:
        path.unlink(missing_ok=True)
    return to_remove


def memory_bytes_to_mb(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value) / 1024.0 / 1024.0


def _run_command(*args: str) -> str | None:
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _parse_darwin_vm_stat(raw: str) -> dict[str, float | None]:
    page_size_match = re.search(r"page size of (\d+) bytes", raw)
    if not page_size_match:
        return {}
    page_size = int(page_size_match.group(1))
    pages: dict[str, int] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        cleaned = value.strip().rstrip(".").replace(".", "").replace(",", "")
        if not cleaned.isdigit():
            continue
        pages[key.strip()] = int(cleaned)
    free_pages = pages.get("Pages free", 0) + pages.get("Pages speculative", 0)
    used_pages = (
        pages.get("Pages active", 0)
        + pages.get("Pages inactive", 0)
        + pages.get("Pages wired down", 0)
        + pages.get("Pages occupied by compressor", 0)
    )
    return {
        "memory_free_mb": memory_bytes_to_mb(free_pages * page_size),
        "memory_used_mb": memory_bytes_to_mb(used_pages * page_size),
    }


def host_telemetry_snapshot(pid: int | None = None) -> dict[str, Any]:
    telemetry: dict[str, Any] = {
        "load_1m": None,
        "load_5m": None,
        "load_15m": None,
        "process_rss_mb": None,
        "process_vsz_mb": None,
        "memory_total_mb": None,
        "memory_used_mb": None,
        "memory_free_mb": None,
        "memory_available_mb": None,
        "swap_used_mb": None,
    }
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
        telemetry["load_1m"] = round(load_1m, 2)
        telemetry["load_5m"] = round(load_5m, 2)
        telemetry["load_15m"] = round(load_15m, 2)
    except (AttributeError, OSError):
        pass

    target_pid = pid or os.getpid()
    ps_out = _run_command("ps", "-o", "rss=,vsz=", "-p", str(target_pid))
    if ps_out:
        parts = ps_out.split()
        if len(parts) >= 2 and all(part.isdigit() for part in parts[:2]):
            telemetry["process_rss_mb"] = round(int(parts[0]) / 1024.0, 1)
            telemetry["process_vsz_mb"] = round(int(parts[1]) / 1024.0, 1)

    total_mb = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        total_mb = memory_bytes_to_mb(page_size * phys_pages)
        free_mb = memory_bytes_to_mb(page_size * avail_pages)
        telemetry["memory_total_mb"] = round(total_mb, 1) if total_mb is not None else None
        telemetry["memory_available_mb"] = round(free_mb, 1) if free_mb is not None else None
        if total_mb is not None and free_mb is not None:
            telemetry["memory_used_mb"] = round(total_mb - free_mb, 1)
    except (AttributeError, OSError, ValueError):
        pass

    if sys.platform == "darwin":
        sysctl_mem = _run_command("sysctl", "-n", "hw.memsize")
        if sysctl_mem and sysctl_mem.isdigit():
            total_mb = memory_bytes_to_mb(int(sysctl_mem))
            telemetry["memory_total_mb"] = round(total_mb, 1) if total_mb is not None else None
        vm_stat = _run_command("vm_stat")
        if vm_stat:
            parsed = _parse_darwin_vm_stat(vm_stat)
            for key, value in parsed.items():
                telemetry[key] = round(value, 1) if value is not None else None
            if telemetry["memory_total_mb"] is not None and telemetry["memory_free_mb"] is not None:
                telemetry["memory_available_mb"] = telemetry["memory_free_mb"]
            if telemetry["memory_total_mb"] is not None and telemetry["memory_used_mb"] is not None:
                telemetry["memory_used_mb"] = min(
                    telemetry["memory_used_mb"],
                    telemetry["memory_total_mb"],
                )
        swap_out = _run_command("sysctl", "-n", "vm.swapusage")
        if swap_out:
            match = re.search(r"used = ([0-9.]+)([MG])", swap_out)
            if match:
                value = float(match.group(1))
                if match.group(2) == "G":
                    value *= 1024.0
                telemetry["swap_used_mb"] = round(value, 1)
    return telemetry


def get_thermal_state() -> str:
    override = os.environ.get("AR_THERMAL_STATE_OVERRIDE")
    if override:
        return override
    if sys.platform != "darwin":
        return "unsupported"
    cmd = (
        "osascript -l JavaScript -e "
        "'ObjC.import(\"Foundation\"); console.log($.NSProcessInfo.processInfo.thermalState)'"
    )
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return "unknown"
    raw = result.stdout.strip().splitlines()
    if not raw:
        return "unknown"
    mapping = {
        "0": "nominal",
        "1": "fair",
        "2": "serious",
        "3": "critical",
    }
    return mapping.get(raw[-1].strip(), "unknown")


def detect_device_type() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mps_memory_snapshot() -> dict[str, Any]:
    snapshot = {
        "current_allocated": None,
        "driver_allocated": None,
        "recommended_max": None,
    }
    try:
        import torch
    except ModuleNotFoundError:
        return snapshot
    if not hasattr(torch, "mps"):
        return snapshot
    try:
        if hasattr(torch.mps, "current_allocated_memory"):
            snapshot["current_allocated"] = int(torch.mps.current_allocated_memory())
        if hasattr(torch.mps, "driver_allocated_memory"):
            snapshot["driver_allocated"] = int(torch.mps.driver_allocated_memory())
        if hasattr(torch.mps, "recommended_max_memory"):
            snapshot["recommended_max"] = int(torch.mps.recommended_max_memory())
    except RuntimeError:
        return snapshot
    return snapshot


def apply_mps_memory_fraction(fraction: float) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return
    if hasattr(torch, "mps") and hasattr(torch.mps, "set_per_process_memory_fraction"):
        torch.mps.set_per_process_memory_fraction(float(fraction))


def system_info(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "profile_id": profile.get("profile_id") if profile else None,
        "dataset_id": profile.get("dataset_id") if profile else None,
        "tokenizer_id": profile.get("tokenizer_id") if profile else None,
        "device_type": detect_device_type(),
        "thermal_state": get_thermal_state(),
        "collected_at": utcnow(),
    }
    info["host"] = host_telemetry_snapshot()
    if info["device_type"] == "mps":
        info["mps"] = mps_memory_snapshot()
    return info


def summary_text(metrics: dict[str, Any]) -> str:
    lines = [
        f"run_id:            {metrics.get('run_id', 'n/a')}",
        f"profile_id:        {metrics.get('profile_id', 'n/a')}",
        f"mode:              {metrics.get('mode', 'n/a')}",
        f"state:             {metrics.get('state', 'n/a')}",
        f"val_bpb:           {metrics.get('val_bpb', 'n/a')}",
        f"training_seconds:  {metrics.get('training_seconds', 'n/a')}",
        f"total_seconds:     {metrics.get('total_seconds', 'n/a')}",
        f"peak_memory_mb:    {metrics.get('peak_memory_mb', 'n/a')}",
        f"tok_per_sec:       {metrics.get('tok_per_sec', 'n/a')}",
        f"total_tokens_M:    {metrics.get('total_tokens_M', 'n/a')}",
        f"num_steps:         {metrics.get('num_steps', 'n/a')}",
        f"num_params_M:      {metrics.get('num_params_M', 'n/a')}",
        f"thermal_state:     {metrics.get('thermal_state', 'n/a')}",
    ]
    if metrics.get("peak_process_rss_mb") is not None:
        lines.append(f"peak_process_rss_mb: {metrics['peak_process_rss_mb']}")
    if metrics.get("peak_system_memory_used_mb") is not None:
        lines.append(f"peak_system_mem_mb: {metrics['peak_system_memory_used_mb']}")
    if metrics.get("load_1m") is not None:
        lines.append(f"load_1m:           {metrics['load_1m']}")
    if metrics.get("mfu_percent") is not None:
        lines.append(f"mfu_percent:       {metrics['mfu_percent']}")
    return "\n".join(lines) + "\n"


def run_record(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_manifest_path(run_dir), {})
    metrics = read_json(run_metrics_path(run_dir), {})
    status = read_json(run_status_path(run_dir), {})
    record = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "profile_id": manifest.get("profile_id"),
        "mode": manifest.get("mode"),
        "state": status.get("state", manifest.get("state")),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "val_bpb": metrics.get("val_bpb"),
        "tok_per_sec": metrics.get("tok_per_sec"),
        "train_log": str(run_train_log_path(run_dir)),
        "sample_path": str(run_sample_path(run_dir)),
        "manifest": manifest,
        "metrics": metrics,
        "status": status,
    }
    return record


def list_run_dirs() -> list[Path]:
    ensure_dir(RUNS_DIR)
    return sorted(
        [path for path in RUNS_DIR.iterdir() if path.is_dir()],
        key=lambda item: item.name,
    )


def list_run_records() -> list[dict[str, Any]]:
    return [run_record(path) for path in list_run_dirs()]


def latest_run_record(
    profile_id: str | None = None,
    mode: str | None = None,
    include_states: set[str] | None = None,
) -> dict[str, Any] | None:
    records = list_run_records()
    filtered: list[dict[str, Any]] = []
    for record in records:
        if profile_id and record.get("profile_id") != profile_id:
            continue
        if mode and record.get("mode") != mode:
            continue
        if include_states and record.get("state") not in include_states:
            continue
        filtered.append(record)
    return filtered[-1] if filtered else None


def render_status(status: dict[str, Any]) -> str:
    if not status:
        return "No service status found.\n"
    ordered = [
        ("state", status.get("state")),
        ("run_id", status.get("run_id")),
        ("profile_id", status.get("profile_id")),
        ("mode", status.get("mode")),
        ("heartbeat_at", status.get("heartbeat_at")),
        ("step", status.get("step")),
        ("tokens_total", status.get("tokens_total")),
        ("train_seconds", status.get("train_seconds")),
        ("tok_per_sec", status.get("tok_per_sec")),
        ("thermal_state", status.get("thermal_state")),
        ("last_checkpoint", status.get("last_checkpoint")),
        ("last_error", status.get("last_error")),
    ]
    lines = [f"{key}: {value}" for key, value in ordered if value is not None]
    memory = status.get("memory") or {}
    if memory:
        for key, value in memory.items():
            lines.append(f"memory.{key}: {value}")
    host = status.get("host") or {}
    if host:
        for key, value in host.items():
            if value is not None:
                lines.append(f"host.{key}: {value}")
    return "\n".join(lines) + "\n"


def stale_status(status: dict[str, Any], threshold_s: int = 30) -> bool:
    heartbeat_at = status.get("heartbeat_at")
    if not heartbeat_at:
        return False
    try:
        heartbeat = dt.datetime.fromisoformat(heartbeat_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    age = dt.datetime.now(dt.timezone.utc) - heartbeat
    return age.total_seconds() > threshold_s


def normalize_run_state(state: str | None) -> str:
    if not state:
        return "idle"
    return state
