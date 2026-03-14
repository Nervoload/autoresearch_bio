from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ar_runtime import (
    PROJECT_ROOT,
    RESUMABLE_STATES,
    SERVICE_DIR,
    TERMINAL_STATES,
    ensure_dir,
    latest_checkpoint_path,
    latest_run_record,
    make_run_dir,
    make_run_id,
    normalize_run_state,
    read_json,
    run_status_path,
    run_train_log_path,
    service_command_path,
    service_status_path,
    stale_status,
    utcnow,
    write_json,
)


POLL_INTERVAL_S = 2.0


class Worker:
    def __init__(self) -> None:
        ensure_dir(SERVICE_DIR)
        self.current: dict[str, object] | None = None

    def write_service_status(self, payload: dict[str, object]) -> None:
        payload["heartbeat_at"] = utcnow()
        write_json(service_status_path(), payload)

    def idle(self, last_error: str | None = None) -> None:
        payload = {
            "state": "idle",
            "run_id": None,
            "profile_id": None,
            "mode": None,
            "step": None,
            "tokens_total": None,
            "train_seconds": None,
            "tok_per_sec": None,
            "memory": {},
            "host": {},
            "thermal_state": None,
            "last_checkpoint": None,
            "last_error": last_error,
            "train_log": None,
        }
        self.write_service_status(payload)

    def load_command(self) -> dict[str, object] | None:
        path = service_command_path()
        if not path.exists():
            return None
        payload = read_json(path, {})
        path.unlink(missing_ok=True)
        return payload

    def resumable_run(self, profile_id: str, mode: str) -> Path | None:
        if mode != "soak":
            return None
        record = latest_run_record(profile_id=profile_id, mode=mode, include_states=RESUMABLE_STATES)
        if record is None:
            return None
        run_dir = Path(record["run_dir"])
        if latest_checkpoint_path(run_dir).exists():
            return run_dir
        return None

    def start_run(self, command: dict[str, object]) -> None:
        if self.current and self.current["process"].poll() is None:
            self.write_service_status(
                {
                    "state": "failed",
                    "run_id": self.current["run_id"],
                    "profile_id": self.current["profile_id"],
                    "mode": self.current["mode"],
                    "last_error": "Cannot start a new run while another run is active.",
                    "train_log": str(self.current["train_log"]),
                }
            )
            return

        profile_id = str(command["profile_id"])
        mode = str(command["mode"])
        force_new = bool(command.get("force_new"))
        run_dir = None if force_new else self.resumable_run(profile_id, mode)
        resume = run_dir is not None
        if run_dir is None:
            run_id = make_run_id(profile_id, mode)
            run_dir = make_run_dir(run_id)
        else:
            run_id = run_dir.name

        source_metadata = command.get("source_metadata", {})
        if source_metadata:
            write_json(run_dir / "source_metadata.json", source_metadata)

        log_path = run_train_log_path(run_dir)
        log_handle = log_path.open("a", encoding="utf-8")
        args = [
            sys.executable,
            "train.py",
            "--profile",
            profile_id,
            "--mode",
            mode,
            "--run-dir",
            str(run_dir),
        ]
        if resume:
            args.append("--resume")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            args,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

        self.current = {
            "process": process,
            "run_id": run_id,
            "run_dir": run_dir,
            "profile_id": profile_id,
            "mode": mode,
            "train_log": log_path,
            "log_handle": log_handle,
        }
        self.write_service_status(
            {
                "state": "running",
                "run_id": run_id,
                "profile_id": profile_id,
                "mode": mode,
                "step": 0,
                "tokens_total": 0,
                "train_seconds": 0,
                "tok_per_sec": None,
                "memory": {},
                "host": {},
                "thermal_state": None,
                "last_checkpoint": str(latest_checkpoint_path(run_dir)) if resume else None,
                "last_error": None,
                "train_log": str(log_path),
            }
        )

    def stop_run(self) -> None:
        if not self.current:
            self.idle(last_error="No active run to stop.")
            return
        process = self.current["process"]
        if process.poll() is None:
            process.terminate()
            self.write_service_status(
                {
                    "state": "stopping",
                    "run_id": self.current["run_id"],
                    "profile_id": self.current["profile_id"],
                    "mode": self.current["mode"],
                    "last_error": None,
                    "train_log": str(self.current["train_log"]),
                }
            )

    def mirror_run_status(self) -> None:
        if not self.current:
            return
        process = self.current["process"]
        run_dir = self.current["run_dir"]
        run_status = read_json(run_status_path(run_dir), {})
        if process.poll() is None:
            state = normalize_run_state(run_status.get("state")) if run_status else "running"
            if state == "running" and stale_status(run_status):
                state = "stalled"
            payload = {
                "state": state,
                "run_id": self.current["run_id"],
                "profile_id": self.current["profile_id"],
                "mode": self.current["mode"],
                "step": run_status.get("step"),
                "tokens_total": run_status.get("tokens_total"),
                "train_seconds": run_status.get("train_seconds"),
                "tok_per_sec": run_status.get("tok_per_sec"),
                "memory": run_status.get("memory", {}),
                "host": run_status.get("host", {}),
                "thermal_state": run_status.get("thermal_state"),
                "last_checkpoint": run_status.get("last_checkpoint"),
                "last_error": run_status.get("last_error"),
                "train_log": str(self.current["train_log"]),
            }
            self.write_service_status(payload)
            return

        return_code = process.returncode
        state = run_status.get("state")
        if return_code != 0 and state not in TERMINAL_STATES:
            state = "failed"
        payload = {
            "state": state or ("completed" if return_code == 0 else "failed"),
            "run_id": self.current["run_id"],
            "profile_id": self.current["profile_id"],
            "mode": self.current["mode"],
            "step": run_status.get("step"),
            "tokens_total": run_status.get("tokens_total"),
            "train_seconds": run_status.get("train_seconds"),
            "tok_per_sec": run_status.get("tok_per_sec"),
            "memory": run_status.get("memory", {}),
            "host": run_status.get("host", {}),
            "thermal_state": run_status.get("thermal_state"),
            "last_checkpoint": run_status.get("last_checkpoint"),
            "last_error": run_status.get("last_error"),
            "train_log": str(self.current["train_log"]),
        }
        self.write_service_status(payload)
        self.current["log_handle"].close()
        self.current = None

    def run(self) -> None:
        self.idle()
        while True:
            self.mirror_run_status()
            command = self.load_command()
            if command:
                if command.get("action") == "start":
                    self.start_run(command)
                elif command.get("action") == "stop":
                    self.stop_run()
            if not self.current and not service_status_path().exists():
                self.idle()
            elif not self.current:
                status = read_json(service_status_path(), {})
                if status.get("state") in {"running", "stopping"}:
                    self.idle(last_error=status.get("last_error"))
                else:
                    status["heartbeat_at"] = utcnow()
                    write_json(service_status_path(), status)
            time.sleep(POLL_INTERVAL_S)


def main() -> int:
    worker = Worker()
    worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
