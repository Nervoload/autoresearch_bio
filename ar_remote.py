from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from ar_runtime import (
    LAUNCHD_DIR,
    LAUNCHD_LABEL,
    PROJECT_ROOT,
    SERVICE_DIR,
    TERMINAL_STATES,
    ensure_dir,
    latest_run_record,
    load_source_metadata_b64,
    read_json,
    render_status,
    run_sample_path,
    service_command_path,
    service_error_log_path,
    service_log_path,
    service_status_path,
    stale_status,
    utcnow,
    write_json,
    write_text,
)


def launchctl_domains() -> list[str]:
    uid = os.getuid()
    return [f"gui/{uid}", f"user/{uid}"]


def render_launchagent(repo_root: Path) -> str:
    template_path = LAUNCHD_DIR / "com.autoresearch.worker.plist.template"
    template = template_path.read_text(encoding="utf-8")
    values = {
        "__LABEL__": LAUNCHD_LABEL,
        "__REPO_ROOT__": str(repo_root),
        "__STDOUT__": str(service_log_path()),
        "__STDERR__": str(service_error_log_path()),
    }
    for key, value in values.items():
        template = template.replace(key, value)
    return template


def install_launchagent(repo_root: Path, quiet: bool = False) -> int:
    ensure_dir(SERVICE_DIR)
    ensure_dir(Path.home() / "Library" / "LaunchAgents")
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    write_text(plist_path, render_launchagent(repo_root))

    active_domain = None
    for domain in launchctl_domains():
        subprocess.run(
            ["launchctl", "bootout", domain, str(plist_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        result = subprocess.run(
            ["launchctl", "bootstrap", domain, str(plist_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            active_domain = domain
            break

    if active_domain is None:
        if not quiet:
            print("Failed to bootstrap launch agent.", file=sys.stderr)
        return 1

    subprocess.run(
        ["launchctl", "enable", f"{active_domain}/{LAUNCHD_LABEL}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run(
        ["launchctl", "kickstart", "-k", f"{active_domain}/{LAUNCHD_LABEL}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if not quiet:
        print(f"Installed {LAUNCHD_LABEL} in {active_domain}")
        print(plist_path)
    return 0


def enqueue_command(payload: dict[str, object]) -> int:
    payload["command_id"] = uuid.uuid4().hex[:10]
    payload["requested_at"] = utcnow()
    write_json(service_command_path(), payload)
    print(json.dumps(payload, indent=2))
    return 0


def status_command(as_json: bool) -> int:
    status = read_json(service_status_path(), {})
    if status and status.get("state") == "running" and stale_status(status):
        status["state"] = "stalled"
    if as_json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(render_status(status), end="")
    return 0


def runs_command(selector: str, run_id: str | None) -> int:
    from ar_runtime import list_run_records

    if selector == "list":
        records = list_run_records()
        if not records:
            print("No runs found.")
            return 0
        for record in records:
            print(
                f"{record['run_id']}\t{record.get('state')}\t"
                f"{record.get('profile_id')}\t{record.get('mode')}\t"
                f"{record.get('val_bpb')}"
            )
        return 0

    if selector == "latest":
        record = latest_run_record()
    else:
        record = None
        for candidate in list_run_records():
            if candidate["run_id"] == run_id:
                record = candidate
                break

    if record is None:
        print("Run not found.", file=sys.stderr)
        return 1

    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def sample_command(selector: str) -> int:
    if selector != "latest":
        print("Only `sample latest` is supported.", file=sys.stderr)
        return 1
    record = latest_run_record()
    if record is None:
        print("No runs found.", file=sys.stderr)
        return 1
    path = Path(run_sample_path(Path(record["run_dir"])))
    if not path.exists():
        print(f"No sample found at {path}", file=sys.stderr)
        return 1
    print(path.read_text(encoding="utf-8"), end="")
    return 0


def logs_command(follow: bool, service_only: bool) -> int:
    status = read_json(service_status_path(), {})
    log_path = None
    if follow and not service_only:
        deadline = time.time() + 30
        while time.time() < deadline:
            status = read_json(service_status_path(), {})
            if status.get("train_log"):
                candidate = Path(status["train_log"])
                if candidate.exists():
                    log_path = candidate
                    break
            time.sleep(1)
    elif not service_only and status.get("train_log"):
        log_path = Path(status["train_log"])
    if log_path is None or not log_path.exists():
        if follow and not service_only:
            print("Training log is not ready yet; following worker service log instead.", file=sys.stderr)
        log_path = service_log_path()
    if not log_path.exists():
        print(f"No log file found at {log_path}", file=sys.stderr)
        return 1
    if follow:
        os.execvp("tail", ["tail", "-n", "50", "-f", str(log_path)])
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()[-50:]
    print("".join(lines), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote autoresearch control plane")
    subparsers = parser.add_subparsers(dest="command", required=True)

    install = subparsers.add_parser("install-launchagent")
    install.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    install.add_argument("--quiet", action="store_true")

    enqueue_start = subparsers.add_parser("enqueue-start")
    enqueue_start.add_argument("--profile", required=True)
    enqueue_start.add_argument("--mode", choices=["search", "soak"], required=True)
    enqueue_start.add_argument("--source-json-base64")
    enqueue_start.add_argument("--force-new", action="store_true")

    subparsers.add_parser("enqueue-stop")

    status = subparsers.add_parser("status")
    status.add_argument("--json", action="store_true")

    runs = subparsers.add_parser("runs")
    runs.add_argument("selector", choices=["list", "show", "latest"], nargs="?", default="list")
    runs.add_argument("run_id", nargs="?")

    sample = subparsers.add_parser("sample")
    sample.add_argument("selector", choices=["latest"])

    logs = subparsers.add_parser("logs")
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("--service", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "install-launchagent":
        return install_launchagent(args.repo_root, quiet=args.quiet)
    if args.command == "enqueue-start":
        source_metadata = load_source_metadata_b64(args.source_json_base64)
        payload = {
            "action": "start",
            "profile_id": args.profile,
            "mode": args.mode,
            "force_new": args.force_new,
            "source_metadata": source_metadata,
        }
        return enqueue_command(payload)
    if args.command == "enqueue-stop":
        return enqueue_command({"action": "stop"})
    if args.command == "status":
        return status_command(as_json=args.json)
    if args.command == "runs":
        if args.selector == "show" and not args.run_id:
            parser.error("runs show requires a run_id")
        return runs_command(args.selector, args.run_id)
    if args.command == "sample":
        return sample_command(args.selector)
    if args.command == "logs":
        return logs_command(follow=args.follow, service_only=args.service)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
