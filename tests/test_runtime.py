import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ar_remote import render_launchagent
from ar_runtime import (
    PROJECT_ROOT,
    get_thermal_state,
    load_profile,
    prune_checkpoints,
    render_status,
    run_checkpoints_dir,
    summary_text,
)


class RuntimeTests(unittest.TestCase):
    def test_load_profile_applies_defaults(self):
        profile = load_profile("tinystories_8gb_search")
        self.assertEqual(profile["profile_id"], "tinystories_8gb_search")
        self.assertIn("sample", profile)
        self.assertIn("prompts", profile["sample"])
        self.assertEqual(profile["mps"]["memory_fraction"], 0.7)

    def test_thermal_override(self):
        import os

        os.environ["AR_THERMAL_STATE_OVERRIDE"] = "serious"
        try:
            self.assertEqual(get_thermal_state(), "serious")
        finally:
            os.environ.pop("AR_THERMAL_STATE_OVERRIDE", None)

    def test_render_status(self):
        rendered = render_status(
            {
                "state": "running",
                "run_id": "demo",
                "profile_id": "tinystories_8gb_search",
                "memory": {"current_allocated_mb": 100.0},
            }
        )
        self.assertIn("state: running", rendered)
        self.assertIn("memory.current_allocated_mb: 100.0", rendered)

    def test_summary_text(self):
        summary = summary_text(
            {
                "run_id": "demo",
                "profile_id": "tinystories_8gb_search",
                "mode": "search",
                "state": "completed",
            }
        )
        self.assertIn("run_id:            demo", summary)
        self.assertIn("state:             completed", summary)

    def test_launchagent_rendering(self):
        rendered = render_launchagent(PROJECT_ROOT)
        self.assertIn("com.autoresearch.worker", rendered)
        self.assertIn(str(PROJECT_ROOT), rendered)

    def test_prune_checkpoints_keeps_latest_final_and_recent_tags(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "demo-run"
            checkpoints_dir = run_checkpoints_dir(run_dir)
            names = [
                "step-000001.pt",
                "step-000002.pt",
                "thermal-serious.pt",
                "latest.pt",
                "final.pt",
            ]
            for index, name in enumerate(names, start=1):
                path = checkpoints_dir / name
                path.write_text(name, encoding="utf-8")
                stamp = 100 + index
                os.utime(path, (stamp, stamp))

            removed = prune_checkpoints(run_dir, keep_last=2)

            self.assertEqual({path.name for path in removed}, {"step-000001.pt"})
            remaining = {path.name for path in checkpoints_dir.glob("*.pt")}
            self.assertEqual(
                remaining,
                {"step-000002.pt", "thermal-serious.pt", "latest.pt", "final.pt"},
            )


if __name__ == "__main__":
    unittest.main()
