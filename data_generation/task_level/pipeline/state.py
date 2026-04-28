"""Pipeline run state management for resumability."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PipelineState:
    """Tracks pipeline progress for a single run, persisted as JSON."""

    def __init__(self, run_dir: Path, config: dict[str, Any] | None = None) -> None:
        self.run_dir = run_dir
        self._state_path = run_dir / "state.json"
        if self._state_path.exists():
            self._data = json.loads(self._state_path.read_text(encoding="utf-8"))
        else:
            self._data = {
                "run_id": run_dir.name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": config or {},
                "phases": {},
            }

    def mark_phase_started(self, phase: str) -> None:
        self._data["phases"].setdefault(phase, {})
        self._data["phases"][phase]["status"] = "in_progress"
        self._data["phases"][phase]["started_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        self._save()

    def mark_phase_completed(self, phase: str) -> None:
        self._data["phases"].setdefault(phase, {})
        self._data["phases"][phase]["status"] = "completed"
        self._data["phases"][phase]["completed_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        self._save()

    def phase_status(self, phase: str) -> str:
        return self._data["phases"].get(phase, {}).get("status", "pending")

    def update_phase_meta(self, phase: str, key: str, value: Any) -> None:
        self._data["phases"].setdefault(phase, {})[key] = value
        self._save()

    @property
    def config(self) -> dict[str, Any]:
        return self._data.get("config", {})

    def _save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(self._data, indent=2), encoding="utf-8"
        )
