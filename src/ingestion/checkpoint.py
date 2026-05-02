"""
Checkpoint / resume system for incremental scraping.

Persists a set of completed (and failed) game IDs to a JSON file so that
long-running scrapes can be interrupted and resumed without re-fetching
already-collected data.

File layout::

    data/checkpoints/{data_type}_{season}.json

Writes are **atomic** — content is flushed to a temporary file first, then
renamed over the target path.  This prevents half-written checkpoints if the
process is killed mid-write.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR = Path("data/checkpoints")


@dataclass
class CheckpointState:
    """In-memory representation of a checkpoint file."""

    completed: set[int] = field(default_factory=set)
    failed: dict[int, str] = field(default_factory=dict)  # game_id → error message


class CheckpointManager:
    """Manage checkpoint state for a specific data type and season.

    Parameters
    ----------
    data_type : str
        One of ``"pbp"`` or ``"shifts"``.
    season : int
        The *start year* of the season (e.g. ``2024`` for 2024-25).
    base_dir : Path | None
        Override for the checkpoint directory (useful in tests).
    """

    def __init__(
        self,
        data_type: str,
        season: int,
        base_dir: Path | None = None,
    ) -> None:
        self.data_type = data_type
        self.season = season
        self._dir = base_dir or _CHECKPOINT_DIR
        self._path = self._dir / f"{data_type}_{season}.json"
        self._state = self._load()

    # -- public interface ---------------------------------------------------

    def is_complete(self, game_id: int) -> bool:
        """Return ``True`` if *game_id* was already successfully scraped."""
        return game_id in self._state.completed

    def mark_complete(self, game_id: int) -> None:
        """Record *game_id* as successfully scraped and persist to disk."""
        self._state.completed.add(game_id)
        # Remove from failed if it was there previously
        self._state.failed.pop(game_id, None)
        self._save()

    def mark_failed(self, game_id: int, error: str) -> None:
        """Record *game_id* as failed with an error message."""
        self._state.failed[game_id] = error
        self._save()

    @property
    def completed_count(self) -> int:
        return len(self._state.completed)

    @property
    def failed_count(self) -> int:
        return len(self._state.failed)

    @property
    def completed_ids(self) -> set[int]:
        return set(self._state.completed)

    # -- persistence --------------------------------------------------------

    def _load(self) -> CheckpointState:
        """Load state from disk, returning empty state if file is missing."""
        if not self._path.exists():
            logger.debug("No checkpoint file at %s — starting fresh", self._path)
            return CheckpointState()

        try:
            with open(self._path, encoding="utf-8") as f:
                raw = json.load(f)
            state = CheckpointState(
                completed=set(raw.get("completed", [])),
                failed={int(k): v for k, v in raw.get("failed", {}).items()},
            )
            logger.info(
                "Loaded checkpoint %s: %d completed, %d failed",
                self._path.name,
                len(state.completed),
                len(state.failed),
            )
            return state
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Corrupt checkpoint %s — starting fresh: %s", self._path, exc)
            return CheckpointState()

    def _save(self) -> None:
        """Atomically persist the current state to disk."""
        self._dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "data_type": self.data_type,
            "season": self.season,
            "completed": sorted(self._state.completed),
            "failed": {str(k): v for k, v in self._state.failed.items()},
        }

        # Write to temp file in same directory, then atomic rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._dir),
            prefix=f".{self.data_type}_{self.season}_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            # On Windows os.replace is atomic within the same volume
            os.replace(tmp_path, self._path)
        except BaseException:
            # Clean up temp file on any error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
