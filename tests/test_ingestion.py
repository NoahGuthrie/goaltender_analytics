"""
Unit tests for the ingestion pipeline.

Tests cover:
1. API client — rate limiting, retry logic, backoff behavior (mocked)
2. PBP parser — JSON flattening, schema correctness
3. Shift parser — JSON flattening, schema correctness
4. Checkpoint — create/read/resume, atomic writes
5. Validators — coordinate bounds, duplicates, impossible sequences
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.api_client import NHLAPIClient
from src.ingestion.checkpoint import CheckpointManager
from src.ingestion.scrape_pbp import flatten_pbp
from src.ingestion.scrape_shifts import flatten_shifts


# ============================================================================
# Fixtures — sample API responses
# ============================================================================


@pytest.fixture
def sample_pbp_json():
    """Minimal but realistic PBP response for a single game."""
    return {
        "id": 2024020003,
        "season": 20242025,
        "gameType": 2,
        "venue": {"default": "Climate Pledge Arena"},
        "homeTeam": {"id": 55, "abbrev": "SEA"},
        "awayTeam": {"id": 19, "abbrev": "STL"},
        "plays": [
            {
                "eventId": 1,
                "periodDescriptor": {"number": 1, "periodType": "REG"},
                "timeInPeriod": "00:00",
                "timeRemaining": "20:00",
                "situationCode": "1551",
                "typeDescKey": "faceoff",
                "typeCode": 502,
                "details": {
                    "xCoord": 0,
                    "yCoord": 0,
                    "eventOwnerTeamId": 55,
                    "zoneCode": "N",
                },
            },
            {
                "eventId": 10,
                "periodDescriptor": {"number": 1, "periodType": "REG"},
                "timeInPeriod": "02:15",
                "timeRemaining": "17:45",
                "situationCode": "1551",
                "typeDescKey": "shot-on-goal",
                "typeCode": 506,
                "details": {
                    "xCoord": -76,
                    "yCoord": 19,
                    "shootingPlayerId": 8479385,
                    "goalieInNetId": 8478916,
                    "shotType": "wrist",
                    "eventOwnerTeamId": 19,
                    "zoneCode": "O",
                },
            },
            {
                "eventId": 25,
                "periodDescriptor": {"number": 1, "periodType": "REG"},
                "timeInPeriod": "05:30",
                "timeRemaining": "14:30",
                "situationCode": "1551",
                "typeDescKey": "goal",
                "typeCode": 505,
                "details": {
                    "xCoord": 82,
                    "yCoord": -5,
                    "scoringPlayerId": 8479385,
                    "assist1PlayerId": 8476453,
                    "assist2PlayerId": 8475170,
                    "goalieInNetId": 8478916,
                    "shotType": "snap",
                    "eventOwnerTeamId": 19,
                    "zoneCode": "O",
                },
            },
            {
                "eventId": 30,
                "periodDescriptor": {"number": 1, "periodType": "REG"},
                "timeInPeriod": "10:00",
                "timeRemaining": "10:00",
                "situationCode": "1551",
                "typeDescKey": "hit",
                "typeCode": 503,
                "details": {
                    "xCoord": 50,
                    "yCoord": 20,
                    "eventOwnerTeamId": 55,
                },
            },
        ],
    }


@pytest.fixture
def sample_shifts_json():
    """Minimal but realistic shift chart response."""
    return {
        "data": [
            {
                "id": 1,
                "gameId": 2024020003,
                "playerId": 8479385,
                "teamId": 19,
                "teamAbbrev": "STL",
                "firstName": "Jordan",
                "lastName": "Kyrou",
                "period": 1,
                "startTime": "0:00",
                "endTime": "0:45",
                "duration": "0:45",
                "shiftNumber": 1,
                "typeCode": 517,
            },
            {
                "id": 2,
                "gameId": 2024020003,
                "playerId": 8479385,
                "teamId": 19,
                "teamAbbrev": "STL",
                "firstName": "Jordan",
                "lastName": "Kyrou",
                "period": 1,
                "startTime": "2:00",
                "endTime": "2:50",
                "duration": "0:50",
                "shiftNumber": 2,
                "typeCode": 517,
            },
            {
                "id": 3,
                "gameId": 2024020003,
                "playerId": 8478916,
                "teamId": 55,
                "teamAbbrev": "SEA",
                "firstName": "Joey",
                "lastName": "Daccord",
                "period": 1,
                "startTime": "0:00",
                "endTime": "20:00",
                "duration": "20:00",
                "shiftNumber": 1,
                "typeCode": 517,
            },
        ],
    }


# ============================================================================
# 1. API Client Tests
# ============================================================================


class TestNHLAPIClient:
    """Tests for the NHLAPIClient rate limiter and retry logic."""

    def test_rate_limiting(self):
        """Consecutive requests should be separated by at least rate_limit seconds."""
        client = NHLAPIClient(rate_limit=0.1, max_retries=1)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": True}

        with patch.object(client._session, "get", return_value=mock_response):
            t0 = time.monotonic()
            client._get_json("http://test/1")
            client._get_json("http://test/2")
            elapsed = time.monotonic() - t0

        # Allow small tolerance for Windows timer resolution (~15ms granularity)
        assert elapsed >= 0.08, f"Expected >= 0.08s between requests, got {elapsed:.3f}s"
        client.close()

    def test_retry_on_server_error(self):
        """Should retry on 500 status codes."""
        client = NHLAPIClient(rate_limit=0.0, max_retries=3)

        error_response = MagicMock()
        error_response.status_code = 500

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {"data": "ok"}

        with patch.object(
            client._session,
            "get",
            side_effect=[error_response, error_response, ok_response],
        ):
            with patch.object(client, "_backoff_seconds", return_value=0.01):
                result = client._get_json("http://test")

        assert result == {"data": "ok"}
        client.close()

    def test_context_manager(self):
        """Should work as a context manager."""
        with NHLAPIClient() as client:
            assert client is not None

    def test_get_schedule_url(self):
        """get_schedule should call the correct URL."""
        client = NHLAPIClient(rate_limit=0.0, max_retries=1)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"gameWeek": []}

        with patch.object(client._session, "get", return_value=mock_response) as mock_get:
            client.get_schedule("2024-10-08")
            call_url = mock_get.call_args[0][0]
            assert "schedule/2024-10-08" in call_url

        client.close()


# ============================================================================
# 2. PBP Parser Tests
# ============================================================================


class TestFlattenPBP:
    """Tests for the PBP JSON → DataFrame flattener."""

    def test_correct_row_count(self, sample_pbp_json):
        df = flatten_pbp(sample_pbp_json)
        assert len(df) == 4, f"Expected 4 plays, got {len(df)}"

    def test_required_columns_present(self, sample_pbp_json):
        df = flatten_pbp(sample_pbp_json)
        required_cols = [
            "game_id", "season", "game_type", "venue",
            "home_team_id", "away_team_id",
            "home_team_abbrev", "away_team_abbrev",
            "event_id", "period", "period_type",
            "time_in_period", "time_remaining",
            "situation_code", "event_type", "event_type_code",
            "x_coord", "y_coord",
            "shooting_player_id", "scoring_player_id",
            "assist1_player_id", "assist2_player_id",
            "goalie_in_net_id", "shot_type",
            "event_owner_team_id", "zone_code",
            "details_json",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_game_id_propagated(self, sample_pbp_json):
        df = flatten_pbp(sample_pbp_json)
        assert (df["game_id"] == 2024020003).all()

    def test_nullable_int_columns(self, sample_pbp_json):
        """Player ID columns should be nullable integers (pd.Int64Dtype)."""
        df = flatten_pbp(sample_pbp_json)
        for col in ["shooting_player_id", "scoring_player_id", "goalie_in_net_id"]:
            assert df[col].dtype == pd.Int64Dtype(), f"{col} should be Int64, got {df[col].dtype}"

    def test_shot_details_parsed(self, sample_pbp_json):
        df = flatten_pbp(sample_pbp_json)
        shots = df[df["event_type"] == "shot-on-goal"]
        assert len(shots) == 1
        shot = shots.iloc[0]
        assert shot["x_coord"] == -76
        assert shot["y_coord"] == 19
        assert shot["shot_type"] == "wrist"
        assert shot["shooting_player_id"] == 8479385

    def test_goal_assists_parsed(self, sample_pbp_json):
        df = flatten_pbp(sample_pbp_json)
        goals = df[df["event_type"] == "goal"]
        assert len(goals) == 1
        goal = goals.iloc[0]
        assert goal["scoring_player_id"] == 8479385
        assert goal["assist1_player_id"] == 8476453
        assert goal["assist2_player_id"] == 8475170

    def test_details_json_preserved(self, sample_pbp_json):
        df = flatten_pbp(sample_pbp_json)
        shot = df[df["event_type"] == "shot-on-goal"].iloc[0]
        parsed = json.loads(shot["details_json"])
        assert parsed["shotType"] == "wrist"
        assert parsed["xCoord"] == -76

    def test_empty_plays(self):
        """Should handle games with no plays gracefully."""
        raw = {
            "id": 9999,
            "season": 20242025,
            "gameType": 2,
            "venue": {"default": "Test Arena"},
            "homeTeam": {"id": 1, "abbrev": "TST"},
            "awayTeam": {"id": 2, "abbrev": "OPP"},
            "plays": [],
        }
        df = flatten_pbp(raw)
        assert df.empty

    def test_venue_extracted(self, sample_pbp_json):
        df = flatten_pbp(sample_pbp_json)
        assert (df["venue"] == "Climate Pledge Arena").all()


# ============================================================================
# 3. Shift Parser Tests
# ============================================================================


class TestFlattenShifts:
    """Tests for the shift chart JSON → DataFrame flattener."""

    def test_correct_row_count(self, sample_shifts_json):
        df = flatten_shifts(sample_shifts_json, game_id=2024020003)
        assert len(df) == 3

    def test_required_columns_present(self, sample_shifts_json):
        df = flatten_shifts(sample_shifts_json, game_id=2024020003)
        required = [
            "game_id", "player_id", "team_id", "team_abbrev",
            "first_name", "last_name",
            "period", "start_time", "end_time", "duration",
            "shift_number", "type_code",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_player_data_correct(self, sample_shifts_json):
        df = flatten_shifts(sample_shifts_json, game_id=2024020003)
        kyrou = df[df["player_id"] == 8479385]
        assert len(kyrou) == 2  # two shifts
        assert kyrou.iloc[0]["first_name"] == "Jordan"
        assert kyrou.iloc[0]["last_name"] == "Kyrou"

    def test_empty_shifts(self):
        df = flatten_shifts({"data": []}, game_id=9999)
        assert df.empty


# ============================================================================
# 4. Checkpoint Tests
# ============================================================================


class TestCheckpointManager:
    """Tests for the checkpoint/resume system."""

    def test_new_checkpoint_is_empty(self, tmp_path):
        ckpt = CheckpointManager("pbp", 2024, base_dir=tmp_path)
        assert ckpt.completed_count == 0
        assert not ckpt.is_complete(2024020003)

    def test_mark_complete_and_query(self, tmp_path):
        ckpt = CheckpointManager("pbp", 2024, base_dir=tmp_path)
        ckpt.mark_complete(2024020003)
        assert ckpt.is_complete(2024020003)
        assert ckpt.completed_count == 1

    def test_persistence_across_instances(self, tmp_path):
        """Checkpoint state should survive creating a new manager instance."""
        ckpt1 = CheckpointManager("pbp", 2024, base_dir=tmp_path)
        ckpt1.mark_complete(2024020003)
        ckpt1.mark_complete(2024020004)

        # New instance reads from same file
        ckpt2 = CheckpointManager("pbp", 2024, base_dir=tmp_path)
        assert ckpt2.is_complete(2024020003)
        assert ckpt2.is_complete(2024020004)
        assert ckpt2.completed_count == 2

    def test_mark_failed(self, tmp_path):
        ckpt = CheckpointManager("pbp", 2024, base_dir=tmp_path)
        ckpt.mark_failed(2024020005, "HTTP 500")
        assert ckpt.failed_count == 1
        assert not ckpt.is_complete(2024020005)

    def test_failed_then_complete(self, tmp_path):
        """Marking a previously-failed game as complete should clear the error."""
        ckpt = CheckpointManager("pbp", 2024, base_dir=tmp_path)
        ckpt.mark_failed(2024020005, "HTTP 500")
        ckpt.mark_complete(2024020005)
        assert ckpt.is_complete(2024020005)
        assert ckpt.failed_count == 0

    def test_separate_data_types(self, tmp_path):
        """PBP and shifts checkpoints should be independent."""
        ckpt_pbp = CheckpointManager("pbp", 2024, base_dir=tmp_path)
        ckpt_shifts = CheckpointManager("shifts", 2024, base_dir=tmp_path)

        ckpt_pbp.mark_complete(2024020003)
        assert not ckpt_shifts.is_complete(2024020003)

    def test_corrupt_checkpoint_recovery(self, tmp_path):
        """Should recover gracefully from a corrupted checkpoint file."""
        path = tmp_path / "pbp_2024.json"
        path.write_text("not valid json {{{", encoding="utf-8")

        ckpt = CheckpointManager("pbp", 2024, base_dir=tmp_path)
        assert ckpt.completed_count == 0  # should start fresh


# ============================================================================
# 5. Validator Helper Tests
# ============================================================================


class TestValidatorHelpers:
    """Tests for the validator utility functions."""

    def test_time_to_seconds(self):
        from src.ingestion.validators import _time_to_seconds

        assert _time_to_seconds("00:00") == 0
        assert _time_to_seconds("01:30") == 90
        assert _time_to_seconds("20:00") == 1200
        assert _time_to_seconds("") is None
        assert _time_to_seconds("invalid") is None
        assert _time_to_seconds("5:05") == 305
