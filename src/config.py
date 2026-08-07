"""Project-wide configuration values."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = RAW_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
HISTORICAL_STORE_DIR = PROCESSED_DIR / "player_games"

# Public player-tracking data is required by the current rebound model.
MIN_TRACKING_SEASON_START = 2013
MAX_SUPPORTED_SEASON_START = 2025

LOW_TRUST_SAMPLE_THRESHOLD = 5
HIGH_TRUST_SAMPLE_THRESHOLD = 10
MAX_SAMPLE_THRESHOLD = 15
LOW_TRUST_WEIGHT = 0.35
HIGH_TRUST_WEIGHT = 0.80
MAX_TRUST_WEIGHT = 0.95

# A matchup can move an opportunity projection by at most +/-50% before the
# sample-size trust curve shrinks it further toward neutral. These are safety
# bounds, not observed-stat filters: the unbounded ratio is retained for audit.
MIN_MATCHUP_MULTIPLIER = 0.50
MAX_MATCHUP_MULTIPLIER = 1.50

REQUEST_SLEEP_SECONDS = 0.6
REQUEST_RETRIES = 2
REQUEST_TIMEOUT_SECONDS = 45
