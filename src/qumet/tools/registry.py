"""Registry for QuMET cache directories and paths.

This module defines the main cache directory location for QuMET.
"""

from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parents[2]
MAIN_CACHE_DIR = MAIN_DIR / ".qumet_cache"
