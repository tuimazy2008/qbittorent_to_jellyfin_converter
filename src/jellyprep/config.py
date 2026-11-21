"""Settings with defaults. Everything user-facing is stored in the DB and edited in the UI."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .db import DB

CONFIG_DIR = Path(os.environ.get("JELLYPREP_CONFIG", "/config"))

DEFAULTS: dict[str, Any] = {
    "setup_done": False,
    # qBittorrent
    "qbit_url": "http://qbittorrent:8080",
    "qbit_user": "",
    "qbit_pass": "",
    # [{"from": "<path as qBittorrent sees it>", "to": "<path inside this container>"}]
    "path_map": [],
    "library_root": "/media/library",
    # TMDB
    "tmdb_key": "",
    "tmdb_language": "ru-RU",
    # Optional local model (used from v0.3)
    "llm_kind": "none",          # none | ollama | openai
    "llm_url": "",
    "llm_model": "",
    "llm_key": "",
    "llm_auto": True,            # ask the model by itself when the rules can't identify something
    "llm_confirm_auto": True,
    "llm_reasoning": "off",
    "llm_vision": "auto",
    "llm_search": False,         # let the model look things up on Wikipedia (only search phrases leave the server)        # auto | off — show the model frames of films it can't name from file names      # off | on — "thinking" models (gemma 4, qwen 3) go in circles on long lists    # rules sure + model gives the same answer → continue without asking
    # Optional Jellyfin
    "jellyfin_url": "",
    "jellyfin_key": "",
    # [{"from": "<path inside jellyprep>", "to": "<same folder as Jellyfin sees it>"}]; empty = same paths
    "jellyfin_path_map": [],
    # Behaviour
    "dry_run": True,
    "poll_minutes": 5,
    "min_free_gb": 50,
    "confidence": 0.8,
    # Containers to repack into MKV (stream copy). Off by default: it makes a full copy and Jellyfin plays these anyway.
    "remux_containers": [],
    "link_fallback": "copy",     # copy | fail — when a hardlink is impossible (different disks)
    # Video (v0.2)
    "video_mode": "compat",      # compat | balanced | off
    "video_max_mbps": 12.0,      # compat: re-encode when the video bitrate is above this
    "video_quality": "standard", # small | standard | high  (target bitrate, see transcode.QUALITY)
    "hw_accel": "auto",          # auto | vaapi | software
    "hw_device": "/dev/dri/renderD128",
    # Audio and languages (v0.2)
    "voice_track": True,
    "voice_strength": "medium",  # light | medium | strong
    "voice_langs": ["rus", "eng"],
    "keep_langs": ["rus", "eng"],
    "keep_unknown_lang": True,
    "max_tracks_per_lang": 0,    # 0 = keep all; N = keep the first N audio tracks of each language
    # When heavy jobs (re-encoding) may run
    "night_only": True,
    "night_start": "01:00",
    "night_end": "08:00",
}

SECRET_KEYS = {"qbit_pass", "tmdb_key", "llm_key", "jellyfin_key"}


class Settings(dict):
    """Dict of settings with defaults filled in."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def load(db: DB) -> Settings:
    return Settings({**DEFAULTS, **db.get_settings()})


def save(db: DB, values: dict[str, Any]) -> None:
    """Save values. An empty string for a secret keeps the stored secret."""
    for key, value in values.items():
        if key not in DEFAULTS:
            continue
        if key in SECRET_KEYS and value in ("", None):
            continue
        db.set_setting(key, value)


def public(settings: dict[str, Any]) -> dict[str, Any]:
    """Settings safe to render: secrets replaced by a flag."""
    return {k: (bool(v) if k in SECRET_KEYS else v) for k, v in settings.items()}
