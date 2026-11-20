"""Filesystem scan and ffprobe helpers."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_EXT = {".mkv", ".mp4", ".m4v", ".avi", ".ts", ".m2ts", ".mts", ".mov", ".wmv",
             ".mpg", ".mpeg", ".vob", ".webm", ".flv", ".divx"}
SUB_EXT = {".srt", ".ass", ".ssa", ".sub", ".idx", ".vtt", ".sup", ".smi"}
AUDIO_EXT = {".mka", ".ac3", ".eac3", ".dts", ".aac", ".flac", ".mp3", ".m4a", ".wav", ".opus"}
EXTRA_EXT = SUB_EXT | AUDIO_EXT

MIN_VIDEO_BYTES = 20 * 1024 * 1024
SAMPLE_RE = re.compile(r"(?i)(?:^|[\W_])(sample|trailer|трейлер)(?:[\W_]|$)")


@dataclass
class Scan:
    root: Path                     # torrent content path (dir or single file)
    base: Path                     # dir that relative paths are relative to
    videos: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    sizes: dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.root.name


def scan(content_path: str | Path) -> Scan:
    root = Path(content_path)
    if not root.exists():
        raise FileNotFoundError(f"Нет такого пути: {root}")
    if root.is_file():
        base, files = root.parent, [root]
    else:
        base, files = root, sorted(p for p in root.rglob("*") if p.is_file())
    s = Scan(root=root, base=base)
    for p in files:
        rel = str(p.relative_to(base))
        size = p.stat().st_size
        s.sizes[rel] = size
        ext = p.suffix.lower()
        if ext in VIDEO_EXT:
            if size < MIN_VIDEO_BYTES or (SAMPLE_RE.search(p.stem) and size < 600 * 1024 * 1024):
                s.ignored.append(rel)
            else:
                s.videos.append(rel)
        elif ext in EXTRA_EXT:
            s.extras.append(rel)
        else:
            s.ignored.append(rel)
    return s


def ffprobe(path: str | Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout)


def duration(path: str | Path) -> float:
    return float(ffprobe(path).get("format", {}).get("duration") or 0.0)


def free_bytes(path: str | Path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free
