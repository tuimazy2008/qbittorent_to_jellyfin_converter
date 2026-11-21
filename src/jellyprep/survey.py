"""Counts for the "Перенос" page: what is in each old folder / what is left to do in the new library.

Runs in the background (probing hundreds of files takes a while the first time); ffprobe results are
cached per file (path + size + mtime), so later surveys are quick.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from . import config, media, transcode
from .worker import _inodes, _related, map_path, qbit_client

TTL = 600


def _signature(s: dict) -> tuple:
    keys = ("video_mode", "video_max_mbps", "voice_track", "keep_langs", "keep_unknown_lang", "max_tracks_per_lang")
    return tuple(str(s.get(k)) for k in keys)


class Surveys:
    def __init__(self, worker):
        self.w = worker
        self.db = worker.db
        self._lock = threading.Lock()
        self.data: dict[str, dict] = {}

    def get(self, path: str, kind: str, refresh: bool = False) -> dict:
        """Current survey of a folder; starts (or restarts) it in the background when needed."""
        s = config.load(self.db)
        with self._lock:
            cur = self.data.get(path)
            stale = (cur is None or refresh or (cur["state"] == "done" and time.time() - cur["at"] > TTL)
                     or cur.get("sig") != _signature(s))
            if stale and (cur is None or cur["state"] != "running"):
                cur = {"state": "running", "done": 0, "total": 0, "at": time.time(), "sig": _signature(s),
                       "result": (cur or {}).get("result"), "kind": kind}
                self.data[path] = cur
                threading.Thread(target=self._run, args=(path, kind), daemon=True, name="survey").start()
            return dict(cur)

    def invalidate(self, path: str) -> None:
        with self._lock:
            self.data.pop(path, None)

    def _run(self, path: str, kind: str) -> None:
        cur = self.data[path]
        try:
            cur["result"] = self.survey(path, kind, cur)
            cur["error"] = None
        except Exception as e:  # noqa: BLE001 — shown on the page
            cur["error"] = str(e)
        cur["state"], cur["at"] = "done", time.time()

    # --- the actual counting -------------------------------------------------------------
    def survey(self, path: str, kind: str, progress: dict | None = None) -> dict:
        s = config.load(self.db)
        root = Path(path)
        entries = [e for e in sorted(root.iterdir()) if not e.name.startswith(".")] if root.is_dir() else []
        if progress is not None:
            progress["total"] = len(entries)
        torrents = []
        try:
            torrents = [Path(map_path(t["content_path"], s["path_map"])) for t in qbit_client(s).torrents()]
        except Exception:  # noqa: BLE001 — without qBittorrent we just can't tell what is seeding
            pass
        t_inodes = _inodes(torrents) if kind == "old" else set()
        known = self.db.known_paths() if kind == "old" else self.db.active_paths()
        active = self.db.active_paths()
        out = []
        for n, entry in enumerate(entries):
            if progress is not None:
                progress["done"] = n
            try:
                sc = media.scan(entry)
            except OSError:
                continue
            if not sc.videos:
                continue
            size = sum(sc.sizes.get(v, 0) for v in sc.videos)
            row = {"name": entry.name, "size": size, "videos": len(sc.videos), "needs": []}
            if str(entry) in known:
                row["status"] = "known" if str(entry) in active else "leftover"
            elif kind == "old" and (any(_related(entry, t) for t in torrents) or _inodes([entry]) & t_inodes):
                row["status"] = "seeding"
            elif kind == "old":
                row["status"] = "move"
            else:
                voice = heavy = 0
                for v in sc.videos:
                    info = self._probe(sc.base / v)
                    if not info or not info.get("video"):
                        continue
                    tc, _ = transcode.video_decision(info, s)
                    reasons = transcode.processing_reasons(transcode.select_tracks(info, s))
                    heavy += bool(tc)
                    voice += any("Голос" in r for r in reasons)
                    if tc or reasons:
                        row["needs"].append(v)
                row["status"] = "work" if row["needs"] else "ok"
                row["voice"], row["heavy"] = voice, heavy
            out.append(row)
        if progress is not None:
            progress["done"] = len(entries)
        totals = {k: sum(1 for r in out if r["status"] == k) for k in ("move", "seeding", "known", "leftover", "work", "ok")}
        totals["size_move"] = sum(r["size"] for r in out if r["status"] == "move")
        totals["size"] = sum(r["size"] for r in out)
        totals["voice"] = sum(r.get("voice", 0) for r in out)
        totals["heavy"] = sum(r.get("heavy", 0) for r in out)
        return {"entries": out, "totals": totals}

    def _probe(self, path: Path) -> dict | None:
        try:
            st = path.stat()
        except OSError:
            return None
        cached = self.db.probe_get(str(path), st.st_size, st.st_mtime)
        if cached is not False:
            return cached
        try:
            info = transcode.probe(path)
        except Exception:  # noqa: BLE001
            info = None
        self.db.probe_set(str(path), st.st_size, st.st_mtime, info)
        return info


def is_video_library(lib: dict) -> bool:
    return (lib.get("CollectionType") or "") in ("", "movies", "tvshows", "mixed", "homevideos")


def folder_exists(p: str) -> bool:
    return os.path.isdir(p)
