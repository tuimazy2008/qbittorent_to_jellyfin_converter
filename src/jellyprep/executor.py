"""Execute a plan: hardlink or remux into the library, verify, report."""
from __future__ import annotations

import errno
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from . import media, transcode

GB = 1024 ** 3


class NotEnoughSpace(Exception):
    pass


def needed_bytes(plan: dict, link_fallback: str = "copy") -> int:
    """Bytes the plan will write (hardlinks are free unless they fall back to copying)."""
    total = 0
    for a in plan["actions"]:
        if a["action"] in ("transcode", "process"):
            total += a.get("estimate") or a["size"]
        elif a["action"] == "remux":
            total += a["size"]
        elif a["action"] == "hardlink" and link_fallback == "copy" and not _same_device(a["src"], a["dst"]):
            total += a["size"]
    return total


def check_space(plan: dict, min_free_gb: float, link_fallback: str = "copy") -> None:
    need = needed_bytes(plan, link_fallback)
    if not need or not plan.get("target"):
        return
    free = media.free_bytes(plan["target"])
    if free - need < min_free_gb * GB:
        raise NotEnoughSpace(f"Нужно {need / GB:.1f} ГБ, свободно {free / GB:.1f} ГБ, "
                             f"минимум оставить {min_free_gb} ГБ")


Progress = Callable[[int, int, str, float, str], None]   # (file no, files, name, fraction, speed)


def execute(plan: dict, link_fallback: str = "copy", log: Callable[[str], None] = print,
            settings: dict | None = None, progress: Progress | None = None) -> list[dict]:
    done = []
    heavy = [a for a in plan["actions"] if a["action"] in ("transcode", "process", "remux")]
    for a in plan["actions"]:
        src, dst = Path(a["src"]), Path(a["dst"])
        if a["action"] == "exists":
            done.append({**_slim(a), "method": "exists"})
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if a["action"] == "hardlink":
            method = _link(src, dst, link_fallback)
        elif a["action"] == "remux":
            _remux(src, dst)
            method = "remux"
        elif a["action"] in ("transcode", "process"):
            n = heavy.index(a) + 1
            cb = (lambda f, sp, n=n, name=src.name: progress(n, len(heavy), name, f, sp)) if progress else None
            if progress:
                progress(n, len(heavy), src.name, 0.0, "")
            extra = transcode.process(str(src), str(dst), a["info"], settings or {},
                                      transcode=a["action"] == "transcode", progress=cb)
            method = a["action"]
        else:
            raise ValueError(f"unknown action {a['action']}")
        msg = f"{method}: {src.name} → {dst}"
        if extra:
            msg += f" ({extra['size'] / GB:.2f} ГБ из {a['size'] / GB:.2f} ГБ, {extra['seconds'] // 60} мин)"
        log(msg)
        done.append({**_slim(a), "method": method, **({"out_size": extra["size"]} if extra else {})})
    return done


def _slim(a: dict) -> dict:
    return {k: v for k, v in a.items() if k != "info"}


def verify(results: list[dict]) -> list[str]:
    """Return a list of problems with produced files (empty = all good)."""
    problems = []
    for r in results:
        dst = Path(r["dst"])
        if not dst.exists() or dst.stat().st_size == 0:
            problems.append(f"Нет файла: {dst}")
        elif r["method"] in ("hardlink", "exists") and Path(r["src"]).exists() \
                and not os.path.samefile(r["src"], dst):
            problems.append(f"Не хардлинк: {dst}")
        elif r["method"] == "copy" and Path(r["src"]).exists() and dst.stat().st_size != Path(r["src"]).stat().st_size:
            problems.append(f"Размер копии не совпадает: {dst}")
    return problems


def _same_device(src: str, dst: str) -> bool:
    d = Path(dst)
    while not d.exists():
        d = d.parent
    try:
        return os.stat(src).st_dev == os.stat(d).st_dev
    except OSError:
        return False


def _link(src: Path, dst: Path, fallback: str) -> str:
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError as e:
        if e.errno != errno.EXDEV or fallback != "copy":
            raise
    partial = dst.with_name(dst.name + ".partial")
    try:
        shutil.copy2(src, partial)
        partial.rename(dst)
    finally:
        partial.unlink(missing_ok=True)
    return "copy"


def _remux(src: Path, dst: Path) -> None:
    partial = dst.with_name(dst.stem + ".partial.mkv")
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
           "-fflags", "+genpts", "-i", str(src),
           "-map", "0", "-c", "copy", "-ignore_unknown", "-f", "matroska", str(partial)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"ffmpeg: {out.stderr.strip()[-500:]}")
        d_src, d_dst = media.duration(src), media.duration(partial)
        if d_src and abs(d_src - d_dst) > 1.5:
            raise RuntimeError(f"Длительность не совпадает: {d_src:.1f} c → {d_dst:.1f} c")
        partial.rename(dst)
    finally:
        partial.unlink(missing_ok=True)
