"""Turn an identification into a list of file actions with Jellyfin-style target paths."""
from __future__ import annotations

import os
import re
from pathlib import Path, PurePath
from typing import Callable

from . import transcode
from .media import Scan

BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize(name: str) -> str:
    name = BAD_CHARS.sub(" ", name)
    name = " ".join(name.split()).strip(" .")
    return name[:180] or "Unknown"


def folder_name(chosen: dict) -> str:
    title, year, tmdb_id = chosen.get("title") or "Unknown", chosen.get("year"), chosen.get("id")
    base = f"{title} ({year})" if year else title
    return sanitize(base) + (f" [tmdbid-{tmdb_id}]" if tmdb_id else "")


def movie_stem(chosen: dict) -> str:
    title, year = chosen.get("title") or "Unknown", chosen.get("year")
    return sanitize(f"{title} ({year})" if year else title)


ProbeFn = Callable[[str], "dict | None"]
HEAVY = {"transcode", "process", "remux"}


def build(ident: dict, s: Scan, dest_root: Path, settings: dict, probe: ProbeFn | None = None) -> dict:
    """Return {"target", "actions": [...], "problems": [...], "skipped": [...]}.

    Video actions: hardlink (unchanged), remux (container only), process (video copied, audio/subtitles
    reworked), transcode (video re-encoded). With `probe` (path → transcode.summarize) the media rules
    from settings decide; without it everything is hardlinked.
    """
    remux = {"." + e.lower().lstrip(".") for e in settings.get("remux_containers", [])}
    actions: list[dict] = []
    problems: list[str] = []
    video_targets: dict[str, Path] = {}   # source rel → target path (without ext change for extras)

    def add_video(rel: str, dst_noext: Path) -> None:
        src = s.base / rel
        ext = src.suffix.lower()
        info = probe(str(src)) if probe else None
        action, reasons = "hardlink", []
        if info and info.get("video"):
            needs_tc, reasons = transcode.video_decision(info, settings)
            if needs_tc:
                action = "transcode"
            elif reasons := transcode.processing_reasons(transcode.select_tracks(info, settings)):
                action = "process"
        elif probe and info is None:
            problems.append(f"Не удалось прочитать файл: {rel}")
        if action == "hardlink" and ext in remux:
            action, reasons = "remux", [f"контейнер {ext} → mkv"]
        dst = dst_noext.with_name(dst_noext.name + (ext if action == "hardlink" else ".mkv"))
        a = {"src": str(src), "dst": str(dst), "action": action, "reasons": reasons,
             "size": s.sizes.get(rel, 0), "type": "video"}
        if info:
            a["info"] = info
            a["summary"] = transcode.describe(info)
            if action in ("transcode", "process"):
                a["estimate"] = transcode.estimate_bytes(info, settings, action == "transcode")
        actions.append(a)
        video_targets[rel] = dst

    if ident["kind"] == "movie":
        targets = set()
        for m in ident.get("movies", []):
            chosen = m.get("chosen")
            if not chosen:
                problems.append(f"Не распознан фильм: {', '.join(m['files'])}")
                continue
            folder = dest_root / folder_name(chosen)
            targets.add(str(folder))
            stem = movie_stem(chosen)
            for i, rel in enumerate(m["files"], 1):
                suffix = f" - part{i}" if len(m["files"]) > 1 else ""
                add_video(rel, folder / f"{stem}{suffix}")
        target = sorted(targets)[0] if len(targets) == 1 else str(dest_root)
    else:
        chosen = ident.get("show", {}).get("chosen")
        if not chosen:
            problems.append("Сериал не распознан")
            return {"target": None, "actions": [], "problems": problems, "skipped": []}
        folder = dest_root / folder_name(chosen)
        target = str(folder)
        show = sanitize(chosen.get("title") or "Unknown")
        for e in ident.get("episodes", []):
            add_video(e["file"], folder / f"Season {e['season']:02d}" / f"{show} S{e['season']:02d}E{e['episode']:02d}")
        for rel in ident.get("unmatched", []):
            problems.append(f"Файл без номера серии пропущен: {rel}")

    skipped = []
    for rel in s.extras:
        dst = _extra_target(rel, s, video_targets)
        if dst is None:
            skipped.append(rel)
            continue
        actions.append({"src": str(s.base / rel), "dst": str(dst), "action": "hardlink",
                        "size": s.sizes.get(rel, 0), "type": "extra"})

    seen: set[str] = set()
    clashes, occupied = [], []
    for a in actions:
        if a["dst"] in seen:
            clashes.append(PurePath(a["dst"]).name)
        seen.add(a["dst"])
        if os.path.exists(a["dst"]):
            if _same_file(a["src"], a["dst"]):
                if a["action"] in ("transcode", "process", "remux"):
                    a["inplace"] = True   # re-processing a file that already sits in the library
                else:
                    a["action"] = "exists"
            else:
                occupied.append(a["dst"])
    if clashes:
        names = list(dict.fromkeys(clashes))
        problems.append(f"Несколько файлов получают одно имя ({len(clashes)}): " + ", ".join(names[:3])
                        + (" …" if len(names) > 3 else ""))
    if occupied:
        problems.append(f"В библиотеке уже есть другие файлы с такими именами ({len(occupied)}): "
                        + ", ".join(occupied[:2]) + (" …" if len(occupied) > 2 else ""))
    return {"target": target, "actions": actions, "problems": problems, "skipped": skipped + s.ignored}


def _extra_target(rel: str, s: Scan, video_targets: dict[str, Path]) -> Path | None:
    """Match subtitles/audio to a video by name prefix; keep the rest of the name as a label."""
    p = PurePath(rel)
    best = None
    for vrel, vdst in video_targets.items():
        vstem = PurePath(vrel).stem
        if p.stem.lower().startswith(vstem.lower()) and (best is None or len(vstem) > len(best[0])):
            best = (vstem, vdst, PurePath(vrel))
    if best is None:
        return None
    vstem, vdst, vrel_path = best
    label = p.stem[len(vstem):].strip(" ._-")
    if not label and p.parent != vrel_path.parent:
        label = p.parent.name  # e.g. "RUS Sound/<video>.mka"
    label = sanitize(label).replace(" ", "_") if label else ""
    return vdst.with_name(Path(vdst).stem + (f".{label}" if label else "") + p.suffix.lower())


def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False
