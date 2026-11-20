"""Identify what a torrent is: movie(s) or a TV show with seasons/episodes, matched against TMDB."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import PurePath
from typing import Any

from guessit import guessit

from . import parse
from .clients import TMDB, Candidate, ClientError
from .media import Scan

NO_TMDB_CONFIDENCE = 0.5


def score(query_titles: list[str], year: int | None, c: Candidate) -> float:
    sim = max((parse.similarity(q, t) for q in query_titles for t in (c.title, c.original_title) if t),
              default=0.0)
    if year and c.year:
        diff = abs(year - c.year)
        sim += 0.1 if diff == 0 else 0.05 if diff == 1 else -0.25
    return max(0.0, min(1.0, sim))


def match(tmdb: TMDB | None, kind: str, queries: list[str], year: int | None) -> dict[str, Any]:
    """Search TMDB with several queries; return candidates ranked by score."""
    result: dict[str, Any] = {"queries": queries, "year": year, "candidates": [], "chosen": None, "score": 0.0}
    if tmdb is None:
        return result
    found: dict[int, Candidate] = {}
    for q in queries:
        hits = tmdb.search(kind, q, year)
        if not hits and year:
            hits = tmdb.search(kind, q)
        for c in hits:
            found.setdefault(c.id, c)
    ranked = sorted(found.values(), key=lambda c: (score(queries, year, c), c.popularity), reverse=True)
    result["candidates"] = [{**c.as_dict(), "score": round(score(queries, year, c), 3)} for c in ranked[:8]]
    if ranked:
        top = result["candidates"][0]
        s = top["score"]
        if len(ranked) > 1 and result["candidates"][1]["score"] >= s - 0.05:
            s -= 0.15  # ambiguous: two candidates look the same
        if year is None and len(ranked) > 1:
            s = min(s, 0.75)  # "Radar" with no year matches many shows exactly: always confirm
        result["chosen"], result["score"] = top, round(max(0.0, s), 3)
    return result


def _manual(title: str, year: int | None) -> dict:
    return {"id": None, "title": title, "original_title": "", "year": year, "score": 1.0}


def _apply_choice(entry: dict, tmdb: TMDB | None, kind: str, override: dict | None) -> None:
    """Apply a user choice from the review page: TMDB id, or a manual title/year."""
    if not override:
        return
    if override.get("tmdb_id") and tmdb is not None:
        c = next((c for c in entry["candidates"] if c["id"] == override["tmdb_id"]), None)
        if c is None:
            c = {**tmdb.candidate(kind, override["tmdb_id"]).as_dict(), "score": 1.0}
            entry["candidates"].insert(0, c)
        entry["chosen"] = c
    elif override.get("title"):
        entry["chosen"] = _manual(override["title"], override.get("year"))
    else:
        return
    entry["score"] = 1.0
    entry["manual"] = True


TMDBID = re.compile(r"\[tmdbid-(\d+)\]")


def _id_in_name(path: str) -> dict | None:
    """Folders made by jellyprep (or named for Jellyfin) carry the TMDB id: "Title (2003) [tmdbid-508]"."""
    m = TMDBID.search(path)
    return {"tmdb_id": int(m[1])} if m else None


def _safe_choice(entry: dict, tmdb: TMDB | None, kind: str, choice: dict | None, problems: list[str]) -> None:
    try:
        _apply_choice(entry, tmdb, kind, choice)
    except (ClientError, StopIteration, KeyError) as e:
        problems.append(f"Не удалось взять TMDB {choice}: {e}")


def identify(s: Scan, kind_hint: str, tmdb: TMDB | None, overrides: dict | None = None) -> dict:
    overrides = overrides or {}
    kind = overrides.get("kind") or kind_hint
    if kind not in ("movie", "tv"):
        kind = _detect_kind(s)
    ident = identify_tv(s, tmdb, overrides) if kind == "tv" else identify_movies(s, tmdb, overrides)
    ident["kind"] = kind
    return ident


def _detect_kind(s: Scan) -> str:
    if parse.guess_kind(s.name) == "tv":
        return "tv"
    numbered = sum(1 for v in s.videos if parse.parse_episode(v)[1] is not None)
    explicit = sum(1 for v in s.videos if parse.parse_episode(v)[2] in ("SxxEyy", "NxM"))
    if explicit or (len(s.videos) >= 3 and numbered >= len(s.videos) * 0.8):
        return "tv"
    return "movie"


# --- movies --------------------------------------------------------------
def identify_movies(s: Scan, tmdb: TMDB | None, overrides: dict) -> dict:
    problems: list[str] = []
    groups: dict[tuple, list[str]] = defaultdict(list)
    single = len(s.videos) == 1
    for v in s.videos:
        g = guessit(PurePath(v).name)
        title, year, alts = parse.guess_title(PurePath(v).name)
        has_part = g.get("cd") or g.get("part") or g.get("disc")
        key = (parse._norm(title), year) if has_part else (v,)
        groups[key].append(v)
    movies = []
    for idx, files in enumerate(groups.values()):
        files.sort()
        title, year, alts = parse.guess_title(PurePath(files[0]).name)
        queries = [title, *alts]
        if single or (len(groups) == 1):
            t2, y2, alts2 = parse.guess_title(s.name)
            queries += [t2, *alts2]
            year = year or y2
        queries = [q for q in dict.fromkeys(queries) if q]
        entry = {"files": files, "guess": {"title": title, "year": year}, **_safe_match(tmdb, "movie", queries, year, problems)}
        choice = (overrides.get("movies") or {}).get(str(idx)) or _id_in_name(f"{s.name}/{files[0]}")
        _safe_choice(entry, tmdb, "movie", choice, problems)
        if entry["chosen"] is None and tmdb is None and title:
            entry["chosen"], entry["score"] = _manual(title, year), NO_TMDB_CONFIDENCE
        movies.append(entry)
    if not s.videos:
        problems.append("Видеофайлы не найдены")
    if len(movies) > 1:
        problems.append(f"Несколько фильмов в одном торренте ({len(movies)}): проверьте каждый")
    conf = min((m["score"] for m in movies), default=0.0)
    return {"movies": movies, "confidence": round(conf, 3), "problems": problems}


# --- TV ------------------------------------------------------------------
def identify_tv(s: Scan, tmdb: TMDB | None, overrides: dict) -> dict:
    problems: list[str] = []
    title, year, alts = parse.guess_title(s.name)
    root_season = parse.season_from_name(s.name)
    sibling_dirs: dict[str, set[str]] = defaultdict(set)
    for v in s.videos:
        parts = PurePath(v).parts[:-1]
        for i in range(len(parts)):
            sibling_dirs["/".join(parts[:i])].add(parts[i])
    siblings = {k: sorted(v) for k, v in sibling_dirs.items()}

    episodes, unmatched = [], []
    manual_eps = overrides.get("episodes") or {}
    for v in s.videos:
        if v in manual_eps:
            se, ep = manual_eps[v]
            episodes.append({"file": v, "season": se, "episode": ep, "how": "вручную"})
            continue
        season, ep, how = parse.parse_episode(PurePath(v).name)
        if season is None:
            season = parse.season_from_dirs(v, siblings)
        if season is None:
            season = root_season
        if ep is None:
            unmatched.append(v)
            continue
        if season is None:
            season = 1
            how += ", сезон не найден → 1"
        episodes.append({"file": v, "season": season, "episode": ep, "how": how})

    queries = [q for q in dict.fromkeys([title, *alts]) if q]
    show = {"guess": {"title": title, "year": year}, **_safe_match(tmdb, "tv", queries, year, problems)}
    _safe_choice(show, tmdb, "tv", overrides.get("show") or _id_in_name(s.name), problems)
    if show["chosen"] is None and tmdb is None and title:
        show["chosen"], show["score"] = _manual(title, year), NO_TMDB_CONFIDENCE
    conf = show["score"]

    if unmatched:
        problems.append(f"Не найден номер серии у {len(unmatched)} файлов")
        conf = min(conf, 0.5)
    if any("сезон не найден" in e["how"] for e in episodes):
        problems.append("Сезон не указан в именах, принят сезон 1")
        conf -= 0.1
    dupes = _duplicates(episodes)
    if dupes:
        problems.append("Повторяются номера серий: " + ", ".join(dupes[:5]))
        conf = min(conf, 0.4)
    chosen = show.get("chosen") or {}
    if tmdb is not None and chosen.get("id") and episodes:
        try:
            sizes = tmdb.season_sizes(chosen["id"])
        except ClientError as e:
            problems.append(str(e))
            sizes = {}
        if _remap_absolute(episodes, sizes, set(manual_eps)):
            problems[:] = [p for p in problems if "принят сезон 1" not in p]
            first, last = episodes[0], episodes[-1]
            problems.append(f"Сквозная нумерация серий разложена по сезонам TMDB "
                            f"(до S{last['season']:02d}E{last['episode']:02d}) — проверьте")
            conf -= 0.15
        for season in sorted({e["season"] for e in episodes}):
            top = max(e["episode"] for e in episodes if e["season"] == season)
            if season not in sizes:
                problems.append(f"В TMDB у сериала нет сезона {season}")
                conf -= 0.3
            elif top > sizes[season] > 0:
                problems.append(f"Сезон {season}: серия {top}, а в TMDB всего {sizes[season]}")
                conf -= 0.3
        show["season_sizes"] = sizes
    if not s.videos:
        problems.append("Видеофайлы не найдены")
        conf = 0.0
    episodes.sort(key=lambda e: (e["season"], e["episode"], e["file"]))
    return {"show": show, "episodes": episodes, "unmatched": unmatched,
            "confidence": round(max(0.0, conf), 3), "problems": problems}


def _remap_absolute(episodes: list[dict], sizes: dict[int, int], manual: set[str]) -> bool:
    """Releases often number a show 1..78 in one folder while TMDB has seasons of 26.

    When every (non-manual) episode is in season 1, numbers are unique and run past TMDB's season 1
    but fit into the regular seasons together, convert absolute numbers to season/episode.
    Works for raw releases and for files already laid out as S01E27.
    """
    auto = [e for e in episodes if e["file"] not in manual]
    regular = [k for k in sorted(sizes) if k >= 1 and sizes[k] > 0]
    if not auto or not regular or regular[0] != 1:
        return False
    if any(e["season"] != 1 for e in auto) or len({e["episode"] for e in auto}) != len(auto):
        return False
    top = max(e["episode"] for e in auto)
    if top <= sizes[1] or top > sum(sizes[k] for k in regular):
        return False
    # An explicit "S01E15" in a 9-episode season is a numbering problem, not absolute numbering —
    # unless there are more files than season 1 can hold (78 files laid out as S01E01..S01E78).
    no_season_in_names = all("сезон не найден" in e["how"] for e in auto)
    if not no_season_in_names and len(auto) <= sizes[1]:
        return False
    for e in auto:
        n = e["episode"]
        for season in regular:
            if n <= sizes[season]:
                break
            n -= sizes[season]
        if (season, n) != (1, e["episode"]):
            e["how"] += f", сквозная №{e['episode']} → S{season:02d}E{n:02d}"
        e["season"], e["episode"] = season, n
    episodes.sort(key=lambda e: (e["season"], e["episode"], e["file"]))
    return True


def _duplicates(episodes: list[dict]) -> list[str]:
    seen: dict[tuple, int] = defaultdict(int)
    for e in episodes:
        seen[(e["season"], e["episode"])] += 1
    return [f"S{s:02d}E{e:02d}" for (s, e), n in sorted(seen.items()) if n > 1]


def _safe_match(tmdb: TMDB | None, kind: str, queries: list[str], year: int | None, problems: list[str]) -> dict:
    try:
        return match(tmdb, kind, queries, year)
    except ClientError as e:
        problems.append(str(e))
        return match(None, kind, queries, year)
