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
NO_YEAR_PROBLEM = "Нет года в имени, а в TMDB несколько похожих — проверьте"


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
            result["no_year_ambiguous"] = True
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


def identify(s: Scan, kind_hint: str, tmdb: TMDB | None, overrides: dict | None = None,
             hints: list[str] | None = None) -> dict:
    """hints: names found inside the file (tags) or given by the model; used as extra TMDB queries."""
    overrides = overrides or {}
    hints = [h for h in [*(hints or []), *(overrides.get("hints") or [])] if h]
    kind = overrides.get("kind") or kind_hint
    if kind not in ("movie", "tv"):
        kind = _detect_kind(s)
    ident = identify_tv(s, tmdb, overrides, hints) if kind == "tv" else identify_movies(s, tmdb, overrides, hints)
    ident["kind"] = kind
    if hints:
        ident["hints"] = hints
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
def identify_movies(s: Scan, tmdb: TMDB | None, overrides: dict, hints: list[str] | None = None) -> dict:
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
            for h in hints or []:                       # "The.Time.Machine.2002…" written in the file's tags
                t3, y3, _ = parse.guess_title(h)
                queries.append(t3)
                year = year or y3
        queries = [q for q in dict.fromkeys(queries) if q]
        entry = {"files": files, "guess": {"title": title, "year": year}, **_safe_match(tmdb, "movie", queries, year, problems)}
        choice = (overrides.get("movies") or {}).get(str(idx)) or _id_in_name(f"{s.name}/{files[0]}")
        _safe_choice(entry, tmdb, "movie", choice, problems)
        if entry["chosen"] is None and tmdb is None and title:
            entry["chosen"], entry["score"] = _manual(title, year), NO_TMDB_CONFIDENCE
        if entry.get("no_year_ambiguous") and not entry.get("manual"):
            problems.append(f"{NO_YEAR_PROBLEM}: {', '.join(entry['files'][:2])}")
        movies.append(entry)
    if not s.videos:
        problems.append("Видеофайлы не найдены")
    if len(movies) > 1:
        problems.append(f"Несколько фильмов в одном торренте ({len(movies)}): проверьте каждый")
    conf = min((m["score"] for m in movies), default=0.0)
    return {"movies": movies, "confidence": round(conf, 3), "problems": problems}


# --- TV ------------------------------------------------------------------
def identify_tv(s: Scan, tmdb: TMDB | None, overrides: dict, hints: list[str] | None = None) -> dict:
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
            se, ep, *rest = manual_eps[v]
            e = {"file": v, "season": se, "episode": ep, "how": overrides.get("episodes_how") or "вручную"}
            if rest and rest[0]:
                e["episode_end"] = rest[0]
            episodes.append(e)
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

    base_title = re.sub(r"[\s.:-]*\d+(?:\.\d+)?$", "", title).strip()   # "Пин-код 2" → also "Пин-код"
    queries = [q for q in dict.fromkeys([title, *alts, base_title, *(parse.guess_title(h)[0] for h in hints or [])]) if q]
    show = {"guess": {"title": title, "year": year}, **_safe_match(tmdb, "tv", queries, year, problems)}
    _safe_choice(show, tmdb, "tv", overrides.get("show") or _id_in_name(s.name), problems)
    if show["chosen"] is None and tmdb is None and title:
        show["chosen"], show["score"] = _manual(title, year), NO_TMDB_CONFIDENCE
    if show.get("no_year_ambiguous") and not show.get("manual"):
        problems.append(NO_YEAR_PROBLEM)
    conf = show["score"]

    chosen0 = show.get("chosen") or {}
    if tmdb is not None and chosen0.get("id") and (episodes or unmatched):
        matched, changed, switched = _match_episode_titles(
            episodes, unmatched, tmdb, show, set(manual_eps), show_fixed=bool(overrides.get("show")))
        if switched:
            problems.append(f"По названиям серий это сериал «{switched['title']}» ({switched.get('year') or '?'}), "
                            f"а не первый найденный — проверьте")
            conf = show["score"]
        if changed:
            problems.append(f"Серии сопоставлены по названиям в TMDB ({changed} из {matched} файлов "
                            f"не по номерам в именах) — проверьте")
        if matched and any("сезон не найден" in e["how"] for e in episodes) is False:
            problems[:] = [p for p in problems if "принят сезон 1" not in p]

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


PART_IN_FILE = re.compile(r"\((\d+(?:\s*[,\-–]\s*\d+)*)\s*(?:сер|серия|серии|ч|часть|part)\.?\)", re.I)
PART_WORD_IN_FILE = re.compile(r"\((?:часть|ч\.|серия|part)\s*(\d+)\)", re.I)
PART_IN_NAME = re.compile(r"^(.*?)[\s.,:;-]*(?:часть|ч\.|серия|part)\s*(\d+)\s*\.?$", re.I)
LEADING_NUM = re.compile(r"^\s*\d{1,3}[.\s_-]+")


def _file_title(path: str) -> tuple[str, list[int]]:
    """'1. Мягкая лапа смерти (1,2 сер.).avi' → ('Мягкая лапа смерти', [1, 2])"""
    stem = PurePath(path).stem
    parts: list[int] = []
    if m := PART_IN_FILE.search(stem):
        nums = [int(x) for x in re.findall(r"\d+", m[1])]
        parts = list(range(nums[0], nums[-1] + 1)) if re.search(r"[-–]", m[1]) else nums
        stem = stem[:m.start()] + stem[m.end():]
    elif m := PART_WORD_IN_FILE.search(stem):          # "День Биби (часть 1)"
        parts = [int(m[1])]
        stem = stem[:m.start()] + stem[m.end():]
    elif m := PART_IN_NAME.match(stem):
        stem, parts = m[1], [int(m[2])]
    return LEADING_NUM.sub("", stem).strip(" .-_"), parts


def _episode_title(name: str) -> tuple[str, int | None]:
    """'Мягкая лапа смерти. Часть 2' → ('Мягкая лапа смерти', 2)"""
    if m := PART_IN_NAME.match(name):
        return m[1].strip(" .,-"), int(m[2])
    return name.strip(), None


def _title_map(files: list[tuple[str, int | None, bool]], tmdb: TMDB, show_id: int) -> dict:
    """Map files to TMDB episodes of one show by episode titles.

    files: (path, season guessed from names/folders, whether that season was explicit).
    Tries the file's own season first; when the season wasn't explicit, every season of the show.
    Returns {file: (season, episode, episode_end, tmdb_title)} — only unambiguous matches.
    """
    cache: dict[int, list[dict]] = {}
    all_seasons: list[int] | None = None

    def season_eps(n: int) -> list[dict]:
        if n not in cache:
            try:
                cache[n] = [{**e, "base": parse._norm(_episode_title(e["name"])[0]),
                             "part": _episode_title(e["name"])[1]} for e in tmdb.season_episodes(show_id, n)]
            except (ClientError, KeyError):
                cache[n] = []
        return cache[n]

    def seasons_of_show() -> list[int]:
        nonlocal all_seasons
        if all_seasons is None:
            try:
                all_seasons = sorted(n for n, c in tmdb.season_sizes(show_id).items() if n >= 1 and c)
            except ClientError:
                all_seasons = []
        return all_seasons

    found: dict[str, tuple[int, int, int | None, str]] = {}
    file_seasons = sorted({s for _, s, _ in files if s})
    for f, season, explicit in files:
        title, parts = _file_title(f)
        if not title or len(parse._norm(title)) < 3:
            continue
        order = ([season] if season else []) + [x for x in file_seasons if x != season]
        if not explicit:
            order += [x for x in seasons_of_show() if x not in order]
        for sn in order:
            hits = []
            for part in parts or [None]:
                scored = sorted(((parse.similarity(title, e["base"]), e) for e in season_eps(sn)
                                 if e["part"] == part or (e["part"] is None and part in (None, 1))),
                                key=lambda x: -x[0])
                # the closest title wins ("Бортовой дневник 2" vs "Бортовой дневник"), if it is clearly the closest
                if not scored or scored[0][0] < 0.8 or (len(scored) > 1 and scored[1][0] > scored[0][0] - 0.03):
                    break
                hits.append(scored[0][1])
            if hits and len(hits) == len(parts or [None]):
                end = hits[-1]["episode"] if len(hits) > 1 else None
                found[f] = (sn, hits[0]["episode"], end, hits[0]["name"])
                break
    # two files claiming one episode: leave both to the person, keep the rest
    claims: dict[tuple, list[str]] = {}
    for f, (s, e, _, _) in found.items():
        claims.setdefault((s, e), []).append(f)
    for fs in claims.values():
        if len(fs) > 1:
            for f in fs:
                found.pop(f)
    return found


def _match_episode_titles(episodes: list[dict], unmatched: list[str], tmdb: TMDB, show: dict,
                          manual: set[str], show_fixed: bool) -> tuple[int, int, dict | None]:
    """Prefer TMDB episode titles over numbers in file names (releases often use their own order).

    Returns (files matched, files whose season/episode changed, other show chosen instead or None).
    Applies the result only when most files match.
    """
    files = [(e["file"], e["season"], "сезон не найден" not in e["how"]) for e in episodes if e["file"] not in manual]
    files += [(f, None, False) for f in unmatched]
    if not files:
        return 0, 0, None
    chosen = show.get("chosen") or {}
    found = _title_map(files, tmdb, chosen["id"])
    switched = None
    any_explicit = any(explicit for _, _, explicit in files)
    if len(found) < 0.6 * len(files) and not show_fixed and not any_explicit:
        for c in (show.get("candidates") or [])[:8]:   # titles may belong to another show / season of it
            if c["id"] == chosen.get("id"):
                continue
            alt = _title_map(files, tmdb, c["id"])
            if len(alt) >= 0.8 * len(files) and len(alt) > len(found):
                found, switched = alt, c
                break
    if len(found) < 0.6 * len(files):
        return 0, 0, None
    changed = 0
    for e in episodes:
        if e["file"] in found:
            sn, ep, end, name = found[e["file"]]
            if (e["season"], e["episode"], e.get("episode_end")) != (sn, ep, end):
                changed += 1
            e.update(season=sn, episode=ep, how=f"по названию в TMDB: «{name}»")
            e.pop("episode_end", None)
            if end:
                e["episode_end"] = end
    for f in list(unmatched):
        if f in found:
            sn, ep, end, name = found[f]
            e = {"file": f, "season": sn, "episode": ep, "how": f"по названию в TMDB: «{name}»"}
            if end:
                e["episode_end"] = end
            episodes.append(e)
            unmatched.remove(f)
            changed += 1
    if changed >= 0.3 * len(found):
        # the release numbers its files in its own order: a number from a name that matched no title means nothing
        for e in [e for e in episodes if e["file"] not in found and e["file"] not in manual]:
            episodes.remove(e)
            unmatched.append(e["file"])
    if switched:
        show["chosen"], show["score"] = switched, max(show.get("score") or 0, 0.85)
    return len(found), changed, switched


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
