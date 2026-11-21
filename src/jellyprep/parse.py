"""Name parsing: titles, seasons, episodes. guessit plus rules for common Russian release names."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import PurePath

from guessit import guessit

SXE = re.compile(r"(?i)(?<![a-z\d])S(\d{1,2})[ ._-]?E(\d{1,3})(?!\d)")
NXN = re.compile(r"(?i)(?<![\d])(\d{1,2})x(\d{1,3})(?!\d)")
EP_AFTER = re.compile(r"(?i)(?<!\d)(\d{1,3})[ ._-]*(?:серия|эпизод|episode|ep)(?!\w)")
EP_BEFORE = re.compile(r"(?i)(?<![^\W\d_])(?:серия|эпизод|episode|ep|e)[ ._#-]*(\d{1,3})(?!\d)")
LEADING = re.compile(r"^(\d{1,3})(?:[ ._)\-]|$)")

SEASON_DIR = [
    re.compile(r"(?i)season[ ._-]*(\d{1,2})(?!\d)"),
    re.compile(r"(?i)(?<!\d)(\d{1,2})[ ._-]*(?:-?(?:й|ый|ой))?[ ._-]*сезон"),
    re.compile(r"(?i)сезон[ ._-]*(\d{1,2})(?!\d)"),
    re.compile(r"(?i)(?<![a-z\d])S(\d{1,2})(?![\dE])"),
]
TRAILING_NUM = re.compile(r"^(.*?)[ ._-]+(\d{1,2})$")
BRACKETED = re.compile(r"[\(\[]([^\)\]]+)[\)\]]")
JUNK_WORDS = re.compile(r"(?i)rutracker|kinozal|rip|web-?dl|1080|720|2160|hdr|season|сезон|серии|"
                        r"quality|dvd|bd|x26[45]|hevc|avc|ntsc|pal")


def _first(v):
    return v[0] if isinstance(v, list) and v else v


def parse_episode(filename: str) -> tuple[int | None, int | None, str]:
    """Return (season, episode, how) parsed from a file name."""
    stem = PurePath(filename).stem
    if m := SXE.search(stem):
        return int(m[1]), int(m[2]), "SxxEyy"
    if m := NXN.search(stem):
        return int(m[1]), int(m[2]), "NxM"
    if m := EP_AFTER.search(stem):
        return None, int(m[1]), "«N серия»"
    if m := EP_BEFORE.search(stem):
        return None, int(m[1]), "«серия N»"
    g = guessit(stem, {"type": "episode"})
    season, episode = _first(g.get("season")), _first(g.get("episode"))
    if isinstance(season, int) and not 0 <= season < 100:
        season = None
    if isinstance(episode, int) and 0 < episode < 1000:
        return season, episode, "guessit"
    if m := LEADING.match(stem):
        return None, int(m[1]), "номер в начале имени"
    return None, None, ""


EP_RANGE = re.compile(r"(?<![\d.])(\d{1,3})\s*[-–]\s*(\d{1,3})(?!\d|p\b)")


def episode_range(filename: str) -> tuple[int, int] | None:
    """'Незнайка на Луне_01-06.avi' → (1, 6): several episodes in one file."""
    stem = PurePath(filename).stem
    for m in EP_RANGE.finditer(stem):
        a, b = int(m[1]), int(m[2])
        if 0 < a < b and b - a < 40 and not (1900 <= a <= 2100):
            return a, b
    return None


def season_from_name(name: str) -> int | None:
    for rx in SEASON_DIR:
        if m := rx.search(name):
            n = int(m[1])
            if 0 <= n < 100:
                return n
    return None


def season_from_dirs(rel_file: str, sibling_dirs: dict[str, list[str]]) -> int | None:
    """Season from the file's sub-directories inside the torrent (deepest first).

    Handles "Show/Show-2/…": a trailing number is a season when a sibling dir has the
    same base name; the sibling without a number is season 1.
    """
    parts = PurePath(rel_file).parts[:-1]
    for depth in range(len(parts) - 1, -1, -1):
        name = parts[depth]
        if (n := season_from_name(name)) is not None:
            return n
        parent = "/".join(parts[:depth])
        siblings = sibling_dirs.get(parent, [])
        if m := TRAILING_NUM.match(name):
            if any(_norm(s) == _norm(m[1]) for s in siblings):
                return int(m[2])
        elif any((sm := TRAILING_NUM.match(s)) and _norm(sm[1]) == _norm(name) for s in siblings):
            return 1
    return None


def guess_title(name: str) -> tuple[str, int | None, list[str]]:
    """Return (title, year, alternative queries) for a torrent/file name."""
    g = guessit(name)
    title = str(_first(g.get("title")) or "").strip()
    year = _first(g.get("year"))
    if not (isinstance(year, int) and 1900 <= year <= 2100):
        year = None
    alts = []
    for m in BRACKETED.finditer(name):
        text = m[1].strip()
        if len(re.sub(r"[\W\d_]", "", text)) >= 3 and not JUNK_WORDS.search(text):
            alts.append(text)
    for key in ("alternative_title", "episode_title"):
        if (v := _first(g.get(key))) and len(re.sub(r"[\W\d_]", "", str(v))) >= 3:
            alts.append(str(v))
    if not title:
        title = re.sub(r"[._]+", " ", PurePath(name).stem).strip()
    # "Show.2001.DVDRip-kvvk_2009": guessit takes the release group's year and leaves the real one in the title
    if year and re.search(rf"[A-Za-z]{{2,}}[_-]{year}(?!\d)", name) and (m := TRAILING_YEAR.search(title)):
        title, year = title[:m.start()].strip(), int(m[1])
    return title, year, [a for a in dict.fromkeys(alts) if _norm(a) != _norm(title)]


TRAILING_YEAR = re.compile(r"(?<=\S)[ ._(]+((?:19|20)\d\d)\)?$")


HINT_JUNK = re.compile(r"(?i)rip|films?\.ws|\bhq\b|rutracker|kinozal|torrent|www\.|\.(ws|ru|org|com|net)\b|=-|-=|"
                       r"\bby\b|релиз|release")


def hints_from_info(info: dict | None) -> list[str]:
    """Titles written inside the file (container title, video track title) that look like a real name."""
    out = []
    for raw in ((info or {}).get("title") or "", ((info or {}).get("video") or {}).get("title") or ""):
        raw = raw.strip()
        if not raw:
            continue
        title = guess_title(raw)[0]
        if len(re.sub(r"[\W\d_]", "", title)) < 4 or HINT_JUNK.search(raw.split(".")[0] if "." in raw else raw):
            continue
        out.append(raw)
    return out


GENERIC_NAME = re.compile(
    r"(?i)^\W*(season|сезон|series|serie|disc|disk|dvd|cd|extras?|bonus|torrent|video|movies?|films?|"
    r"сериалы?|фильмы?|s)?[\W_]*\d{0,3}[\W_]*(сезон|season)?\W*$")


def is_generic_name(name: str) -> bool:
    """'Season 01', 'Сезон 2', 'Disc 1', 'S03' — a folder name that says nothing about the show."""
    return bool(GENERIC_NAME.match(PurePath(name).stem if "." in name[-5:] else name))


def skeleton(text: str) -> str:
    """Consonants only: 'Life or Something Like It' and 'LF_R_SMTHNG_LK_T' both give 'lfrsmthnglkt'."""
    return re.sub(r"[^a-z]|[aeiouy]", "", text.lower())


def disemvoweled(name: str) -> str | None:
    """The title part of a file name written without vowels ('LF_R_SMTHNG_LK_T_2002_720p…'), else None."""
    head = re.split(r"[ ._-](?:19|20)\d\d(?:[ ._-]|$)", PurePath(name).stem)[0]
    letters = re.sub(r"[^A-Za-z]", "", head)
    if len(letters) >= 6 and not re.search(r"[AEIOUaeiou]", letters):
        return head
    return None


def episode_title_from_name(filename: str) -> str:
    """'S01E01 - The Pants Tent.mkv' → 'The Pants Tent' ('' when there's no title after the number)."""
    stem = PurePath(filename).stem
    m = re.search(r"(?i)(?:s\d{1,2}[ ._-]?e\d{1,3}|\d{1,2}x\d{1,3})[ ._-]*(.*)$", stem)
    rest = (m[1] if m else "").strip(" .-_")
    rest = re.split(r"(?i)(?:^|[ ._\[(]+)(?:\d{3,4}p|web|bdrip|bluray|hdtv|dvdrip|x26[45]|rus|eng)\b", rest)[0]
    return rest.replace(".", " ").strip() if len(re.sub(r"[\W\d_]", "", rest)) >= 3 else ""


def guess_kind(name: str) -> str:
    return "tv" if guessit(name).get("type") == "episode" else "movie"


def _norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[\W_]+", " ", s)
    return " ".join(s.split())


def similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()
