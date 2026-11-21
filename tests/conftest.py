from __future__ import annotations

from pathlib import Path

import pytest

from jellyprep.clients import Candidate
from jellyprep.db import DB

MB = 1024 * 1024


class FakeTMDB:
    """Tiny in-memory TMDB."""

    def __init__(self):
        self.movies = [
            Candidate(508, "Реальная любовь", "Love Actually", 2003, 50),
            Candidate(598, "Смерть ей к лицу", "Death Becomes Her", 1992, 30),
            Candidate(550776, "Надежда на счастье", "Hope Gap", 2019, 5),
            Candidate(10192, "Шрэк навсегда", "Shrek Forever After", 2010, 40),
        ]
        self.tv = [
            Candidate(95396, "Разделение", "Severance", 2022, 80),
            Candidate(4556, "Клиника", "Scrubs", 2001, 60),
            Candidate(12345, "Тайны следствия", "Тайны следствия", 2000, 3),
            Candidate(82701, "Царевны", "Царевны", 2018, 4),
        ]
        self.seasons = {95396: {1: 9, 2: 10}, 4556: {i: 25 for i in range(1, 10)}, 12345: {1: 4, 2: 4}, 82701: {0: 3, 1: 26, 2: 26, 3: 26}}

    def search(self, kind, query, year=None):
        from jellyprep.parse import similarity
        pool = self.movies if kind == "movie" else self.tv
        return [c for c in pool if max(similarity(query, c.title), similarity(query, c.original_title)) > 0.6
                and (year is None or c.year == year)]

    def candidate(self, kind, tmdb_id):
        return next(c for c in (self.movies if kind == "movie" else self.tv) if c.id == tmdb_id)

    def season_sizes(self, tmdb_id):
        return self.seasons.get(tmdb_id, {})

    def movie_card(self, tmdb_id):
        c = self.candidate("movie", tmdb_id)
        return {"id": c.id, "title": c.title, "original_title": c.original_title, "year": c.year, "runtime": 100,
                "directors": [], "cast": [], "companies": ["Мосфильм"] if c.id == 79580 else [], "countries": [],
                "overview": ""}

    episodes = {(12345, 1): ["Мягкая лапа смерти. Часть 1", "Мягкая лапа смерти. Часть 2",
                             "Гроб на две персоны. Часть 1", "Гроб на две персоны. Часть 2"],
                (12345, 2): ["Заказчик. Часть 1", "Заказчик. Часть 2", "Падение", "Марионетки"]}

    def season_episodes(self, tmdb_id, season):
        return [{"episode": i + 1, "name": n} for i, n in enumerate(self.episodes.get((tmdb_id, season), []))]


class FakeQBit:
    def __init__(self):
        self.torrents_list: list[dict] = []
        self.deleted: list[tuple[str, bool]] = []
        self.cats = {"movies": {"savePath": ""}, "tv-sonarr": {"savePath": ""}, "kids_series": {"savePath": ""}}

    def version(self):
        return "v5.0.0"

    def torrents(self):
        return [t for t in self.torrents_list if t["hash"] not in {h for h, _ in self.deleted}]

    def categories(self):
        return self.cats

    def default_save_path(self):
        return "/downloads"

    def delete(self, h, delete_files):
        self.deleted.append((h, delete_files))


def make_file(path: Path, size_mb: int = 25) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size_mb * MB)  # sparse file: big size, no real disk usage
    return path


@pytest.fixture
def tmdb():
    return FakeTMDB()


@pytest.fixture
def db():
    return DB(":memory:")


def fake_info(width=1920, height=1080, codec="h264", mbps=8.0, hdr=False, audio=(("rus", 6), ("eng", 6)),
              subs=(), duration=7200.0):
    return {"duration": duration, "size": int((mbps + 1) * 1e6 * duration / 8),
            "video": {"index": 0, "codec": codec, "width": width, "height": height,
                      "pix_fmt": "yuv420p10le" if hdr else "yuv420p", "bps": int(mbps * 1e6),
                      "hdr": hdr, "interlaced": False},
            "audio": [{"index": i + 1, "codec": "ac3", "channels": ch, "lang": lang, "title": "",
                       "default": i == 0, "bps": 448_000} for i, (lang, ch) in enumerate(audio)],
            "subs": [{"index": 10 + i, "codec": "subrip", "lang": lang, "title": "", "forced": False}
                     for i, lang in enumerate(subs)]}


from jellyprep import llm as _llm  # noqa: E402

REAL_DETECT_VISION = _llm._detect_vision


@pytest.fixture(autouse=True)
def _no_vision_probe(monkeypatch):
    """Tests never ask a real model server whether it can see images."""
    from jellyprep import llm
    monkeypatch.setattr(llm, "_detect_vision", lambda kind, url, model: False)
    llm._VISION.clear()
