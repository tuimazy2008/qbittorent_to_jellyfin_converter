from fastapi.testclient import TestClient

from jellyprep import config, identify, main, media, parse, pipeline, worker as worker_mod
from jellyprep.clients import Candidate
from conftest import fake_info, make_file


def _neznayka(tmdb):
    tmdb.tv.append(Candidate(99952, "Незнайка на Луне", "Незнайка на Луне", 1997, 2))
    tmdb.seasons[99952] = {1: 12}


def test_episode_range():
    assert parse.episode_range("Незнайка на Луне_01-06.XVid.DVDRip.avi") == (1, 6)
    assert parse.episode_range("Film.1080p-2160p.mkv") is None
    assert parse.episode_range("Movie 1999-2005.mkv") is None


def test_film_not_found_but_series_is(tmp_path, tmdb):
    _neznayka(tmdb)
    root = tmp_path / "Neznayka"
    make_file(root / "Незнайка на Луне_01-06.XVid.DVDRip.avi")
    make_file(root / "Незнайка на Луне_07-12.XVid.DVDRip.avi")
    ident = identify.identify(media.scan(root), "movie", tmdb)
    assert ident["kind"] == "tv" and ident["show"]["chosen"]["id"] == 99952
    got = sorted((e["season"], e["episode"], e.get("episode_end")) for e in ident["episodes"])
    assert got == [(1, 1, 6), (1, 7, 12)]
    assert ident["problems"][0].startswith(identify.SERIES_PROBLEM)


def test_category_hint():
    cats = {"movies": {"name": "movies", "kind": "movies", "enabled": 1},
            "movies/kids": {"name": "movies/kids", "kind": "movies", "enabled": 1},
            "adult_series": {"name": "adult_series", "kind": "tv", "enabled": 1},
            "kids_series": {"name": "kids_series", "kind": "tv", "enabled": 1}}
    assert worker_mod.category_hint("tv", cats["movies/kids"], cats) == "kids_series"
    assert worker_mod.category_hint("tv", cats["movies"], cats) in ("adult_series", "kids_series")
    assert worker_mod.category_hint("movie", cats["movies"], cats) is None


def _setup(db, tmp_path, monkeypatch, tmdb):
    _neznayka(tmdb)
    root = tmp_path / "media" / "Neznayka"
    make_file(root / "Незнайка на Луне_01-06.XVid.DVDRip.avi")
    make_file(root / "Незнайка на Луне_07-12.XVid.DVDRip.avi")
    monkeypatch.setattr(worker_mod, "tmdb_client", lambda s: tmdb)
    monkeypatch.setattr(worker_mod, "safe_probe", lambda p: fake_info(codec="mpeg4", audio=(("rus", 2),)))
    config.save(db, {"library_root": str(tmp_path / "lib")})
    db.upsert_category("movies/kids", enabled=1, kind="movies", library_dir="movies_kids", approval="auto")
    db.upsert_category("kids_series", enabled=1, kind="tv", library_dir="kids_series", approval="auto")
    item_id = db.add_item(source="folder", name="Neznayka", category="movies/kids", content_path=str(root))
    w = worker_mod.Worker(db)
    w.analyze(item_id)
    return w, item_id


def test_series_in_films_category_waits_with_a_suggestion(db, tmp_path, monkeypatch, tmdb):
    w, item_id = _setup(db, tmp_path, monkeypatch, tmdb)
    item = db.item(item_id)
    assert item["state"] == "review"                                  # even in an "auto" category
    assert item["ident"]["category_hint"] == "kids_series"
    assert item["plan"]["target"].endswith("movies_kids/Незнайка на Луне (1997) [tmdbid-99952]")
    with TestClient(main.create_app(db, start_worker=False), base_url="http://jp.test") as c:
        html = c.get(f"/items/{item_id}/panel").text
        assert "★ подходит лучше" in html and "Если подтвердить" in html and "Разложить и перекодировать" in html
        c.post(f"/items/{item_id}/category", data={"category": "kids_series"})
    item = db.item(item_id)
    assert item["category"] == "kids_series" and "category_hint" not in item["ident"]
    assert item["plan"]["target"].endswith("kids_series/Незнайка на Луне (1997) [tmdbid-99952]")
    names = sorted(a["dst"].rsplit("/", 1)[-1] for a in item["plan"]["actions"])
    assert names == ["Незнайка на Луне S01E01-E06.mkv", "Незнайка на Луне S01E07-E12.mkv"]


def test_keep_category(db, tmp_path, monkeypatch, tmdb):
    w, item_id = _setup(db, tmp_path, monkeypatch, tmdb)
    with TestClient(main.create_app(db, start_worker=False), base_url="http://jp.test") as c:
        c.post(f"/items/{item_id}/category", data={"keep": "1"})
    item = db.item(item_id)
    assert item["category"] == "movies/kids" and "category_hint" not in item["ident"]


def test_confirm_summary_lines(db, tmp_path, monkeypatch, tmdb):
    w, item_id = _setup(db, tmp_path, monkeypatch, tmdb)
    item = db.item(item_id)
    summ = pipeline.confirm_summary(item, {**config.load(db), "night_only": True}, db.categories()["movies/kids"])
    text = "\n".join(summ["lines"])
    assert "Незнайка на Луне" in text and "перекодировать 2" in text and "ночью" in text
    assert "оригиналы в" in text and summ["button"] == "Разложить и перекодировать ночью 2 файла"
