from jellyprep import config, identify, llm, media, parse, research, worker as worker_mod
from jellyprep.clients import Candidate
from conftest import fake_info, make_file


def test_names_from_articles():
    res = [{"title": "Curb Your Enthusiasm (season 1)"}, {"title": "List of Curb Your Enthusiasm episodes"},
           {"title": "Список серий телесериала «Клиника»"}, {"title": "Larry David"}]
    names = research.names_from_articles(res)
    assert names[0] == "Curb Your Enthusiasm" and "Larry David" in names
    assert research.quoted(["The Pants Tent", "Ted And Mary"]) == '"The Pants Tent" "Ted And Mary"'


def test_episode_title_from_name():
    assert parse.episode_title_from_name("S01E01 - The Pants Tent.mkv") == "The Pants Tent"
    assert parse.episode_title_from_name("Severance.S01E01.1080p.BluRay.mkv") == ""
    assert parse.episode_title_from_name("Scrubs.S09E01 Our First Day of School [WEB-DL 1080p].mkv") == "Our First Day of School"


def test_series_found_by_searching(db, tmp_path, monkeypatch, tmdb):
    tmdb.tv += [Candidate(4546, "Умерь свой энтузиазм", "Curb Your Enthusiasm", 2000, 30)]
    tmdb.seasons[4546] = {1: 10}
    tmdb.tv_card = lambda tid: {"id": tid, "title": "Умерь свой энтузиазм", "original_title": "Curb Your Enthusiasm",
                                "year": 2000, "creators": ["Larry David"], "networks": ["HBO"], "cast": [], "countries": []}
    root = tmp_path / "media" / "Season 01"
    for name in ("S01E01 - The Pants Tent.mkv", "S01E02 - Ted And Mary.mkv"):
        make_file(root / name)
    monkeypatch.setattr(worker_mod, "tmdb_client", lambda s: tmdb)
    monkeypatch.setattr(worker_mod, "safe_probe", lambda p: fake_info(duration=1700, audio=(("eng", 2),)))
    searched = []
    monkeypatch.setattr(research, "wiki_search", lambda q, lang="en", limit=4: searched.append(q) or (
        [{"lang": "en", "title": "Curb Your Enthusiasm (season 1)", "snippet": "episodes The Pants Tent, Ted and Mary"}]
        if lang == "en" else []))
    asks = []

    def fake_ask(s, prompt, conv=None, images=None, schema=None, system=None):
        conv.status, conv.system = "ok", system or llm.SYSTEM
        asks.append(schema)
        if schema is llm.QUERIES_SCHEMA:
            return {"queries": ["The Pants Tent episode"]}
        return {"tmdb_id": 4546, "comment": "Википедия: это 1-й сезон Curb Your Enthusiasm"}

    monkeypatch.setattr(llm, "ask", fake_ask)
    config.save(db, {"llm_kind": "openai", "llm_url": "http://lm/v1", "llm_model": "m", "llm_auto": False,
                     "llm_search": True, "library_root": str(tmp_path / "lib")})
    db.upsert_category("adult_series", enabled=1, kind="tv")
    item_id = db.add_item(source="folder", name="Season 01", category="adult_series", content_path=str(root))
    w = worker_mod.Worker(db)
    w.analyze(item_id)
    w.ask_llm(item_id, wait=True)
    item = db.item(item_id)
    assert '"The Pants Tent" "Ted And Mary"' in searched                       # our own query goes first
    assert asks == [llm.QUERIES_SCHEMA, llm.CHOOSE_SCHEMA]                     # no vision: straight to search
    assert item["ident"]["show"]["chosen"]["id"] == 4546
    assert "нашла через поиск" in item["note"] and item["state"] == "review"


def test_search_off_keeps_the_old_behaviour(db, tmp_path, monkeypatch, tmdb):
    root = tmp_path / "media" / "Season 01"
    make_file(root / "S01E01 - The Pants Tent.mkv")
    monkeypatch.setattr(worker_mod, "tmdb_client", lambda s: tmdb)
    monkeypatch.setattr(worker_mod, "safe_probe", lambda p: fake_info(duration=1700))
    monkeypatch.setattr(research, "wiki_search", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no search")))
    monkeypatch.setattr(llm, "ask", lambda s, prompt, conv=None, **kw: (setattr(conv, "status", "ok"),
                        {"kind": "tv", "tmdb_id": None, "title": "", "episodes": [], "comment": ""})[1])
    config.save(db, {"llm_kind": "openai", "llm_url": "http://lm/v1", "llm_model": "m", "llm_auto": False,
                     "llm_search": False})
    db.upsert_category("adult_series", enabled=1, kind="tv")
    item_id = db.add_item(source="folder", name="Season 01", category="adult_series", content_path=str(root))
    w = worker_mod.Worker(db)
    w.analyze(item_id)
    w.ask_llm(item_id, wait=True)
    assert "модель" in db.item(item_id)["note"]
