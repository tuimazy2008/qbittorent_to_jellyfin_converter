import os

from jellyprep import config, survey, transcode, worker as worker_mod
from conftest import FakeQBit, fake_info, make_file


def setup(db, tmp_path, monkeypatch):
    qb = FakeQBit()
    monkeypatch.setattr(worker_mod, "qbit_client", lambda s: qb)
    monkeypatch.setattr(survey, "qbit_client", lambda s: qb)
    config.save(db, {"library_root": str(tmp_path / "library"), "voice_track": True, "keep_langs": ["rus", "eng"],
                     "path_map": [{"from": "/media", "to": str(tmp_path / "media")}]})
    db.upsert_category("movies", enabled=1, kind="movies", library_dir="movies")
    w = worker_mod.Worker(db)
    return w, qb, survey.Surveys(w)


def test_old_folder_counts(db, tmp_path, monkeypatch):
    w, qb, sv = setup(db, tmp_path, monkeypatch)
    old = tmp_path / "media" / "movies" / "adult_movies"
    make_file(old / "Death Becomes Her.mkv", 30)
    make_file(old / "Hope Gap.mkv", 40)
    make_file(old / "Known.mkv")
    make_file(old / "notes" / "readme.txt", 0)
    qb.torrents_list = [{"hash": "h" * 40, "name": "Hope Gap.mkv", "category": "movies",
                         "content_path": "/media/movies/adult_movies/Hope Gap.mkv"}]
    db.add_item(source="folder", name="Known.mkv", category="movies", content_path=str(old / "Known.mkv"))
    r = sv.survey(str(old), "old")
    st = {e["name"]: e["status"] for e in r["entries"]}
    assert st == {"Death Becomes Her.mkv": "move", "Hope Gap.mkv": "seeding", "Known.mkv": "known"}
    assert r["totals"]["move"] == 1 and r["totals"]["size_move"] == 30 * 1024 * 1024


def test_new_library_counts_and_probe_cache(db, tmp_path, monkeypatch):
    w, qb, sv = setup(db, tmp_path, monkeypatch)
    lib = tmp_path / "library" / "movies"
    needs = make_file(lib / "A (2001)" / "A (2001).mkv")
    done = make_file(lib / "B (2002)" / "B (2002).mkv")
    big = make_file(lib / "C (2003)" / "C (2003).mkv")
    finished = fake_info(audio=(("rus", 2),))
    finished["audio"][0]["title"] = transcode.VOICE_TITLE
    infos = {str(needs): fake_info(audio=(("rus", 6),)), str(done): finished, str(big): fake_info(3840, 2160, audio=(("rus", 2),))}
    infos[str(big)]["audio"][0]["title"] = transcode.VOICE_TITLE
    calls = []
    monkeypatch.setattr(transcode, "probe", lambda p: calls.append(str(p)) or infos[str(p)])
    r = sv.survey(str(lib), "new")
    st = {e["name"]: (e["status"], e.get("voice"), e.get("heavy")) for e in r["entries"]}
    assert st == {"A (2001)": ("work", 1, 0), "B (2002)": ("ok", 0, 0), "C (2003)": ("work", 0, 1)}
    assert r["totals"]["work"] == 2 and r["totals"]["voice"] == 1 and r["totals"]["heavy"] == 1
    sv.survey(str(lib), "new")
    assert len(calls) == 3                       # second survey used the cache
    os.utime(needs, (1, 1))
    sv.survey(str(lib), "new")
    assert len(calls) == 4                       # a changed file is probed again


def test_import_only_selected_and_in_place_reprocessing(db, tmp_path, monkeypatch):
    w, qb, sv = setup(db, tmp_path, monkeypatch)
    lib = tmp_path / "library" / "movies"
    make_file(lib / "A (2001)" / "A (2001).mkv")
    make_file(lib / "B (2002)" / "B (2002).mkv")
    db.add_item(source="folder", name="A (2001)", category="movies", content_path=str(lib / "A (2001)"), state="done")
    res = w.import_folder(str(lib), "movies", only={"A (2001)"}, in_place=True)
    assert res["added"] == ["A (2001)"]          # finished earlier, but needs work again
    res = w.import_folder(str(lib), "movies", only={"B (2002)"}, in_place=False)
    assert res["added"] == ["B (2002)"]


def test_import_page_sections(db, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from jellyprep import main
    setup(db, tmp_path, monkeypatch)
    (tmp_path / "library" / "movies").mkdir(parents=True)
    with TestClient(main.create_app(db, start_worker=False), base_url="http://jp.test") as c:
        html = c.get("/import").text
    assert "Старые папки → новая библиотека" in html and "Доделать новую библиотеку" in html


def test_suggestion_prefers_folder_name_over_jellyfin_type():
    from jellyprep.main import _suggest_category
    cats = [{"name": "movies", "kind": "movies", "enabled": 1}, {"name": "adult_series", "kind": "tv", "enabled": 1},
            {"name": "movies/kids", "kind": "movies", "enabled": 1}, {"name": "kids_series", "kind": "tv", "enabled": 1}]
    assert _suggest_category({"path": "/media/movies/series", "type": "movies"}, cats) == "adult_series"
    assert _suggest_category({"path": "/media/movies/kids_movies", "type": "movies"}, cats) == "movies/kids"
    assert _suggest_category({"path": "/media/movies/kids_series", "type": "tvshows"}, cats) == "kids_series"
    assert _suggest_category({"path": "/media/movies/adult_movies", "type": "movies"}, cats) == "movies"
