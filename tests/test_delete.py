import os

import pytest
from fastapi.testclient import TestClient

from jellyprep import config, main, worker as worker_mod
from conftest import FakeQBit, fake_info, make_file


@pytest.fixture
def env(db, tmp_path, monkeypatch):
    qb = FakeQBit()
    monkeypatch.setattr(worker_mod, "qbit_client", lambda s: qb)
    config.save(db, {"library_root": str(tmp_path / "library")})
    db.upsert_category("movies", enabled=1, kind="movies")
    return worker_mod.Worker(db), qb, tmp_path


def test_delete_folder_item(env, db):
    w, qb, tmp = env
    root = tmp / "media" / "old" / "Neznayka"
    make_file(root / "a.avi", 25)
    make_file(root / "b.avi", 25)
    item_id = db.add_item(source="folder", name="Neznayka", category="movies", content_path=str(root), state="review")
    prev = w.delete_preview(item_id)
    assert len(prev["files"]) == 2 and prev["torrent"] is None and not prev["refusal"]
    w.delete_item(item_id)
    assert not root.exists()
    item = db.item(item_id)
    assert item["state"] == "deleted" and "удалено вручную" in item["note"]


def test_delete_torrent_goes_through_qbittorrent(env, db):
    w, qb, tmp = env
    f = make_file(tmp / "media" / "dl" / "Some.Film.mkv")
    item_id = db.add_item(hash="h" * 40, name="Some.Film.mkv", category="movies", content_path=str(f), state="review")
    w.delete_item(item_id)
    assert qb.deleted == [("h" * 40, True)]
    assert f.exists()          # files are removed by qBittorrent itself, not by jellyprep


def test_delete_with_library_copy(env, db):
    w, qb, tmp = env
    src = make_file(tmp / "media" / "dl" / "Film.mkv")
    lib = tmp / "library" / "movies" / "Film (2000)"
    dst = make_file(lib / "Film (2000).mkv")
    item_id = db.add_item(hash="k" * 40, name="Film.mkv", category="movies", content_path=str(src), state="done",
                          plan={"target": str(lib), "actions": []},
                          result={"files": [{"src": str(src), "dst": str(dst), "method": "transcode", "type": "video", "size": 1}]})
    assert len(w.delete_preview(item_id)["library"]) == 1
    w.delete_item(item_id, with_library=True)
    assert not dst.exists() and not lib.exists()


def test_refusals(env, db):
    w, qb, tmp = env
    lib_root = tmp / "library"
    lib_root.mkdir(parents=True, exist_ok=True)
    a = db.add_item(source="folder", name="x", category="movies", content_path=str(lib_root), state="review")
    assert "не отдельная раздача" in w.delete_preview(a)["refusal"]
    b = db.add_item(source="folder", name="y", category="movies", content_path=str(tmp / "media" / "y"), state="running")
    with pytest.raises(ValueError):
        w.delete_item(b)
    c = db.add_item(source="folder", name="z", category="movies", content_path=str(lib_root / "movies" / "Z (2000)"),
                    state="review")
    assert "новой библиотеки" in w.delete_preview(c)["refusal"]


def test_delete_via_page(env, db, monkeypatch):
    w, qb, tmp = env
    root = tmp / "media" / "old" / "Junk"
    make_file(root / "junk.avi")
    item_id = db.add_item(source="folder", name="Junk", category="movies", content_path=str(root), state="review",
                          ident={"kind": "movie", "confidence": 0, "problems": [], "movies": []},
                          plan={"actions": [], "problems": [], "target": None})
    app = main.create_app(db, start_worker=False)
    app.state.worker.delete_item = w.delete_item
    with TestClient(app, base_url="http://jp.test") as c:
        html = c.get(f"/items/{item_id}/panel").text
        assert "Удалить раздачу" in html and "Да, удалить 1 файл" in html
        c.post(f"/items/{item_id}/delete", headers={"Referer": "http://jp.test/items"})
    assert db.item(item_id)["state"] == "deleted" and not root.exists()
