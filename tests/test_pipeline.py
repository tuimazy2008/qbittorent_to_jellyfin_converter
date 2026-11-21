import time

import pytest
from fastapi.testclient import TestClient

from jellyprep import main, pipeline
from conftest import fake_info

S = {"dry_run": False, "night_only": True, "night_start": "01:00", "night_end": "08:00"}
CAT = {"name": "movies", "delete_enabled": 1, "seed_ratio": 2.0, "seed_days": 30.0}
IDENT = {"kind": "movie", "confidence": 0.92, "problems": [],
         "movies": [{"files": ["a.mkv"], "chosen": {"id": 508, "title": "Реальная любовь", "year": 2003}}]}


def item(state, plan_actions=("hardlink",), **kw):
    now = time.time()
    plan = {"target": "/lib/movies/Реальная любовь (2003) [tmdbid-508]", "problems": [],
            "actions": [{"src": "/dl/a.mkv", "dst": "/lib/a.mkv", "action": a, "type": "video", "size": 10}
                        for a in plan_actions]}
    base = {"id": 1, "name": "Love.Actually.mkv", "category": "movies", "source": "torrent", "hash": "h" * 40,
            "state": state, "ident": IDENT, "plan": plan, "result": None, "error": None, "note": None,
            "ratio": 0.5, "seeding_time": 3 * 86400, "created_at": now - 100, "processed_at": None,
            "removed_at": None, "options": None, "trace": {"identified_at": now - 90, "decision": "уверенность 92%",
                                                           "approved_by": "auto", "approved_at": now - 90}}
    base.update(kw)
    return base


def stage(p, key):
    return next(s for s in p["stages"] if s["key"] == key)


def test_review_shows_problems():
    it = item("review", trace={"decision": "уверенность 60% < 80%"},
              ident={**IDENT, "confidence": 0.6, "problems": ["В TMDB у сериала нет сезона 2"]})
    p = pipeline.build(it, S, CAT, [])
    assert p["current"]["key"] == "review" and p["current"]["status"] == "wait"
    assert any("60%" in l for l in p["current"]["lines"]) and any("сезона 2" in l for l in p["current"]["lines"])


def test_waiting_for_night(monkeypatch):
    monkeypatch.setattr(pipeline.w, "in_window", lambda s, now=None: False)
    p = pipeline.build(item("approved", ("transcode",)), S, CAT, [])
    assert p["current"]["key"] == "prepare" and "ночного окна" in p["current"]["lines"][-1]
    assert stage(p, "review")["lines"][0].startswith("подтверждено автоматически")


def test_dry_run_and_queue_reasons(monkeypatch):
    monkeypatch.setattr(pipeline.w, "in_window", lambda s, now=None: True)
    assert "dry-run" in pipeline.build(item("approved", ("process",)), {**S, "dry_run": True}, CAT, [])["current"]["lines"][-1]
    assert "впереди задач: 2" in pipeline.build(item("approved", ("process",)), S, CAT, [], queue_ahead=2)["current"]["lines"][-1]
    assert "ждёт места" in pipeline.build(item("approved", ("process",), note="ждёт места на диске: …"), S, CAT, [])["current"]["lines"][-1]


def test_running_shows_progress():
    job = {"item_id": 1, "name": "x", "file": "S01E02.mkv", "n": 2, "total": 4, "percent": 50.0, "speed": "3x"}
    p = pipeline.build(item("running", ("process",) * 4), S, CAT, [], job=job)
    assert p["current"]["key"] == "prepare" and p["current"]["progress"] == pytest.approx(0.375)
    assert "файл 2 из 4" in p["current"]["lines"][-1]


def test_done_and_seeding():
    files = [{"src": "/dl/a.mkv", "dst": "/lib/a.mkv", "type": "video", "method": "transcode", "size": 18 * 1024**3,
              "out_size": 8 * 1024**3, "seconds": 3600}]
    p = pipeline.build(item("done", ("transcode",), result={"files": files}, processed_at=time.time()), S, CAT, [])
    prep = stage(p, "prepare")
    assert prep["status"] == "done" and "18.0 ГБ → 8.0 ГБ" in prep["lines"] and "заняло 1 ч 0 мин" in prep["lines"]
    assert p["current"]["key"] == "seeding" and p["current"]["progress"] == pytest.approx(0.25)


def test_removed_reason_and_folder_items():
    p = pipeline.build(item("removed", removed_at=time.time(),
                            trace={"removed_reason": "условие раздачи выполнено: рейтинг 2.10 ≥ 2"}), S, CAT, [])
    assert p["current"]["key"] == "removed" and "рейтинг 2.10" in p["current"]["lines"][0]
    folder = item("done", source="folder", hash=None, options={"delete_source": False})
    p = pipeline.build(folder, S, CAT, [])
    assert stage(p, "seeding")["status"] == "skip" and stage(p, "removed")["status"] == "skip"


def test_old_items_use_the_log():
    logs = [{"ts": 5.0, "message": "Торрент удалён вместе с файлами (рейтинг 10.17 ≥ 2)"},
            {"ts": 3.0, "message": "Одобрено вручную"}]
    p = pipeline.build(item("removed", trace=None, removed_at=6.0), S, CAT, logs)
    assert stage(p, "review")["lines"] == ["подтверждено вручную"] and stage(p, "review")["when"] == 3.0
    assert "рейтинг 10.17" in stage(p, "removed")["lines"][0]


def test_pages_render_pipeline(db):
    it = item("approved", ("process",))
    item_id = db.add_item(**{k: it[k] for k in ("name", "category", "source", "hash", "state", "ident", "plan")},
                          content_path="/dl/a.mkv")
    db.trace(item_id, decision="уверенность 92%", approved_by="auto")
    db.upsert_category("movies", enabled=1, delete_enabled=1, seed_ratio=2.0, seed_days=30.0)
    with TestClient(main.create_app(db, start_worker=False), base_url="http://jp.test") as c:
        assert "board-data" in c.get("/items").text
        rows = [r for g in c.get("/api/board").json()["groups"] for r in g["rows"]]
        assert rows[0]["id"] == item_id and len(rows[0]["cells"]) == 5
        page = c.get(f"/items/{item_id}").text
        assert "Путь задачи" in page and "подтверждено автоматически" in page
        panel = c.get(f"/items/{item_id}/panel").text
        assert "Путь задачи" in panel and "<html" not in panel
        r = c.post(f"/items/{item_id}/skip", headers={"Referer": "http://jp.test/items"}, follow_redirects=False)
        assert r.headers["location"] == f"/items#item-{item_id}"
