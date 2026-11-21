import time

from jellyprep import board
from test_pipeline import CAT, S, item


def row(it, **kw):
    return board.item_row(it, S, CAT, [], kw.get("ahead", 0), kw.get("job"), kw.get("started"))


def statuses(r):
    return [c["status"] for c in r["cells"]]


def test_review_goes_to_decision_group():
    r = row(item("review", ident={**item("x")["ident"], "problems": ["Сезон 8: серия 19, а в TMDB всего 18"]}))
    assert r["group"] == "decision" and r["icon"] == "❓"
    assert statuses(r)[:3] == ["done", "wait", "todo"]
    assert "серия 19" in r["line"] and any(a.get("open") for a in r["actions"])


def test_running_has_progress_and_eta():
    # 76 files were finished before a restart: only the last 10 minutes of real work count for the ETA
    overall = (76 + 0.5) / 119
    job = {"item_id": 1, "name": "x", "file": "S06E04.mkv", "n": 77, "total": 119, "percent": 50.0, "speed": "6x",
           "overall": overall, "frac0": 76 / 119, "since": time.time() - 600}
    r = row(item("running", ("process",) * 119), job=job)
    assert r["group"] == "work" and statuses(r)[2] == "current"
    assert 0.64 < r["cells"][2]["progress"] < 0.65
    assert "осталось ~" in r["line"] and " ч " in r["line"]      # ~42.5 files left at 10 min per half file


def test_night_wait_offers_run_now(monkeypatch):
    monkeypatch.setattr(board.w, "in_window", lambda s, now=None: False)
    monkeypatch.setattr(board.pipeline.w, "in_window", lambda s, now=None: False)
    r = row(item("approved", ("transcode",)))
    assert statuses(r)[2] == "wait" and "ночного окна" in r["line"]
    assert r["actions"][0]["post"].endswith("/run-now")


def test_seeding_row_and_done_group():
    r = row(item("done", processed_at=time.time()))
    assert r["group"] == "seeding" and statuses(r)[:4] == ["done"] * 4 and statuses(r)[4] == "current"
    assert r["cells"][4]["progress"] == 0.25 and "удалится не позже чем через" in r["line"]
    gone = row(item("removed", removed_at=time.time(), trace={"removed_reason": "условие раздачи выполнено"}))
    assert gone["group"] == "done" and statuses(gone) == ["done"] * 5


def test_folder_item_skips_download_and_seeding():
    r = row(item("done", source="folder", hash=None))
    assert statuses(r)[0] == "skip" and r["group"] == "done"


def test_download_row_and_grouping():
    t = {"hash": "a" * 40, "name": "Dune.Part.Two.2024.mkv", "category": "movies", "progress": 0.63,
         "dlspeed": 4_000_000, "eta": 720, "state": "downloading"}
    b = board.build([item("review"), item("removed", removed_at=1.0)], [t], S, {"movies": CAT},
                    lambda i: [], {}, None, None)
    groups = {g["key"]: g for g in b["groups"]}
    dl = groups["work"]["rows"][0]
    assert dl["icon"] == "↓" and dl["cells"][0] == {"status": "current", "progress": 0.63} and "12 мин" in dl["line"]
    assert len(groups["decision"]["rows"]) == 1 and groups["done"]["collapsed"]
    assert b["counts"] == [1, 1, 0, 0, 0]


def test_thinking_model_line_and_cancel():
    from jellyprep import llm
    conv = llm.Conversation(model="google/gemma-4-12b-qat", prompt="p")
    conv.started -= 100
    conv.chunks = 1234
    it = item("review", note="модель думает…")
    r = board.item_row(it, {**S, "llm_kind": "openai", "llm_url": "u", "llm_model": "m"}, CAT, [], 0, None, None, conv)
    assert "думает 1 мин 40 с" in r["line"] and "1234" in r["line"]
    assert any(a.get("post", "").endswith("/llm-cancel") and a["label"] == "Отменить" for a in r["actions"])
    stale = board.item_row(it, {**S, "llm_kind": "openai", "llm_url": "u", "llm_model": "m"}, CAT, [], 0, None, None, None)
    assert any(a["label"] == "Сбросить" for a in stale["actions"])
