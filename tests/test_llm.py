import json

import pytest

from jellyprep import llm

FILES = ["Тайны следствия/1. Мягкая лапа смерти (1 сер.).avi", "Тайны следствия/1. Мягкая лапа смерти (2 сер.).avi",
         "Тайны следствия-2/1. Заказчик (1,2 сер.).avi"]


def test_answer_to_overrides():
    answer = {"kind": "tv", "tmdb_id": 76713, "title": "Тайны следствия", "year": 2000, "comment": "по названиям",
              "episodes": [{"file": FILES[0], "season": 1, "episode": 1},
                           {"file": "1. Мягкая лапа смерти (2 сер.).avi", "season": 1, "episode": 2},  # basename only
                           {"file": FILES[2], "season": 2, "episode": 1, "episode_end": 2},
                           {"file": "invented.avi", "season": 3, "episode": 1}]}
    ov, comment = llm.to_overrides(answer, FILES, {76713})
    assert ov["show"] == {"tmdb_id": 76713} and ov["episodes_how"] == "модель"
    assert ov["episodes"] == {FILES[0]: [1, 1, None], FILES[1]: [1, 2, None], FILES[2]: [2, 1, 2]}
    assert comment == "по названиям"


def test_unknown_tmdb_id_becomes_manual_title_and_duplicates_are_refused():
    ans = {"kind": "tv", "tmdb_id": 999, "title": "Сериал", "year": 2001, "comment": "",
           "episodes": [{"file": FILES[0], "season": 1, "episode": 1}]}
    ov, _ = llm.to_overrides(ans, FILES, {76713})
    assert ov["show"] == {"title": "Сериал", "year": 2001}
    ans["episodes"].append({"file": FILES[1], "season": 1, "episode": 1})
    with pytest.raises(llm.LLMError):
        llm.to_overrides(ans, FILES, set())


def test_parse_answer_strips_thinking():
    assert llm.parse_answer('<think>hmm</think>\n{"kind": "movie"}') == {"kind": "movie"}
    with pytest.raises(llm.LLMError):
        llm.parse_answer("no json here")


def test_prompt_contains_files_candidates_and_titles():
    p = llm.build_prompt("Тайны следствия", FILES, "tv",
                         [{"id": 76713, "title": "Тайны следствия", "original_title": "Тайны следствия", "year": 2000}],
                         {76713: {1: [{"episode": 1, "name": "Мягкая лапа смерти. Часть 1"}]}})
    assert FILES[2] in p and "id 76713" in p and "E1 Мягкая лапа смерти. Часть 1" in p


class FakeStream:
    def __init__(self, lines, sent):
        self.lines, self.sent = lines, sent

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        yield from self.lines


def sse(**delta):
    return "data: " + json.dumps({"choices": [{"delta": delta}]})


ANSWER = json.dumps({"kind": "movie", "tmdb_id": None, "title": "X", "episodes": [], "comment": "ok"})


def test_ask_streams_answer_and_reasoning(monkeypatch):
    sent = {}
    lines = [sse(reasoning_content="Смотрю на файлы. "), sse(reasoning_content="Это фильм."),
             sse(content=ANSWER[:20]), sse(content=ANSWER[20:]),
             "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": 42}}), "data: [DONE]"]

    def stream(method, url, **kw):
        sent.update(url=url, **kw)
        return FakeStream(lines, sent)

    monkeypatch.setattr(llm.httpx, "stream", stream)
    s = {"llm_kind": "openai", "llm_url": "http://lm:1234", "llm_model": "qwen/qwen3-14b", "llm_key": ""}
    conv = llm.Conversation(model="qwen/qwen3-14b", prompt="prompt")
    assert llm.ask(s, "prompt", conv)["title"] == "X"
    assert sent["url"] == "http://lm:1234/v1/chat/completions" and sent["json"]["stream"] is True
    assert sent["json"]["response_format"]["type"] == "json_schema"
    assert conv.status == "ok" and conv.reasoning == "Смотрю на файлы. Это фильм." and conv.content == ANSWER
    assert conv.usage == {"completion_tokens": 42} and conv.finished
    d = conv.as_dict()
    assert "cancel" not in d and d["system"].startswith("You map video files")


def test_ask_can_be_cancelled(monkeypatch):
    conv = llm.Conversation(model="m", prompt="p")

    def lines():
        yield sse(reasoning_content="думаю…")
        conv.cancel.set()
        yield sse(reasoning_content="ещё думаю…")

    monkeypatch.setattr(llm.httpx, "stream", lambda *a, **kw: FakeStream(lines(), {}))
    with pytest.raises(llm.LLMCancelled):
        llm.ask({"llm_kind": "openai", "llm_url": "http://lm", "llm_model": "m"}, "p", conv)
    assert conv.status == "cancelled" and conv.reasoning == "думаю…"


def test_ollama_stream_and_think_tags(monkeypatch):
    lines = [json.dumps({"message": {"content": "<think>хм</think>" + ANSWER}}),
             json.dumps({"done": True, "eval_count": 7, "prompt_eval_count": 3, "message": {}})]
    monkeypatch.setattr(llm.httpx, "stream", lambda *a, **kw: FakeStream(lines, {}))
    conv = llm.Conversation(model="m", prompt="p")
    assert llm.ask({"llm_kind": "ollama", "llm_url": "http://o:11434", "llm_model": "m"}, "p", conv)["kind"] == "movie"
    assert conv.reasoning == "хм" and conv.usage["completion_tokens"] == 7


def test_worker_applies_model_answer_for_review(tmp_path, db, tmdb, monkeypatch):
    from jellyprep import config, worker as worker_mod
    from conftest import make_file
    root = tmp_path / "media" / "Непонятное"
    a = make_file(root / "a.avi")
    b = make_file(root / "b.avi")
    monkeypatch.setattr(worker_mod, "tmdb_client", lambda s: tmdb)
    monkeypatch.setattr(worker_mod, "safe_probe", lambda p: None)
    def fake_ask(s, prompt, conv=None):
        conv.reasoning, conv.status = "подумала", "ok"
        return {
        "kind": "tv", "tmdb_id": None, "title": "Разделение", "year": 2022, "comment": "угадала",
        "episodes": [{"file": "a.avi", "season": 1, "episode": 1}, {"file": "b.avi", "season": 1, "episode": 2}]}

    monkeypatch.setattr(llm, "ask", fake_ask)
    config.save(db, {"llm_kind": "openai", "llm_url": "http://lm:1234/v1", "llm_model": "m", "llm_auto": False,
                     "library_root": str(tmp_path / "lib")})
    db.upsert_category("tv", enabled=1, kind="tv", approval="auto_if_confident")
    item_id = db.add_item(source="folder", name="Непонятное", category="tv", content_path=str(root))
    w = worker_mod.Worker(db)
    w.analyze(item_id)
    assert db.item(item_id)["state"] == "review"
    w.ask_llm(item_id, wait=True)
    item = db.item(item_id)
    assert item["state"] == "review"                           # model answers always wait for confirmation
    assert [(e["season"], e["episode"], e["how"]) for e in item["ident"]["episodes"]] == [(1, 1, "модель"), (1, 2, "модель")]
    assert item["ident"]["show"]["chosen"]["title"] == "Разделение"
    assert "предложение модели" in item["note"] and item["trace"]["llm_comment"] == "угадала"
    conv = item["trace"]["llm_log"][0]
    assert conv["reasoning"] == "подумала" and "a.avi" in conv["prompt"] and conv["model"] == "m"
    assert not w.llm_jobs


def test_restart_resets_stale_thinking_label(db):
    from jellyprep import worker as worker_mod
    item_id = db.add_item(source="folder", name="x", category="tv", content_path="/x", state="review")
    db.execute("UPDATE items SET note=? WHERE id=?", (worker_mod.LLM_THINKING, item_id))
    worker_mod.Worker(db).recover()
    assert "прерван перезапуском" in db.item(item_id)["note"]


def test_panel_shows_conversation(db):
    from fastapi.testclient import TestClient
    from jellyprep import main
    item_id = db.add_item(source="folder", name="x", category="tv", content_path="/x", state="review",
                          ident={"kind": "tv", "confidence": 0.5, "problems": [], "episodes": [], "unmatched": [],
                                 "show": {"candidates": [], "chosen": None, "queries": []}},
                          plan={"actions": [], "problems": [], "target": None})
    conv = llm.Conversation(model="google/gemma-4-12b-qat", prompt="Video files:\n- a.avi", reasoning="Это сериал.",
                            content='{"kind": "tv", "comment": "ок"}', status="ok", finished=None,
                            usage={"completion_tokens": 4457, "completion_tokens_details": {"reasoning_tokens": 2753}})
    conv.finished = conv.started + 113
    db.trace(item_id, llm_log=[conv.as_dict()])
    with TestClient(main.create_app(db, start_worker=False), base_url="http://jp.test") as c:
        html = c.get(f"/items/{item_id}/panel").text
    assert "Разговор с моделью" in html and "gemma-4-12b-qat" in html and "1 мин 53 с" in html
    assert "Это сериал." in html and "&#34;comment&#34;: &#34;ок&#34;" in html and "из них на рассуждения 2753" in html


def _title_matched_item(db, tmp_path, monkeypatch, tmdb, approval="auto_if_confident"):
    from jellyprep import config, worker as worker_mod
    from conftest import make_file
    root = tmp_path / "media" / "Тайны следствия"
    for name in ("1. Мягкая лапа смерти (1 сер.).avi", "1. Мягкая лапа смерти (2 сер.).avi",
                 "2. Гроб на две персоны (1 сер.).avi", "2. Гроб на две персоны (2 сер.).avi"):
        make_file(root / name)
    monkeypatch.setattr(worker_mod, "tmdb_client", lambda s: tmdb)
    monkeypatch.setattr(worker_mod, "safe_probe", lambda p: {"video": None})
    config.save(db, {"llm_kind": "openai", "llm_url": "http://lm/v1", "llm_model": "m", "llm_auto": False,
                     "library_root": str(tmp_path / "lib")})
    db.upsert_category("tv", enabled=1, kind="tv", approval=approval)
    item_id = db.add_item(source="folder", name="Тайны следствия", category="tv", content_path=str(root))
    w = worker_mod.Worker(db)
    w.analyze(item_id)
    assert db.item(item_id)["state"] == "review"
    return w, item_id


def _model_says(monkeypatch, mapping, tmdb_id=12345):
    def fake_ask(s, prompt, conv=None):
        conv.status = "ok"
        return {"kind": "tv", "tmdb_id": tmdb_id, "title": "Тайны следствия", "year": 2000, "comment": "по названиям",
                "episodes": [{"file": f, "season": se, "episode": ep} for f, (se, ep) in mapping.items()]}
    monkeypatch.setattr(llm, "ask", fake_ask)


RIGHT = {"1. Мягкая лапа смерти (1 сер.).avi": (1, 1), "1. Мягкая лапа смерти (2 сер.).avi": (1, 2),
         "2. Гроб на две персоны (1 сер.).avi": (1, 3), "2. Гроб на две персоны (2 сер.).avi": (1, 4)}


def test_model_agreeing_with_tmdb_continues_without_asking(db, tmp_path, monkeypatch, tmdb):
    w, item_id = _title_matched_item(db, tmp_path, monkeypatch, tmdb)
    _model_says(monkeypatch, RIGHT)
    w.ask_llm(item_id, wait=True)
    item = db.item(item_id)
    assert item["state"] == "approved" and item["trace"]["approved_by"] == "model"
    assert "продолжено без подтверждения" in item["note"]
    assert {e["how"][:9] for e in item["ident"]["episodes"]} == {"по назван"}     # the rules' result is kept
    from jellyprep import pipeline
    p = pipeline.build(item, {"dry_run": True}, None, [])
    assert "подтверждено моделью" in p["stages"][2]["lines"][0]


def test_model_disagreeing_waits_and_shows_difference(db, tmp_path, monkeypatch, tmdb):
    w, item_id = _title_matched_item(db, tmp_path, monkeypatch, tmdb)
    wrong = {**RIGHT, "2. Гроб на две персоны (2 сер.).avi": (2, 1)}
    _model_says(monkeypatch, wrong)
    w.ask_llm(item_id, wait=True)
    item = db.item(item_id)
    assert item["state"] == "review"
    assert "расходится с TMDB" in item["note"] and "TMDB S01E04, модель S02E01" in item["note"]


def test_always_confirm_category_still_waits(db, tmp_path, monkeypatch, tmdb):
    w, item_id = _title_matched_item(db, tmp_path, monkeypatch, tmdb, approval="always_confirm")
    _model_says(monkeypatch, RIGHT)
    w.ask_llm(item_id, wait=True)
    item = db.item(item_id)
    assert item["state"] == "review" and "модель согласна с TMDB — подтвердите" in item["note"]


def test_looping_model_is_stopped(monkeypatch):
    conv = llm.Conversation(model="m", prompt="p")

    def lines():
        while True:
            yield sse(reasoning_content="Wait, I'll check the file names again. " * 50)

    monkeypatch.setattr(llm.httpx, "stream", lambda *a, **kw: FakeStream(lines(), {}))
    with pytest.raises(llm.LLMError, match="зациклилась"):
        llm.ask({"llm_kind": "openai", "llm_url": "http://lm", "llm_model": "m"}, "p", conv)
    assert conv.status == "error" and len(conv.reasoning) > llm.MAX_REASONING_CHARS


def test_model_may_leave_a_file_unmatched():
    ans = {"kind": "tv", "tmdb_id": 4556, "title": "Клиника", "year": 2001, "comment": "финал в TMDB одной серией",
           "episodes": [{"file": FILES[0], "season": 8, "episode": 18}, {"file": FILES[1], "season": None, "episode": None}]}
    ov, comment = llm.to_overrides(ans, FILES, {4556})
    assert ov["episodes"] == {FILES[0]: [8, 18, None]} and "одной серией" in comment


def test_model_requests_run_one_at_a_time(db, tmp_path, monkeypatch):
    import threading
    import time as _time
    from jellyprep import config, worker as worker_mod
    from conftest import make_file
    config.save(db, {"llm_kind": "openai", "llm_url": "http://lm/v1", "llm_model": "m", "llm_auto": False})
    db.upsert_category("tv", enabled=1, kind="tv")
    ids = []
    for n in range(3):
        root = tmp_path / f"show{n}"
        make_file(root / "a.avi")
        ids.append(db.add_item(source="folder", name=f"show{n}", category="tv", content_path=str(root), state="review",
                               ident={"kind": "tv", "show": {"candidates": []}, "episodes": []}))
    running, peak, gate = [0], [0], threading.Event()

    def slow_ask(s, prompt, conv=None):
        running[0] += 1
        peak[0] = max(peak[0], running[0])
        gate.wait(2)
        _time.sleep(0.05)
        running[0] -= 1
        raise llm.LLMError("тест")

    monkeypatch.setattr(llm, "ask", slow_ask)
    w = worker_mod.Worker(db)
    for i in ids:
        w.ask_llm(i)
    _time.sleep(0.2)
    assert db.item(ids[0])["note"] == worker_mod.LLM_THINKING
    assert db.item(ids[2])["note"] == worker_mod.LLM_QUEUED
    assert w.llm_queue_position(ids[1]) == 1 and w.llm_queue_position(ids[2]) == 2
    w.cancel_llm(ids[2])                                    # take the last one out of the queue
    assert ids[2] not in w.llm_jobs and db.item(ids[2])["note"] is None
    gate.set()
    w.llm_jobs.get(ids[1]) and w.llm_jobs[ids[1]].done.wait(5)
    assert peak[0] == 1                                     # never two requests at once
    assert not w.llm_jobs
