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


def test_ask_openai_compatible(monkeypatch):
    sent = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({"kind": "movie", "tmdb_id": None, "title": "X",
                                                                     "episodes": [], "comment": ""})}}]}

    def post(url, **kw):
        sent.update(url=url, **kw)
        return Resp()

    monkeypatch.setattr(llm.httpx, "post", post)
    s = {"llm_kind": "openai", "llm_url": "http://lm:1234", "llm_model": "qwen/qwen3-14b", "llm_key": ""}
    assert llm.ask(s, "prompt")["title"] == "X"
    assert sent["url"] == "http://lm:1234/v1/chat/completions"
    assert sent["json"]["response_format"]["type"] == "json_schema" and sent["json"]["model"] == "qwen/qwen3-14b"


def test_worker_applies_model_answer_for_review(tmp_path, db, tmdb, monkeypatch):
    from jellyprep import config, worker as worker_mod
    from conftest import make_file
    root = tmp_path / "media" / "Непонятное"
    a = make_file(root / "a.avi")
    b = make_file(root / "b.avi")
    monkeypatch.setattr(worker_mod, "tmdb_client", lambda s: tmdb)
    monkeypatch.setattr(worker_mod, "safe_probe", lambda p: None)
    monkeypatch.setattr(llm, "ask", lambda s, prompt: {
        "kind": "tv", "tmdb_id": None, "title": "Разделение", "year": 2022, "comment": "угадала",
        "episodes": [{"file": "a.avi", "season": 1, "episode": 1}, {"file": "b.avi", "season": 1, "episode": 2}]})
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
