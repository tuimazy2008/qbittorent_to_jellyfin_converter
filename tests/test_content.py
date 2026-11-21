import json
import shutil

import pytest

from jellyprep import config, identify, llm, media, parse, worker as worker_mod
from jellyprep.clients import Candidate
from conftest import fake_info, make_file


def test_hints_from_tags():
    info = {"title": "-=HANSMER=- (HQ-Films.ws)",
            "video": {"title": "The.Time.Machine.2002.1080p.DTheater.DTS.x264-CtrlHD"}}
    assert parse.hints_from_info(info) == ["The.Time.Machine.2002.1080p.DTheater.DTS.x264-CtrlHD"]
    assert parse.hints_from_info({"title": "", "video": {"title": ""}}) == []


def test_identify_movie_by_tag_hint(tmp_path, tmdb):
    tmdb.movies.append(Candidate(2135, "Машина времени", "The Time Machine", 2002, 20))
    f = make_file(tmp_path / "hns-tma.mkv")
    ident = identify.identify(media.scan(f), "movie", tmdb, hints=["The.Time.Machine.2002.1080p.DTheater.DTS.x264-CtrlHD"])
    assert ident["movies"][0]["chosen"]["id"] == 2135 and ident["confidence"] >= 0.9


def test_frame_times_cover_opening_middle_credits():
    t = media.frame_times(7946)
    assert t[0] == 20 and 3970 <= t[len(t) // 2] <= 3980 or 3973 in t
    assert max(t) == 7946 - 40 and len(t) == 13
    assert media.frame_times(30) == [15.0]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_extract_frames_real(tmp_path):
    import subprocess
    src = tmp_path / "clip.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=duration=200:size=320x240:rate=5",
                    "-c:v", "libx264", "-preset", "ultrafast", str(src)], check=True)
    frames = media.extract_frames(src, 200, tmp_path / "frames", width=160)
    assert len(frames) >= 8 and all(f.stat().st_size > 0 for f in frames)


def test_vision_setting_and_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_detect_vision", lambda k, u, m: calls.append(m) or True)
    llm._VISION.clear()
    s = {"llm_kind": "openai", "llm_url": "http://lm:1234/v1", "llm_model": "google/gemma-4-12b-qat"}
    assert llm.supports_vision(s) is True and llm.supports_vision(s) is True and len(calls) == 1
    assert llm.supports_vision({**s, "llm_vision": "off"}) is False


def test_detect_vision_lm_studio_and_ollama(monkeypatch):
    from conftest import REAL_DETECT_VISION as real

    class R:
        def __init__(self, data):
            self.data = data

        def json(self):
            return self.data

    monkeypatch.setattr(llm.httpx, "get", lambda url, timeout: R({"type": "vlm"} if "gemma" in url else {"type": "llm"}))
    monkeypatch.setattr(llm.httpx, "post", lambda url, json, timeout: R({"capabilities": ["completion", "vision"]}))
    assert real("openai", "http://lm:1234/v1", "google/gemma-4-12b-qat") is True
    assert real("openai", "http://lm:1234/v1", "qwen/qwen3-14b") is False
    assert real("ollama", "http://o:11434", "llava") is True


def test_ask_sends_images(monkeypatch):
    from test_llm import FakeStream, sse
    sent = {}
    answer = json.dumps({"kind": "movie", "tmdb_id": None, "title": "Дом", "episodes": [], "comment": "ДОМ на экране"})

    def stream(method, url, **kw):
        sent.update(kw)
        return FakeStream([sse(content=answer)], sent)

    monkeypatch.setattr(llm.httpx, "stream", stream)
    llm.ask({"llm_kind": "openai", "llm_url": "http://lm", "llm_model": "m"}, "p", llm.Conversation(model="m", prompt="p"),
            images=[b"\xff\xd8jpeg"])
    content = sent["json"]["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "p"} and content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    monkeypatch.setattr(llm.httpx, "stream", lambda method, url, **kw: sent.update(kw) or FakeStream(
        [json.dumps({"message": {"content": answer}, "done": True})], sent))
    llm.ask({"llm_kind": "ollama", "llm_url": "http://o", "llm_model": "m"}, "p", llm.Conversation(model="m", prompt="p"),
            images=[b"x"])
    assert sent["json"]["messages"][1]["images"] == ["eA=="]


def _movie_item(db, tmp_path, monkeypatch, tmdb, seen, pick=None):
    f = make_file(tmp_path / "media" / "dom.mkv")
    monkeypatch.setattr(worker_mod, "tmdb_client", lambda s: tmdb)
    monkeypatch.setattr(worker_mod, "safe_probe", lambda p: fake_info(duration=7946))
    monkeypatch.setattr(llm, "supports_vision", lambda s: True)

    def frames(path, duration, out_dir, width=768):
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "f_007906.jpg"
        p.write_bytes(b"jpeg")
        return [p]

    monkeypatch.setattr(media, "extract_frames", frames)
    monkeypatch.setattr(worker_mod, "FRAMES_DIR", tmp_path / "frames")
    got = {"calls": []}

    def fake_ask(s, prompt, conv=None, images=None, schema=None, system=None):
        conv.status, conv.system = "ok", system or llm.SYSTEM
        got["calls"].append({"schema": schema, "images": images, "prompt": prompt})
        return seen if schema is llm.VISION_SCHEMA else pick

    monkeypatch.setattr(llm, "ask", fake_ask)
    config.save(db, {"llm_kind": "openai", "llm_url": "http://lm/v1", "llm_model": "m", "llm_auto": False,
                     "library_root": str(tmp_path / "lib")})
    db.upsert_category("movies", enabled=1, kind="movies", approval="auto_if_confident")
    item_id = db.add_item(source="folder", name="dom.mkv", category="movies", content_path=str(f))
    w = worker_mod.Worker(db)
    w.analyze(item_id)
    w.ask_llm(item_id, wait=True)
    return db.item(item_id), got


SEEN_DOM = {"title": "Дом", "original_title": "", "year": None, "credits": ["Фонд кино", "Мосфильм", "Сергей Сельянов"],
            "comment": "На экране «ДОМ», Фонд кино, Мосфильм"}


def test_frames_read_first_then_model_picks_by_cards(db, tmp_path, monkeypatch, tmdb):
    tmdb.movies += [Candidate(79580, "Дом", "Дом", 2011, 3), Candidate(228161, "Дом", "Home", 2015, 9)]
    item, got = _movie_item(db, tmp_path, monkeypatch, tmdb, SEEN_DOM, {"tmdb_id": 79580, "comment": "Мосфильм в титрах"})
    step_a, step_b = got["calls"]
    assert step_a["schema"] is llm.VISION_SCHEMA and step_a["images"] == [b"jpeg"] and "TMDB" not in step_a["prompt"]
    assert step_b["schema"] is llm.CHOOSE_SCHEMA and "id 79580" in step_b["prompt"] and "Мосфильм" in step_b["prompt"]
    assert item["ident"]["movies"][0]["chosen"]["id"] == 79580
    log = item["trace"]["llm_log"]
    assert len(log) == 2 and log[1]["images"] == ["f_007906.jpg"] and log[0]["system"].startswith("You pick")
    assert "на экране" in item["note"] and item["state"] == "review"


def test_clear_title_match_needs_no_second_step(db, tmp_path, monkeypatch, tmdb):
    tmdb.movies += [Candidate(2135, "Машина времени", "The Time Machine", 2002, 20)]
    item, got = _movie_item(db, tmp_path, monkeypatch, tmdb,
                            {"title": "Машина времени", "original_title": "The Time Machine", "year": 2002,
                             "credits": [], "comment": "титры"})
    assert len(got["calls"]) == 1 and item["ident"]["movies"][0]["chosen"]["id"] == 2135


def test_no_card_fits_offers_candidates(db, tmp_path, monkeypatch, tmdb):
    tmdb.movies += [Candidate(79580, "Дом", "Дом", 2011, 3), Candidate(228161, "Дом", "Home", 2015, 9)]
    item, _ = _movie_item(db, tmp_path, monkeypatch, tmdb, SEEN_DOM, {"tmdb_id": None, "comment": "ни одна не подходит"})
    m = item["ident"]["movies"][0]
    assert m["chosen"]["title"] == "Дом" and not m["chosen"].get("id")
    assert {79580, 228161} <= {c["id"] for c in m["candidates"]}


def test_frame_route(db, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from jellyprep import main
    monkeypatch.setattr(worker_mod, "FRAMES_DIR", tmp_path)
    (tmp_path / "7").mkdir()
    (tmp_path / "7" / "f_000020.jpg").write_bytes(b"\xff\xd8")
    with TestClient(main.create_app(db, start_worker=False), base_url="http://jp.test") as c:
        assert c.get("/items/7/frames/f_000020.jpg").content == b"\xff\xd8"
        assert c.get("/items/7/frames/..%2F..%2Fsecret").status_code in (400, 404)


def test_generic_folder_name_is_not_a_title(tmp_path, tmdb):
    from jellyprep import parse
    assert parse.is_generic_name("Season 01") and parse.is_generic_name("Сезон 2") and parse.is_generic_name("S03")
    assert not parse.is_generic_name("Тайны следствия") and not parse.is_generic_name("Severance (Season 1) BDRip")
    root = tmp_path / "series" / "Season 01"
    for name in ("S01E01 - The Pants Tent.mkv", "S01E02 - Ted And Mary.mkv"):
        make_file(root / name)
    ident = identify.identify(media.scan(root), "tv", tmdb)
    assert any(p.startswith(identify.SHOW_UNKNOWN_PROBLEM) for p in ident["problems"])
    assert worker_mod._show_is_weak(ident)


def test_series_by_frames(db, tmp_path, monkeypatch, tmdb):
    tmdb.tv += [Candidate(4546, "Умерь свой энтузиазм", "Curb Your Enthusiasm", 2000, 30)]
    tmdb.seasons[4546] = {1: 10}
    root = tmp_path / "media" / "Season 01"
    for name in ("S01E01 - The Pants Tent.mkv", "S01E02 - Ted And Mary.mkv"):
        make_file(root / name)
    monkeypatch.setattr(worker_mod, "tmdb_client", lambda s: tmdb)
    monkeypatch.setattr(worker_mod, "safe_probe", lambda p: fake_info(duration=1700, audio=(("eng", 2),)))
    monkeypatch.setattr(llm, "supports_vision", lambda s: True)
    monkeypatch.setattr(worker_mod, "FRAMES_DIR", tmp_path / "frames")

    def frames(path, duration, out_dir, width=768):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "f_000045.jpg").write_bytes(b"j")
        return [out_dir / "f_000045.jpg"]

    monkeypatch.setattr(media, "extract_frames", frames)
    prompts = []

    def fake_ask(s, prompt, conv=None, images=None, schema=None, system=None):
        conv.status, conv.system = "ok", system or llm.SYSTEM
        prompts.append(prompt)
        return {"title": "Curb Your Enthusiasm", "original_title": "", "title_from_name": "", "year": None,
                "credits": ["HBO"], "comment": "заставка «Curb Your Enthusiasm»"}

    monkeypatch.setattr(llm, "ask", fake_ask)
    config.save(db, {"llm_kind": "openai", "llm_url": "http://lm/v1", "llm_model": "m", "llm_auto": False,
                     "library_root": str(tmp_path / "lib")})
    db.upsert_category("adult_series", enabled=1, kind="tv", approval="auto_if_confident")
    item_id = db.add_item(source="folder", name="Season 01", category="adult_series", content_path=str(root))
    w = worker_mod.Worker(db)
    w.analyze(item_id)
    w.ask_llm(item_id, wait=True)
    item = db.item(item_id)
    assert "The Pants Tent" in prompts[0]
    assert item["ident"]["show"]["chosen"]["id"] == 4546
    assert sorted((e["season"], e["episode"]) for e in item["ident"]["episodes"]) == [(1, 1), (1, 2)]


def test_studios_and_year_when_no_title_is_readable(monkeypatch):
    class T:
        def companies(self, name):
            return {"20TH CENTURY FOX": [25], "DAVIS ENTERTAINMENT": [1302], "NEW REGENCY": [508]}.get(name.upper(), [])

        def discover_movies(self, ids, year):
            assert set(ids) >= {25, 1302} and year == 2002
            return [Candidate(15357, "Жизнь или что-то вроде того", "Life or Something Like It", 2002, 5)]

    got = worker_mod._by_studios(T(), ["20th CENTURY FOX", "A DAVIS ENTERTAINMENT/NEW REGENCY PRODUCTION", "Carpenters"], 2002)
    assert [c.id for c in got] == [15357]


def test_skeleton_and_disemvoweled_names():
    from jellyprep import parse
    assert parse.disemvoweled("LF_R_SMTHNG_LK_T_2002_720p_WEB-DL -BLUEBIRD.mkv") == "LF_R_SMTHNG_LK_T"
    assert parse.disemvoweled("Love.Actually.2003.mkv") is None
    assert parse.skeleton("LF_R_SMTHNG_LK_T") == parse.skeleton("Life or Something Like It")


def test_disemvoweled_film_found_via_studios_without_trusting_the_model(db, tmp_path, monkeypatch, tmdb):
    tmdb.movies += [Candidate(87827, "Жизнь Пи", "Life of Pi", 2012, 30),
                    Candidate(15357, "Жизнь или что-то вроде того", "Life or Something Like It", 2002, 5)]
    monkeypatch.setattr(worker_mod, "_by_studios", lambda t, credits, year: [
        Candidate(15357, "Жизнь или что-то вроде того", "Life or Something Like It", 2002, 5),
        Candidate(1000, "Другой фильм Fox", "Another Fox Film", 2002, 9)])
    f = make_file(tmp_path / "media" / "LF_R_SMTHNG_LK_T_2002_720p_WEB-DL -BLUEBIRD.mkv")
    monkeypatch.setattr(worker_mod, "tmdb_client", lambda s: tmdb)
    monkeypatch.setattr(worker_mod, "safe_probe", lambda p: fake_info(duration=6200))
    monkeypatch.setattr(llm, "supports_vision", lambda s: True)
    monkeypatch.setattr(worker_mod, "FRAMES_DIR", tmp_path / "frames")
    monkeypatch.setattr(media, "extract_frames", lambda p, d, o, width=768: (o.mkdir(parents=True, exist_ok=True),
                        (o / "f_000020.jpg").write_bytes(b"j"), [o / "f_000020.jpg"])[2])

    def fake_ask(s, prompt, conv=None, images=None, schema=None, system=None):
        conv.status, conv.system = "ok", system or llm.SYSTEM
        return {"title": "", "original_title": "", "title_from_name": "Life of Pi", "year": 2012,   # bent to its guess
                "credits": ["20th CENTURY FOX", "A DAVIS ENTERTAINMENT/NEW REGENCY PRODUCTION"], "comment": "титры Fox"}

    monkeypatch.setattr(llm, "ask", fake_ask)
    config.save(db, {"llm_kind": "openai", "llm_url": "http://lm/v1", "llm_model": "m", "llm_auto": False,
                     "library_root": str(tmp_path / "lib")})
    db.upsert_category("movies", enabled=1, kind="movies")
    item_id = db.add_item(source="folder", name=f.name, category="movies", content_path=str(f))
    w = worker_mod.Worker(db)
    w.analyze(item_id)
    w.ask_llm(item_id, wait=True)
    item = db.item(item_id)
    assert item["ident"]["movies"][0]["chosen"]["id"] == 15357            # not «Жизнь Пи» (2012)
    assert "без гласных" in item["note"]


def test_episode_title_is_not_taken_for_the_show(db, tmp_path, monkeypatch, tmdb):
    root = tmp_path / "media" / "Season 01"
    for name in ("S01E01 - The Pants Tent.mkv", "S01E02 - Ted And Mary.mkv"):
        make_file(root / name)
    monkeypatch.setattr(worker_mod, "tmdb_client", lambda s: tmdb)
    monkeypatch.setattr(worker_mod, "safe_probe", lambda p: fake_info(duration=1700, audio=(("eng", 2),)))
    monkeypatch.setattr(llm, "supports_vision", lambda s: True)
    monkeypatch.setattr(worker_mod, "FRAMES_DIR", tmp_path / "frames")
    monkeypatch.setattr(media, "extract_frames", lambda p, d, o, width=768: (o.mkdir(parents=True, exist_ok=True),
                        (o / "f_000020.jpg").write_bytes(b"j"), [o / "f_000020.jpg"])[2])

    def fake_ask(s, prompt, conv=None, images=None, schema=None, system=None):
        conv.status, conv.system = "ok", system or llm.SYSTEM
        return {"title": "The Pants Tent", "original_title": "", "title_from_name": "", "year": None,
                "credits": [], "comment": "сериал «The Pants Tent»"}

    monkeypatch.setattr(llm, "ask", fake_ask)
    config.save(db, {"llm_kind": "openai", "llm_url": "http://lm/v1", "llm_model": "m", "llm_auto": False,
                     "library_root": str(tmp_path / "lib")})
    db.upsert_category("adult_series", enabled=1, kind="tv")
    item_id = db.add_item(source="folder", name="Season 01", category="adult_series", content_path=str(root))
    w = worker_mod.Worker(db)
    w.analyze(item_id)
    w.ask_llm(item_id, wait=True)
    item = db.item(item_id)
    assert "не узнала сериал" in item["note"] and not (item["ident"]["show"].get("chosen") or {}).get("title") == "The Pants Tent"
