import os

import pytest

from jellyprep import config, worker as worker_mod
from jellyprep.db import APPROVED, DONE, REMOVED, REVIEW
from conftest import FakeQBit, fake_info, make_file


@pytest.fixture
def env(tmp_path, db, tmdb, monkeypatch):
    qb = FakeQBit()
    monkeypatch.setattr(worker_mod, "qbit_client", lambda s: qb)
    monkeypatch.setattr(worker_mod, "tmdb_client", lambda s: tmdb)
    monkeypatch.setattr(worker_mod, "safe_probe", lambda p: fake_info(audio=(("rus", 2),)))
    config.save(db, {"setup_done": True, "library_root": str(tmp_path / "library"), "dry_run": True,
                     "min_free_gb": 0, "path_map": [{"from": "/media", "to": str(tmp_path / "media")}],
                     "voice_track": False})  # these tests are about hardlinks; media jobs are tested separately
    db.upsert_category("movies", enabled=1, kind="movies", library_dir="movies",
                       approval="auto_if_confident", delete_enabled=1, seed_ratio=2.0, seed_days=30.0)
    db.upsert_category("tv-sonarr", enabled=0, kind="tv", library_dir="tv")
    w = worker_mod.Worker(db)
    return w, qb, tmp_path


def torrent(h, name, cat, ratio=0.5, days=1, progress=1.0, state="stalledUP"):
    return {"hash": h, "name": name, "category": cat, "content_path": f"/media/movies/{name}",
            "ratio": ratio, "seeding_time": days * 86400, "progress": progress, "state": state}


def test_full_cycle_dry_run_then_real(env, db):
    w, qb, tmp = env
    src = make_file(tmp / "media" / "movies" / "Love.Actually.2003.1080p.mkv")
    make_file(tmp / "media" / "movies" / "dom.mkv")
    qb.torrents_list = [
        torrent("a" * 40, "Love.Actually.2003.1080p.mkv", "movies"),
        torrent("b" * 40, "dom.mkv", "movies"),
        torrent("c" * 40, "Severance S01", "tv-sonarr"),                       # category disabled
        torrent("d" * 40, "Hope Gap.mkv", "movies", progress=0.5, state="downloading"),  # not finished
    ]
    w.run_cycle()
    items = {i["name"]: i for i in db.items()}
    assert set(items) == {"Love.Actually.2003.1080p.mkv", "dom.mkv"}
    love, dom = items["Love.Actually.2003.1080p.mkv"], items["dom.mkv"]
    assert love["state"] == APPROVED
    assert dom["state"] == REVIEW
    lib_file = tmp / "library" / "movies" / "Реальная любовь (2003) [tmdbid-508]" / "Реальная любовь (2003).mkv"
    assert not lib_file.exists(), "dry-run must not touch the disk"

    config.save(db, {"dry_run": False})
    w.run_cycle()
    assert db.item(love["id"])["state"] == DONE
    assert os.path.samefile(src, lib_file)
    assert qb.deleted == []  # ratio 0.5 / 1 day: not yet

    qb.torrents_list[0]["ratio"] = 2.1
    w.run_cycle()
    assert qb.deleted == [("a" * 40, True)]
    assert db.item(love["id"])["state"] == REMOVED


def test_cleanup_waits_when_library_file_missing(env, db):
    w, qb, tmp = env
    make_file(tmp / "media" / "movies" / "Death Becomes Her.mkv")
    qb.torrents_list = [torrent("e" * 40, "Death Becomes Her.mkv", "movies", ratio=0.1)]
    config.save(db, {"dry_run": False})
    w.run_cycle()
    item = db.items()[0]
    assert item["state"] == DONE
    os.unlink(item["result"]["files"][0]["dst"])
    qb.torrents_list[0]["ratio"] = 5
    w.run_cycle()
    assert qb.deleted == []
    assert any("библиотека не в порядке" in l["message"] for l in db.logs(item["id"]))


def test_dry_run_never_deletes(env, db):
    w, qb, tmp = env
    make_file(tmp / "media" / "movies" / "Death Becomes Her.mkv")
    qb.torrents_list = [torrent("f" * 40, "Death Becomes Her.mkv", "movies", ratio=0.1)]
    config.save(db, {"dry_run": False})
    w.run_cycle()
    assert db.items()[0]["state"] == DONE
    config.save(db, {"dry_run": True})
    qb.torrents_list[0]["ratio"] = 10
    w.run_cycle()
    assert qb.deleted == []


def test_already_seeded_is_removed_right_after_processing(env, db):
    w, qb, tmp = env
    make_file(tmp / "media" / "movies" / "Death Becomes Her.mkv")
    qb.torrents_list = [torrent("h" * 40, "Death Becomes Her.mkv", "movies", ratio=10)]
    config.save(db, {"dry_run": False})
    w.run_cycle()
    assert qb.deleted == [("h" * 40, True)]
    assert db.items()[0]["state"] == REMOVED


def test_review_then_manual_approve(env, db):
    w, qb, tmp = env
    make_file(tmp / "media" / "movies" / "dom.mkv")
    qb.torrents_list = [torrent("g" * 40, "dom.mkv", "movies")]
    config.save(db, {"dry_run": False})
    w.run_cycle()
    item = db.items()[0]
    assert item["state"] == REVIEW
    w.analyze(item["id"], {"kind": "movie", "movies": {"0": {"title": "Дом", "year": 2011}}})
    assert db.item(item["id"])["state"] == REVIEW  # manual edits always need a confirmation
    w.approve(item["id"])
    w.run_cycle()
    assert db.item(item["id"])["state"] == DONE
    assert (tmp / "library" / "movies" / "Дом (2011)" / "Дом (2011).mkv").exists()


@pytest.mark.parametrize("cat, ratio, days, due", [
    ({"delete_enabled": 1, "seed_ratio": 2.0, "seed_days": 30.0}, 2.0, 1, True),
    ({"delete_enabled": 1, "seed_ratio": 2.0, "seed_days": 30.0}, 0.1, 31, True),
    ({"delete_enabled": 1, "seed_ratio": 2.0, "seed_days": 30.0}, 1.9, 29, False),
    ({"delete_enabled": 1, "seed_ratio": None, "seed_days": None}, 9.0, 99, False),
    ({"delete_enabled": 0, "seed_ratio": 1.0, "seed_days": 1.0}, 9.0, 99, False),
    (None, 9.0, 99, False),
])
def test_deletion_due(cat, ratio, days, due):
    assert worker_mod.deletion_due(cat, ratio, days * 86400)[0] is due


def test_map_path():
    m = [{"from": "/downloads", "to": "/media/downloads"}, {"from": "/downloads/tv", "to": "/tv"}]
    assert worker_mod.map_path("/downloads/x.mkv", m) == "/media/downloads/x.mkv"
    assert worker_mod.map_path("/downloads/tv/x.mkv", m) == "/tv/x.mkv"
    assert worker_mod.map_path("/downloadsX/x.mkv", m) == "/downloadsX/x.mkv"


# --- stage 2: heavy jobs, night window, import ---------------------------------------
import datetime as dt

from jellyprep import transcode
from jellyprep.db import RUNNING


@pytest.fixture
def heavy_env(env, db, monkeypatch):
    w, qb, tmp = env
    calls = []

    def fake_process(src, dst, info, s, *, transcode, progress=None):
        calls.append((src, dst, transcode))
        if progress:
            progress(0.5, "2.0x")
        with open(dst + ".tmp", "wb") as f:
            f.write(b"x" * 10)
        os.replace(dst + ".tmp", dst)
        return {"hw": "test", "seconds": 1, "size": 10}

    monkeypatch.setattr(transcode, "process", fake_process)
    config.save(db, {"dry_run": False, "voice_track": True, "night_only": True,
                     "night_start": "01:00", "night_end": "08:00"})
    return w, qb, tmp, calls


def test_transcode_waits_for_night_unless_run_now(heavy_env, db, monkeypatch):
    w, qb, tmp, calls = heavy_env
    monkeypatch.setattr(worker_mod, "safe_probe", lambda p: fake_info(3840, 2160))
    make_file(tmp / "media" / "movies" / "Death Becomes Her.mkv")
    qb.torrents_list = [torrent("n" * 40, "Death Becomes Her.mkv", "movies")]
    monkeypatch.setattr(worker_mod, "in_window", lambda s, now=None: False)
    w.run_cycle()
    item = db.items()[0]
    assert item["state"] == APPROVED and item["plan"]["actions"][0]["action"] == "transcode"
    assert w.run_next_heavy() is False and calls == []          # daytime: waits
    w.approve(item["id"], run_now=True)
    assert w.run_next_heavy() is True
    assert calls and calls[0][2] is True
    assert db.item(item["id"])["state"] == DONE


def test_audio_processing_runs_any_time(heavy_env, db, monkeypatch):
    w, qb, tmp, calls = heavy_env
    make_file(tmp / "media" / "movies" / "Hope Gap.mkv")
    qb.torrents_list = [torrent("p" * 40, "Hope Gap.mkv", "movies")]
    monkeypatch.setattr(worker_mod, "in_window", lambda s, now=None: False)
    w.run_cycle()
    item = db.items()[0]
    assert item["plan"]["actions"][0]["action"] == "process"
    assert item["state"] == APPROVED                               # heavy: not done by the main cycle
    assert w.run_next_heavy() is True and db.item(item["id"])["state"] == DONE


def test_dry_run_runs_no_heavy_jobs(heavy_env, db):
    w, qb, tmp, calls = heavy_env
    make_file(tmp / "media" / "movies" / "Hope Gap.mkv")
    qb.torrents_list = [torrent("q" * 40, "Hope Gap.mkv", "movies")]
    config.save(db, {"dry_run": True})
    w.run_cycle()
    assert w.run_next_heavy() is False and calls == []


def test_recover_requeues_running_items(heavy_env, db):
    w, qb, tmp, calls = heavy_env
    item_id = db.add_item(source="folder", name="x", category="movies", content_path="/nope", state=RUNNING,
                          plan={"actions": []})
    w.recover()
    assert db.item(item_id)["state"] == APPROVED


@pytest.mark.parametrize("now, expect", [("02:00", True), ("12:00", False), ("23:30", True), ("07:59", True)])
def test_night_window_wraps_midnight(now, expect):
    s = {"night_only": True, "night_start": "23:00", "night_end": "08:00"}
    h, m = map(int, now.split(":"))
    assert worker_mod.in_window(s, dt.datetime(2026, 1, 1, h, m)) is expect


def test_import_folder_moves_files_not_in_torrents(heavy_env, db, monkeypatch):
    w, qb, tmp, calls = heavy_env
    config.save(db, {"voice_track": False})
    old = tmp / "media" / "movies" / "adult_movies"
    lonely = make_file(old / "Death Becomes Her.mkv")
    seeding = make_file(old / "Hope Gap.mkv")
    make_file(old / "Show" / "info.txt", 0)                     # no video → ignored
    qb.torrents_list = [{**torrent("s" * 40, "Hope Gap.mkv", "movies"),
                         "content_path": "/media/movies/adult_movies/Hope Gap.mkv"}]
    res = w.import_folder(str(old), "movies", delete_source=True)
    assert res["added"] == ["Death Becomes Her.mkv"] and res["torrent"] == ["Hope Gap.mkv"]
    assert res["empty"] == ["Show"]
    assert w.import_folder(str(old), "movies")["known"] == ["Death Becomes Her.mkv"]  # no duplicates

    qb.torrents_list = []      # the cycle must not pick up the seeding torrent here
    w.run_cycle()
    item = next(i for i in db.items() if i["source"] == "folder")
    assert item["state"] == REMOVED                              # moved: library copy made, original removed
    assert not lonely.exists() and seeding.exists()
    lib = tmp / "library" / "movies" / "Смерть ей к лицу (1992) [tmdbid-598]" / "Смерть ей к лицу (1992).mkv"
    assert lib.exists()


def test_import_keeps_source_when_asked(heavy_env, db):
    w, qb, tmp, calls = heavy_env
    config.save(db, {"voice_track": False})
    old = tmp / "media" / "old"
    src = make_file(old / "Death Becomes Her.mkv")
    w.import_folder(str(old), "movies", delete_source=False)
    w.run_cycle()
    assert src.exists() and db.items()[0]["state"] == DONE


def test_reprocess_library_in_place(heavy_env, db):
    w, qb, tmp, calls = heavy_env
    lib_movies = tmp / "library" / "movies"
    f = make_file(lib_movies / "Смерть ей к лицу (1992) [tmdbid-598]" / "Смерть ей к лицу (1992).mkv")
    res = w.import_folder(str(lib_movies), "movies", delete_source=True)
    assert len(res["added"]) == 1
    w.run_cycle()
    item = db.items()[0]
    a = item["plan"]["actions"][0]
    assert a["action"] == "process" and a.get("inplace")       # voice track added to the same file
    assert item["ident"]["movies"][0]["chosen"]["id"] == 598   # TMDB id taken from the folder name
    assert w.run_next_heavy() is True
    assert calls[0][0] == calls[0][1] == str(f)
    assert f.exists() and db.item(item["id"])["state"] == DONE


def test_import_skips_hardlinks_of_seeding_torrents(heavy_env, db):
    w, qb, tmp, calls = heavy_env
    seeding = make_file(tmp / "media" / "downloads" / "Severance S01" / "Severance.S01E01.mkv")
    linked = tmp / "media" / "tv" / "Severance" / "Season 1" / "Severance.S01E01.mkv"
    linked.parent.mkdir(parents=True)
    os.link(seeding, linked)                                    # Sonarr-style hardlink
    make_file(tmp / "media" / "tv" / "The Kitchen" / "Kuhnya.s01.e01.mkv")
    qb.torrents_list = [{**torrent("k" * 40, "Severance S01", "tv-sonarr"),
                         "content_path": "/media/downloads/Severance S01"}]
    res = w.import_folder(str(tmp / "media" / "tv"), "movies")
    assert res["torrent"] == ["Severance"] and res["added"] == ["The Kitchen"]


def test_inplace_space_check_counts_largest_file_only(tmp_path, monkeypatch):
    from jellyprep import executor, media
    plan = {"target": str(tmp_path), "actions": [
        {"src": "a", "dst": "a", "action": "process", "inplace": True, "size": 10 * 1024**3, "estimate": 10 * 1024**3},
        {"src": "b", "dst": "b", "action": "process", "inplace": True, "size": 12 * 1024**3, "estimate": 12 * 1024**3}]}
    assert executor.needed_bytes(plan) == 12 * 1024**3
    monkeypatch.setattr(media, "free_bytes", lambda p: 70 * 1024**3)
    executor.check_space(plan, min_free_gb=50)              # 70 - 12 >= 50: fits
