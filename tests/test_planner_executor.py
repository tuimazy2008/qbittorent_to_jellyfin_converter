import os

import pytest

from jellyprep import executor, identify, media, planner
from conftest import make_file


def test_movie_plan_and_hardlink(tmp_path, tmdb):
    src = make_file(tmp_path / "dl" / "Love.Actually.2003.1080p.BluRay.mkv")
    make_file(tmp_path / "dl" / "Love.Actually.2003.1080p.BluRay.rus.srt", 0)
    s = media.scan(src)
    # single-file torrent: the .srt next to it is outside the torrent and must not be picked up
    assert s.extras == [] or all("Love" in e for e in s.extras)
    ident = identify.identify(s, "movie", tmdb)
    lib = tmp_path / "library" / "movies"
    plan = planner.build(ident, s, lib, {"remux_containers": ["avi"]})
    assert plan["target"] == str(lib / "Реальная любовь (2003) [tmdbid-508]")
    video = plan["actions"][0]
    assert video["dst"].endswith("Реальная любовь (2003).mkv") and video["action"] == "hardlink"
    results = executor.execute(plan, log=lambda m: None)
    assert executor.verify(results) == []
    assert os.path.samefile(src, video["dst"])
    # second plan sees the file already there
    plan2 = planner.build(ident, s, lib, {"remux_containers": ["avi"]})
    assert plan2["actions"][0]["action"] == "exists" and not plan2["problems"]


def test_tv_plan_with_external_subs_and_audio(tmp_path, tmdb):
    root = tmp_path / "Severance (Season 1) BDRip 1080p"
    make_file(root / "Severance.S01E01.1080p.mkv")
    make_file(root / "Severance.S01E02.1080p.mkv")
    make_file(root / "Subs" / "Severance.S01E01.1080p.rus.srt", 0)
    make_file(root / "RUS Sound" / "Severance.S01E02.1080p.mka", 1)
    make_file(root / "info.txt", 0)
    s = media.scan(root)
    ident = identify.identify(s, "tv", tmdb)
    plan = planner.build(ident, s, tmp_path / "lib", {})
    dsts = sorted(os.path.relpath(a["dst"], plan["target"]) for a in plan["actions"])
    assert dsts == [
        "Season 01/Разделение S01E01.mkv",
        "Season 01/Разделение S01E01.rus.srt",
        "Season 01/Разделение S01E02.RUS_Sound.mka",
        "Season 01/Разделение S01E02.mkv",
    ]
    assert "info.txt" in plan["skipped"]


def test_conflicting_existing_file_is_a_problem(tmp_path, tmdb):
    src = make_file(tmp_path / "dl" / "Death Becomes Her.mkv")
    s = media.scan(src)
    ident = identify.identify(s, "movie", tmdb)
    lib = tmp_path / "lib"
    make_file(lib / "Смерть ей к лицу (1992) [tmdbid-598]" / "Смерть ей к лицу (1992).mkv")
    plan = planner.build(ident, s, lib, {})
    assert any("уже есть другие файлы" in p for p in plan["problems"])


def test_sanitize():
    assert planner.sanitize('A: B/C? "D"') == "A B C D"
    assert planner.sanitize("...") == "Unknown"


def test_not_enough_space(tmp_path, tmdb, monkeypatch):
    src = make_file(tmp_path / "dl" / "Death Becomes Her.avi", 100)
    s = media.scan(src)
    plan = planner.build(identify.identify(s, "movie", tmdb), s, tmp_path / "lib", {"remux_containers": ["avi"]})
    assert plan["actions"][0]["action"] == "remux"
    monkeypatch.setattr(media, "free_bytes", lambda p: 120 * 1024 * 1024)
    with pytest.raises(executor.NotEnoughSpace):
        executor.check_space(plan, min_free_gb=1)


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="ffmpeg not installed")
def test_remux_real_file(tmp_path):
    import subprocess
    src = tmp_path / "clip.avi"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=25",
                    "-f", "lavfi", "-i", "sine=duration=3", "-c:v", "mpeg4", "-c:a", "mp3", str(src)], check=True)
    dst = tmp_path / "out" / "clip.mkv"
    plan = {"target": str(tmp_path / "out"),
            "actions": [{"src": str(src), "dst": str(dst), "action": "remux", "size": src.stat().st_size}]}
    dst.parent.mkdir()
    results = executor.execute(plan, log=lambda m: None)
    assert results[0]["method"] == "remux" and dst.exists()
    assert abs(media.duration(dst) - 3) < 0.5
    assert executor.verify(results) == []
