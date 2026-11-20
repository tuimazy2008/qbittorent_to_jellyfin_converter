from jellyprep import identify, media
from conftest import make_file


def test_single_movie_matches_tmdb(tmp_path, tmdb):
    make_file(tmp_path / "Love.Actually.2003.1080p.BluRay.DTS.x264.rus(4).ukr.eng.mkv")
    s = media.scan(tmp_path / "Love.Actually.2003.1080p.BluRay.DTS.x264.rus(4).ukr.eng.mkv")
    ident = identify.identify(s, "movie", tmdb)
    m = ident["movies"][0]
    assert m["chosen"]["id"] == 508
    assert ident["confidence"] >= 0.9


def test_movie_without_year(tmp_path, tmdb):
    make_file(tmp_path / "Death Becomes Her.mkv")
    ident = identify.identify(media.scan(tmp_path / "Death Becomes Her.mkv"), "auto", tmdb)
    assert ident["kind"] == "movie"
    assert ident["movies"][0]["chosen"]["id"] == 598


def test_unknown_movie_goes_to_review(tmp_path, tmdb):
    make_file(tmp_path / "dom.mkv")
    ident = identify.identify(media.scan(tmp_path / "dom.mkv"), "movie", tmdb)
    assert ident["movies"][0]["chosen"] is None
    assert ident["confidence"] == 0


def test_tv_season_folder(tmp_path, tmdb):
    root = tmp_path / "Severance (Season 1) BDRip 1080p"
    for i in range(1, 10):
        make_file(root / f"Severance.S01E{i:02d}.1080p.BluRay.RGzsRutracker.mkv")
    make_file(root / "sample.mkv", 5)
    ident = identify.identify(media.scan(root), "auto", tmdb)
    assert ident["kind"] == "tv"
    assert ident["show"]["chosen"]["id"] == 95396
    assert [e["episode"] for e in ident["episodes"]] == list(range(1, 10))
    assert ident["confidence"] >= 0.8 and not ident["problems"]


def test_tv_russian_alt_title_and_season_in_name(tmp_path, tmdb):
    root = tmp_path / "Клиника (Scrubs) - 4 сезон (High Quality DVDRip)"
    for i in (1, 2, 3):
        make_file(root / f"Scrubs 4X{i:02d} - Название.mkv")
    ident = identify.identify(media.scan(root), "tv", tmdb)
    assert ident["show"]["chosen"]["id"] == 4556
    assert {e["season"] for e in ident["episodes"]} == {4}


def test_tv_sibling_season_dirs(tmp_path, tmdb):
    root = tmp_path / "Тайны следствия"
    for d in ("Тайны следствия", "Тайны следствия-2"):
        for i in (1, 2):
            make_file(root / d / f"{i:02d}. Серия.avi")
    ident = identify.identify(media.scan(root), "tv", tmdb)
    got = sorted((e["season"], e["episode"]) for e in ident["episodes"])
    assert got == [(1, 1), (1, 2), (2, 1), (2, 2)]


def test_tv_episode_beyond_tmdb_season_lowers_confidence(tmp_path, tmdb):
    root = tmp_path / "Severance (Season 1)"
    make_file(root / "Severance.S01E01.mkv")
    make_file(root / "Severance.S01E15.mkv")
    ident = identify.identify(media.scan(root), "tv", tmdb)
    assert any("всего 9" in p for p in ident["problems"])
    assert ident["confidence"] < 0.8


def test_manual_override(tmp_path, tmdb):
    make_file(tmp_path / "hns-tma.mkv")
    s = media.scan(tmp_path / "hns-tma.mkv")
    ident = identify.identify(s, "movie", tmdb, {"movies": {"0": {"title": "Hans Zimmer Live", "year": 2017}}})
    assert ident["movies"][0]["chosen"]["title"] == "Hans Zimmer Live"
    ident = identify.identify(s, "movie", tmdb, {"movies": {"0": {"tmdb_id": 550776}}})
    assert ident["movies"][0]["chosen"]["id"] == 550776


def test_without_tmdb_uses_guess_with_low_confidence(tmp_path):
    make_file(tmp_path / "Hope Gap.mkv")
    ident = identify.identify(media.scan(tmp_path / "Hope Gap.mkv"), "movie", None)
    assert ident["movies"][0]["chosen"]["title"] == "Hope Gap"
    assert ident["confidence"] == identify.NO_TMDB_CONFIDENCE


def test_absolute_numbering_is_split_into_tmdb_seasons(tmp_path, tmdb):
    root = tmp_path / "Царевны"
    for i in (1, 26, 27, 52, 53, 78):
        make_file(root / f"{i:02d}. Серия.avi")
    ident = identify.identify(media.scan(root), "tv", tmdb)
    got = [(e["season"], e["episode"]) for e in ident["episodes"]]
    assert got == [(1, 1), (1, 26), (2, 1), (2, 26), (3, 1), (3, 26)]
    assert any("Сквозная нумерация" in p for p in ident["problems"])
    assert not any("сезон 1" in p for p in ident["problems"])
    assert ident["confidence"] < 0.8          # always shown for confirmation


def test_absolute_numbering_fixes_already_laid_out_files(tmp_path, tmdb):
    root = tmp_path / "Царевны (2018) [tmdbid-82701]"
    for i in range(1, 79):
        make_file(root / "Season 01" / f"Царевны S01E{i:02d}.avi")
    ident = identify.identify(media.scan(root), "tv", tmdb)
    got = {(e["season"], e["episode"]) for e in ident["episodes"]}
    assert len(got) == 78 and {(1, 1), (2, 1), (3, 26)} <= got and max(got) == (3, 26)


def test_normal_season_is_not_remapped(tmp_path, tmdb):
    root = tmp_path / "Severance (Season 1)"
    for i in (1, 9):
        make_file(root / f"Severance.S01E{i:02d}.mkv")
    ident = identify.identify(media.scan(root), "tv", tmdb)
    assert [(e["season"], e["episode"]) for e in ident["episodes"]] == [(1, 1), (1, 9)]
    assert not ident["problems"]


def test_no_year_and_several_candidates_needs_confirmation(tmp_path, tmdb):
    from jellyprep.clients import Candidate
    tmdb.tv.append(Candidate(24756, "Radar", "Radar", 2002, 3))
    tmdb.tv.append(Candidate(317339, "Радар", "Radar", 2026, 3))
    tmdb.seasons.update({24756: {1: 10}, 317339: {1: 4}})
    root = tmp_path / "Radar"
    for i in (1, 2):
        make_file(root / "Season 1" / f"Radar.S01E{i:02d}.mkv")
    ident = identify.identify(media.scan(root), "tv", tmdb)
    assert ident["confidence"] <= 0.75
