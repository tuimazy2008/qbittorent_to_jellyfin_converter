from jellyprep import identify, media, planner
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


def test_episodes_matched_by_tmdb_titles(tmp_path, tmdb):
    root = tmp_path / "Тайны следствия"
    for name in ("1. Мягкая лапа смерти (1 сер.).avi", "1. Мягкая лапа смерти (2 сер.).avi",
                 "2. Гроб на две персоны (1 сер.).avi", "2. Гроб на две персоны (2 сер.).avi"):
        make_file(root / "Тайны следствия" / name)
    for name in ("1. Заказчик (1,2 сер.).avi", "2. Падение.avi", "3. Марионетки.avi"):
        make_file(root / "Тайны следствия-2" / name)
    s = media.scan(root)
    ident = identify.identify(s, "tv", tmdb)
    got = {e["file"].split("/")[-1]: (e["season"], e["episode"], e.get("episode_end")) for e in ident["episodes"]}
    assert got["1. Мягкая лапа смерти (2 сер.).avi"] == (1, 2, None)
    assert got["2. Гроб на две персоны (1 сер.).avi"] == (1, 3, None)
    assert got["1. Заказчик (1,2 сер.).avi"] == (2, 1, 2)
    assert got["3. Марионетки.avi"] == (2, 4, None)
    assert not any("Повторяются" in p for p in ident["problems"])
    assert any("по названиям" in p for p in ident["problems"])          # still shown for confirmation
    plan = planner.build(ident, s, tmp_path / "lib", {})
    names = sorted(a["dst"].split("/")[-1] for a in plan["actions"])
    assert "Тайны следствия S02E01-E02.avi" in names and not plan["problems"]


def _smeshariki(tmdb):
    from jellyprep.clients import Candidate
    tmdb.tv += [Candidate(74405, "Смешарики: Пин-код", "Смешарики: Пин-код", 2012, 5),
                Candidate(107951, "Смешарики: Пин-код 2.0", "Смешарики: Пин-код 2.0", 2017, 1)]
    tmdb.seasons.update({74405: {0: 1, 1: 8, 2: 3}, 107951: {1: 3}})
    tmdb.episodes.update({
        (74405, 1): ["Почему?", "Принцесса в поиске", "Наноняни", "Метатыква", "ДНК",
                     "Отель «У весёлого альпиниста»", "Ураган на спор", "Силуэт на снегу"],
        (74405, 2): ["Баранка", "Двигатель прогресса", "Лучший из миров"],
        (107951, 1): ["Первая", "Вторая", "Третья"]})


def test_tmdb_titles_beat_release_numbering(tmp_path, tmdb):
    _smeshariki(tmdb)
    root = tmp_path / "Смешарики Пин-код [rutracker.org]"
    for name in ("01 Наноняни .avi", "02 Принцесса в поиске.avi", "03 Силуэт на снегу.avi",
                 "05 Отель У весёлого альпениста.avi", "06 ДНК.avi"):
        make_file(root / name)
    ident = identify.identify(media.scan(root), "tv", tmdb, {"show": {"tmdb_id": 74405}})
    got = {e["file"]: (e["season"], e["episode"]) for e in ident["episodes"]}
    assert got["01 Наноняни .avi"] == (1, 3) and got["03 Силуэт на снегу.avi"] == (1, 8)
    assert got["05 Отель У весёлого альпениста.avi"] == (1, 6) and got["06 ДНК.avi"] == (1, 5)
    assert any("не по номерам в именах" in p for p in ident["problems"])


def test_titles_find_the_right_show_and_season(tmp_path, tmdb):
    _smeshariki(tmdb)
    root = tmp_path / "Смешарики Пин-код 2.0"
    for name in ("01 Баранка.mp4", "02 Двигатель прогресса.mp4", "03 Лучьший из миров.mp4"):
        make_file(root / name)
    ident = identify.identify(media.scan(root), "tv", tmdb)
    assert ident["show"]["chosen"]["id"] == 74405
    assert sorted((e["season"], e["episode"]) for e in ident["episodes"]) == [(2, 1), (2, 2), (2, 3)]
    assert any("По названиям серий это сериал" in p or "не по номерам" in p for p in ident["problems"])


def test_titles_switch_show_when_first_candidate_has_no_such_episodes(tmp_path, tmdb):
    _smeshariki(tmdb)
    root = tmp_path / "Смешарики Пин-код 2.0"
    for name in ("01 Баранка.mp4", "02 Двигатель прогресса.mp4", "03 Лучьший из миров.mp4"):
        make_file(root / name)
    s = media.scan(root)
    show = {"chosen": {"id": 107951, "title": "Смешарики: Пин-код 2.0", "year": 2017}, "score": 0.9,
            "candidates": [{"id": 107951, "title": "Смешарики: Пин-код 2.0", "year": 2017},
                           {"id": 74405, "title": "Смешарики: Пин-код", "year": 2012}]}
    eps = [{"file": f, "season": 1, "episode": n + 1, "how": "guessit, сезон не найден → 1"} for n, f in enumerate(s.videos)]
    matched, changed, switched = identify._match_episode_titles(eps, [], tmdb, show, set(), show_fixed=False)
    assert switched["id"] == 74405 and show["chosen"]["id"] == 74405 and matched == 3
    assert [(e["season"], e["episode"]) for e in eps] == [(2, 1), (2, 2), (2, 3)]


def test_closest_title_wins_and_part_word_in_brackets(tmp_path, tmdb):
    _smeshariki(tmdb)
    tmdb.episodes[(74405, 1)] += ["Бортовой дневник", "Бортовой дневник 2", "День Биби. Часть 1", "День Биби. Часть 2"]
    tmdb.seasons[74405][1] = 12
    root = tmp_path / "Смешарики Пин-код"
    for name in ("32 Бортовой дневник.avi", "44 Бортовой дневник 2.avi", "09 День Биби (часть 1).mp4",
                 "16 День Биби (часть 2).mp4", "01 Наноняни .avi"):
        make_file(root / name)
    ident = identify.identify(media.scan(root), "tv", tmdb, {"show": {"tmdb_id": 74405}})
    got = {e["file"]: (e["season"], e["episode"]) for e in ident["episodes"]}
    assert got["32 Бортовой дневник.avi"] == (1, 9) and got["44 Бортовой дневник 2.avi"] == (1, 10)
    assert got["09 День Биби (часть 1).mp4"] == (1, 11) and got["16 День Биби (часть 2).mp4"] == (1, 12)



def test_unmatched_titles_drop_their_unreliable_numbers(tmp_path, tmdb):
    _smeshariki(tmdb)
    root = tmp_path / "Смешарики Пин-код"
    for name in ("01 Наноняни .avi", "02 Принцесса в поиске.avi", "03 Силуэт на снегу.avi", "08 Энцелад.avi"):
        make_file(root / name)
    ident = identify.identify(media.scan(root), "tv", tmdb, {"show": {"tmdb_id": 74405}})
    assert ident["unmatched"] == ["08 Энцелад.avi"]                 # its "08" would collide with «Силуэт на снегу»
    assert {e["file"]: e["episode"] for e in ident["episodes"]}["03 Силуэт на снегу.avi"] == 8
    assert not any("Повторяются" in p for p in ident["problems"])


def test_no_year_and_several_movies_always_reviewed(tmp_path, tmdb):
    from jellyprep.clients import Candidate
    tmdb.movies += [Candidate(1401377, "Dom", "Dom", 2024, 1), Candidate(70813, "Dom", "Dom", 2003, 1)]
    f = make_file(tmp_path / "dom.mkv")
    ident = identify.identify(media.scan(f), "movie", tmdb)
    assert any(p.startswith(identify.NO_YEAR_PROBLEM) for p in ident["problems"])
