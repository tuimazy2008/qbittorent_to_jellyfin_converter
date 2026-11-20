import pytest

from jellyprep.parse import guess_title, parse_episode, season_from_dirs, season_from_name


@pytest.mark.parametrize("name, season, episode", [
    ("Severance.S01E04.1080p.BluRay.RGzsRutracker.mkv", 1, 4),
    ("Scrubs 2X03 - Мое Массовое Убийство (My Overkill).mkv", 2, 3),
    ("Гражданин.начальник.01.серия.2001.DVDRip-AVC.kvvk_2009.mkv", None, 1),  # guessit alone says season 2009
    ("28. Секрет.avi", None, 28),
    ("Серия 7.mkv", None, 7),
    ("Episode 12 - Foo.mkv", None, 12),
    ("Смешарики. Пин-код - 05. Название.avi", None, 5),
])
def test_parse_episode(name, season, episode):
    s, e, how = parse_episode(name)
    assert (s, e) == (season, episode), how


def test_no_episode():
    assert parse_episode("Богатыри - Илья Муромец и Соловей-Разбойник.avi")[1] is None


@pytest.mark.parametrize("name, season", [
    ("Клиника (Scrubs) - 4 сезон (High Quality DVDRip)", 4),
    ("Severance (Season 1) BDRip 1080p", 1),
    ("Vstat.na.nogi.S01.2026.WEB-DL.1080p.ExKinoRay", 1),
    ("Сезон 3", 3),
    ("Тайны следствия", None),
])
def test_season_from_name(name, season):
    assert season_from_name(name) == season


def test_season_from_sibling_dirs():
    sib = {"": ["Тайны следствия", "Тайны следствия-2"]}
    assert season_from_dirs("Тайны следствия-2/01.avi", sib) == 2
    assert season_from_dirs("Тайны следствия/01.avi", sib) == 1
    assert season_from_dirs("Extras/01.avi", {"": ["Extras"]}) is None


def test_guess_title_alternatives():
    title, year, alts = guess_title("Клиника (Scrubs) - 4 сезон (High Quality DVDRip)")
    assert title == "Клиника" and "Scrubs" in alts
    title, year, alts = guess_title("Love.Actually.2003.1080p.BluRay.DTS.x264.rus(4).ukr.eng.mkv")
    assert (title, year) == ("Love Actually", 2003)


def test_release_group_year_is_not_the_show_year():
    title, year, _ = guess_title("Гражданин.начальник.2001.DVDRip-AVC.kvvk_2009")
    assert (title, year) == ("Гражданин начальник", 2001)
    assert guess_title("Blade Runner 2049 (2017)")[0] == "Blade Runner 2049"
