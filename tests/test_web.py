import pytest
from fastapi.testclient import TestClient

from jellyprep import config, main, worker as worker_mod
from conftest import FakeQBit


@pytest.fixture
def client(db, tmp_path, monkeypatch):
    qb = FakeQBit()
    monkeypatch.setattr(worker_mod, "qbit_client", lambda s: qb)
    monkeypatch.setattr(worker_mod, "QBit", lambda *a, **k: qb)
    app = main.create_app(db, start_worker=False)
    with TestClient(app, base_url="http://jp.test") as c:
        c.db, c.qb, c.tmp = db, qb, tmp_path
        yield c


def test_first_run_redirects_to_wizard(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/setup/qbit"


def test_wizard_walkthrough(client):
    for step, _ in main.STEPS:
        r = client.get(f"/setup/{step}")
        assert r.status_code == 200, step
        assert "Рекомендуется" in r.text or step in ("qbit", "categories", "tmdb", "jellyfin"), step

    client.post("/setup/qbit", data={"qbit_url": "http://qb:8080", "qbit_user": "admin", "qbit_pass": "secret"})
    client.post("/setup/paths", data={"path_map_text": "/downloads -> /media/downloads",
                                      "library_root": str(client.tmp / "lib"), "link_fallback": "copy"})
    client.get("/setup/categories")  # imports categories from qBittorrent
    cats = client.db.categories()
    assert cats["tv-sonarr"]["enabled"] == 0 and cats["movies"]["enabled"] == 1
    assert cats["kids_series"]["kind"] == "tv"
    client.post("/setup/categories", data={
        "cat": ["movies", "tv-sonarr", "kids_series"],
        "enabled::movies": "1", "kind::movies": "movies", "dir::movies": "films", "approval::movies": "auto",
        "kind::tv-sonarr": "tv", "approval::tv-sonarr": "auto_if_confident",
        "enabled::kids_series": "1", "kind::kids_series": "tv", "approval::kids_series": "always_confirm"})
    client.post("/setup/cleanup", data={"preset": "r1d14"})
    client.post("/setup/limits", data={"min_free_choice": "100", "min_free_gb": "50", "confidence": "0.9",
                                       "poll_minutes": "15"})
    r = client.post("/setup/finish", data={"dry_run": "1"}, follow_redirects=False)
    assert r.headers["location"] == "/items"

    s = config.load(client.db)
    assert s["setup_done"] and s["dry_run"]
    assert s["qbit_pass"] == "secret"
    assert s["path_map"] == [{"from": "/downloads", "to": "/media/downloads"}]
    assert s["min_free_gb"] == 100 and s["confidence"] == 0.9 and s["poll_minutes"] == 15
    cats = client.db.categories()
    assert cats["movies"]["library_dir"] == "films" and cats["movies"]["approval"] == "auto"
    assert cats["movies"]["seed_ratio"] == 1.0 and cats["movies"]["seed_days"] == 14.0

    # an empty password field keeps the saved secret; secrets never reach the page
    client.post("/setup/qbit", data={"qbit_url": "http://qb:8080", "qbit_user": "admin", "qbit_pass": ""})
    assert config.load(client.db)["qbit_pass"] == "secret"
    for page in ("/items", "/categories", "/settings", "/log", "/setup/qbit", "/import"):
        r = client.get(page)
        assert r.status_code == 200 and "secret" not in r.text, page


def test_connection_test_endpoint(client):
    r = client.post("/api/test/qbit", data={"qbit_url": "http://qb:8080"})
    assert r.json()["ok"] and "v5.0.0" in r.json()["message"]
    r = client.post("/api/test/llm", data={"llm_kind": "none"})
    assert r.json()["ok"]


def test_cross_origin_post_refused(client):
    r = client.post("/setup/qbit", data={"qbit_url": "http://evil"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert config.load(client.db)["qbit_url"] != "http://evil"


def test_wizard_pages_survive_unreachable_qbittorrent(db):
    from jellyprep import config as cfg
    cfg.save(db, {"qbit_url": "http://127.0.0.1:9", "qbit_user": "admin", "qbit_pass": "x"})
    with TestClient(main.create_app(db, start_worker=False), base_url="http://jp.test") as c:
        for step in ("paths", "categories"):
            r = c.get(f"/setup/{step}")
            assert r.status_code == 200 and "недоступен" in r.text, step
        r = c.post("/api/test/qbit", data={}).json()
        assert not r["ok"] and "имя контейнера" in r["message"]


@pytest.mark.parametrize("status, body, cookie, ok", [
    (200, "Ok.", False, True),    # qBittorrent < 5.2
    (204, "", True, True),        # qBittorrent 5.2+
    (200, "Fails.", False, False),
    (401, "", False, False),
])
def test_qbit_login_versions(status, body, cookie, ok):
    import httpx
    from jellyprep.clients import ClientError, QBit

    def handler(request):
        if request.url.path.endswith("/auth/login"):
            headers = {"set-cookie": "QBT_SID_8080=abc; HttpOnly; SameSite=Strict; path=/"} if cookie else {}
            return httpx.Response(status, text=body, headers=headers)
        return httpx.Response(200, text="v5.2.3")

    qb = QBit("http://qb:8080", "admin", "pw")
    qb.http = httpx.Client(transport=httpx.MockTransport(handler))
    if ok:
        assert qb.version() == "v5.2.3"
    else:
        with pytest.raises(ClientError):
            qb.version()


class FakeJellyfin:
    def __init__(self, *a, **k):
        self.libs = FakeJellyfin.libs

    def libraries(self):
        return self.libs

    def add_library(self, name, ctype, path):
        FakeJellyfin.calls.append(("create", name, ctype, path))

    def add_path(self, lib, path):
        FakeJellyfin.calls.append(("add", lib, path))


def test_jellyfin_libraries_step(client, monkeypatch):
    lib = str(client.tmp / "lib")
    FakeJellyfin.libs = [{"Name": "Movies", "CollectionType": "movies", "Locations": ["/media/movies/adult_movies"]},
                         {"Name": "Kids", "CollectionType": "tvshows", "Locations": [f"{lib}/kids_series/"]}]
    FakeJellyfin.calls = []
    monkeypatch.setattr(main, "Jellyfin", FakeJellyfin)
    config.save(client.db, {"library_root": lib})
    client.get("/setup/categories")  # import categories (movies, kids_series enabled)

    r = client.post("/setup/jellyfin", data={"jellyfin_url": "http://jf:8096", "jellyfin_key": "k",
                                             "jf_path_map_text": "", "action": "stay"}, follow_redirects=False)
    assert r.headers["location"].startswith("/setup/jellyfin")
    page = client.get("/setup/jellyfin").text
    assert "уже в библиотеке «Kids»" in page            # kids_series folder is already a Jellyfin library
    assert 'name="jf_action::movies"' in page and "Добавить в «Movies»" in page

    r = client.post("/setup/jellyfin/libraries", data={
        "jf_cat": ["movies", "kids_series"], "jf_action::movies": "create",
        "jf_name::movies": "Фильмы (новые)", "jf_type::movies": "movies", "jf_action::kids_series": "skip"})
    assert FakeJellyfin.calls == [("create", "Фильмы (новые)", "movies", f"{lib}/movies")]
    assert (client.tmp / "lib" / "movies").is_dir()       # folder created before asking Jellyfin
    assert "создана библиотека" in r.text                 # shown once after the redirect
    assert "создана библиотека" not in client.get("/setup/jellyfin").text

    FakeJellyfin.calls = []
    client.post("/setup/jellyfin/libraries", data={"jf_cat": ["movies"], "jf_action::movies": "add:Movies"})
    assert FakeJellyfin.calls == [("add", "Movies", f"{lib}/movies")]


def test_jellyfin_client_requests():
    import httpx
    from jellyprep.clients import Jellyfin
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, dict(request.url.params), request.content.decode()))
        return httpx.Response(204)

    j = Jellyfin("http://jf:8096", "key")
    j.http = httpx.Client(transport=httpx.MockTransport(handler), headers=j.http.headers)
    j.add_library("Фильмы", "movies", "/media/library/movies")
    j.add_path("Movies", "/media/library/movies")
    j.add_library("Разное", "", "/media/library/x")
    assert seen[0][1] == "/Library/VirtualFolders" and seen[0][2]["collectionType"] == "movies"
    assert '"Path": "/media/library/movies"' in seen[0][3] or '"Path":"/media/library/movies"' in seen[0][3]
    assert seen[1][1] == "/Library/VirtualFolders/Paths" and "Movies" in seen[1][3]
    assert "collectionType" not in seen[2][2]
