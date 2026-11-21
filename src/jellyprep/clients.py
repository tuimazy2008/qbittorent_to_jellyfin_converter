"""HTTP clients: qBittorrent, TMDB, Jellyfin, local LLM (connection test only for now)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

TIMEOUT = httpx.Timeout(15.0, connect=5.0)
COMPLETE_STATES = {"uploading", "stalledUP", "forcedUP", "pausedUP", "stoppedUP", "queuedUP", "checkingUP"}


class ClientError(Exception):
    pass


# --- qBittorrent ---------------------------------------------------------
class QBit:
    def __init__(self, url: str, user: str = "", password: str = ""):
        self.base = url.rstrip("/") + "/api/v2"
        self.user, self.password = user, password
        self.http = httpx.Client(timeout=TIMEOUT, headers={"Referer": url.rstrip("/")})
        self._logged_in = False

    def _login(self) -> None:
        if not self.user:
            self._logged_in = True
            return
        try:
            r = self.http.post(f"{self.base}/auth/login",
                               data={"username": self.user, "password": self.password})
        except httpx.HTTPError as e:
            raise ClientError(_unreachable("qBittorrent", self.base, e)) from e
        # qBittorrent < 5.2: 200 "Ok." + cookie "SID"; 5.2+: 204 + cookie "QBT_SID_<port>".
        # Wrong password: 200 "Fails." or 401.
        has_sid = any("SID" in c.name for c in self.http.cookies.jar)
        ok = (r.status_code == 200 and r.text.strip() == "Ok.") or (r.status_code == 204 and has_sid)
        if r.status_code == 403:
            raise ClientError("qBittorrent временно заблокировал вход после неудачных попыток, подождите и попробуйте снова")
        if not ok:
            raise ClientError("qBittorrent: неверный логин или пароль")
        self._logged_in = True

    def _req(self, method: str, path: str, **kw: Any) -> httpx.Response:
        if not self._logged_in:
            self._login()
        try:
            r = self.http.request(method, f"{self.base}{path}", **kw)
            if r.status_code == 403:  # session expired
                self._logged_in = False
                self._login()
                r = self.http.request(method, f"{self.base}{path}", **kw)
        except httpx.HTTPError as e:
            raise ClientError(_unreachable("qBittorrent", self.base, e)) from e
        if r.status_code == 403:
            raise ClientError("qBittorrent: нужен логин и пароль")
        if r.status_code >= 400:
            raise ClientError(f"qBittorrent: HTTP {r.status_code} на {path}")
        return r

    def version(self) -> str:
        return self._req("GET", "/app/version").text.strip()

    def torrents(self) -> list[dict]:
        return self._req("GET", "/torrents/info").json()

    def categories(self) -> dict[str, dict]:
        return self._req("GET", "/torrents/categories").json()

    def default_save_path(self) -> str:
        return self._req("GET", "/app/preferences").json().get("save_path", "")

    def delete(self, torrent_hash: str, delete_files: bool) -> None:
        self._req("POST", "/torrents/delete",
                  data={"hashes": torrent_hash, "deleteFiles": "true" if delete_files else "false"})


def _unreachable(name: str, url: str, e: Exception, example: str = "http://qbittorrent:8080") -> str:
    msg = f"{name} недоступен по адресу {url.removesuffix('/api/v2')}: {type(e).__name__} {e}".strip()
    host = httpx.URL(url).host
    if host and host[0].isdigit():
        msg += (". Если сервисы в одной docker-сети, используйте имя контейнера вместо IP, "
                f"например {example} — контейнеру часто закрыт доступ к внешнему адресу своего же сервера.")
    return msg


def is_complete(t: dict) -> bool:
    return t.get("progress", 0) >= 1 and t.get("state") in COMPLETE_STATES


# --- TMDB ----------------------------------------------------------------
@dataclass
class Candidate:
    id: int
    title: str
    original_title: str
    year: int | None
    popularity: float = 0.0
    overview: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class TMDB:
    BASE = "https://api.themoviedb.org/3"

    def __init__(self, key: str, language: str = "ru-RU"):
        self.language = language
        headers, self.params = {}, {}
        if key.startswith("eyJ"):  # v4 read access token
            headers["Authorization"] = f"Bearer {key}"
        else:                      # v3 api key
            self.params["api_key"] = key
        self.http = httpx.Client(timeout=TIMEOUT, headers=headers)

    def _get(self, path: str, **params: Any) -> dict:
        try:
            r = self.http.get(f"{self.BASE}{path}", params={**self.params, **params})
        except httpx.HTTPError as e:
            raise ClientError(f"TMDB недоступен: {e}") from e
        if r.status_code == 401:
            raise ClientError("TMDB: ключ не подходит")
        if r.status_code >= 400:
            raise ClientError(f"TMDB: HTTP {r.status_code}")
        return r.json()

    def test(self) -> None:
        self._get("/configuration")

    def search(self, kind: str, query: str, year: int | None = None) -> list[Candidate]:
        if kind == "movie":
            params = {"query": query, "language": self.language}
            if year:
                params["year"] = year
            data = self._get("/search/movie", **params)
            return [Candidate(r["id"], r.get("title", ""), r.get("original_title", ""),
                              _year(r.get("release_date")), r.get("popularity", 0.0),
                              r.get("overview", "")) for r in data.get("results", [])[:10]]
        params = {"query": query, "language": self.language}
        if year:
            params["first_air_date_year"] = year
        data = self._get("/search/tv", **params)
        return [Candidate(r["id"], r.get("name", ""), r.get("original_name", ""),
                          _year(r.get("first_air_date")), r.get("popularity", 0.0),
                          r.get("overview", "")) for r in data.get("results", [])[:10]]

    def details(self, kind: str, tmdb_id: int) -> dict:
        path = f"/movie/{tmdb_id}" if kind == "movie" else f"/tv/{tmdb_id}"
        return self._get(path, language=self.language)

    def candidate(self, kind: str, tmdb_id: int) -> Candidate:
        d = self.details(kind, tmdb_id)
        if kind == "movie":
            return Candidate(d["id"], d.get("title", ""), d.get("original_title", ""),
                             _year(d.get("release_date")), d.get("popularity", 0.0), d.get("overview", ""))
        return Candidate(d["id"], d.get("name", ""), d.get("original_name", ""),
                         _year(d.get("first_air_date")), d.get("popularity", 0.0), d.get("overview", ""))

    def season_episodes(self, tmdb_id: int, season: int) -> list[dict]:
        """[{"episode": 1, "name": "Мягкая лапа смерти. Часть 1"}, ...]"""
        d = self._get(f"/tv/{tmdb_id}/season/{season}", language=self.language)
        return [{"episode": e["episode_number"], "name": e.get("name") or ""} for e in d.get("episodes", [])]

    def season_sizes(self, tmdb_id: int) -> dict[int, int]:
        d = self.details("tv", tmdb_id)
        return {s["season_number"]: s.get("episode_count", 0) for s in d.get("seasons", [])}


def _year(date: str | None) -> int | None:
    return int(date[:4]) if date and date[:4].isdigit() else None


# --- Jellyfin ------------------------------------------------------------
class Jellyfin:
    def __init__(self, url: str, key: str):
        self.base = url.rstrip("/")
        self.http = httpx.Client(timeout=TIMEOUT,
                                 headers={"Authorization": f'MediaBrowser Token="{key}"'})

    def _req(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            r = self.http.request(method, f"{self.base}{path}", **kw)
        except httpx.HTTPError as e:
            raise ClientError(_unreachable("Jellyfin", self.base, e, "http://jellyfin:8096")) from e
        if r.status_code == 401:
            raise ClientError("Jellyfin: ключ не подходит")
        if r.status_code >= 400:
            raise ClientError(f"Jellyfin: HTTP {r.status_code} {r.text[:200]}".strip())
        return r

    def add_library(self, name: str, collection_type: str, path: str) -> None:
        """Create a library. collection_type: movies | tvshows | "" (mixed). The folder must exist."""
        params = {"name": name, "refreshLibrary": "true"}
        if collection_type:
            params["collectionType"] = collection_type
        self._req("POST", "/Library/VirtualFolders", params=params,
                  json={"LibraryOptions": {"PathInfos": [{"Path": path}]}})

    def add_path(self, library: str, path: str) -> None:
        """Add a folder to an existing library."""
        self._req("POST", "/Library/VirtualFolders/Paths", params={"refreshLibrary": "true"},
                  json={"Name": library, "PathInfo": {"Path": path}})

    def test(self) -> str:
        info = self._req("GET", "/System/Info").json()
        return f'{info.get("ServerName", "")} {info.get("Version", "")}'.strip()

    def libraries(self) -> list[dict]:
        return self._req("GET", "/Library/VirtualFolders").json()

    def refresh(self) -> None:
        self._req("POST", "/Library/Refresh")


# --- Local LLM -----------------------------------------------------------
def llm_models(kind: str, url: str, key: str = "") -> list[str]:
    """List models on the server; raises ClientError when unreachable."""
    base = url.rstrip("/")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        if kind == "ollama":
            r = httpx.get(f"{base}/api/tags", timeout=TIMEOUT)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        r = httpx.get(f"{base}/v1/models" if not base.endswith("/v1") else f"{base}/models",
                      headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
    except (httpx.HTTPError, ValueError, KeyError) as e:
        raise ClientError(f"Модель недоступна по адресу {url}: {e}") from e
