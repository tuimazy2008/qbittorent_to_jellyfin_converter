"""Web lookups for the model: Wikipedia search (ru + en).

Local models don't know less famous titles and invent answers, so instead of trusting their memory we let
them search: they propose queries, we run them on Wikipedia, turn the article titles into TMDB searches and
let the model pick among real TMDB entries. Only search phrases leave the server.
"""
from __future__ import annotations

import html
import re

import httpx

UA = {"User-Agent": "jellyprep/0.2 (self-hosted media organizer)"}
TIMEOUT = httpx.Timeout(15.0, connect=5.0)
ARTICLE_NOISE = re.compile(r"\s*\((?:[^)]*(?:season|сезон|film|фильм|tv series|телесериал|сериал|мультфильм|"
                           r"мультсериал|miniseries|episode|эпизод|\d{4})[^)]*)\)\s*$", re.I)
LIST_PAGE = re.compile(r"(?i)^(list of|список)\b")


def wiki_search(query: str, lang: str = "en", limit: int = 4) -> list[dict]:
    try:
        r = httpx.get(f"https://{lang}.wikipedia.org/w/api.php", headers=UA, timeout=TIMEOUT, params={
            "action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json"})
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
    except (httpx.HTTPError, ValueError):
        return []
    return [{"lang": lang, "title": h["title"],
             "snippet": html.unescape(re.sub(r"<[^>]+>", "", h.get("snippet", "")))} for h in hits]


def search_all(queries: list[str], langs: tuple[str, ...] = ("en", "ru"), cap: int = 16) -> list[dict]:
    seen, out = set(), []
    for q in queries:
        for lang in langs:
            for h in wiki_search(q, lang):
                key = (h["lang"], h["title"])
                if key not in seen:
                    seen.add(key)
                    out.append({**h, "query": q})
            if len(out) >= cap:
                return out
    return out


def names_from_articles(results: list[dict]) -> list[str]:
    """'Curb Your Enthusiasm (season 1)' → 'Curb Your Enthusiasm'; 'List of … episodes' → the show."""
    names = []
    for r in results:
        t = r["title"]
        m = re.match(r"(?i)^(?:list of\s+)?(.+?)\s+episodes$", t) or re.match(r"(?i)^список\s+(?:эпизодов|серий)\s+(.+)$", t)
        if m:
            t = m[1]
        elif LIST_PAGE.match(t):
            continue
        t = ARTICLE_NOISE.sub("", t).strip(" «»\"")
        if t and t not in names:
            names.append(t)
    return names


def quoted(titles: list[str], n: int = 3) -> str:
    """'"The Pants Tent" "Ted and Mary"' — episode titles together find the show's season article."""
    return " ".join(f'"{t}"' for t in titles[:n] if t)
