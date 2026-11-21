"""Ask a local language model (LM Studio / Ollama / any OpenAI-compatible server) to identify a release.

The model gets only names: file paths inside the torrent, TMDB candidates and their episode titles.
Its answer is validated and turned into the same overrides a person would enter on the review page,
so it always ends up waiting for confirmation.
"""
from __future__ import annotations

import json
import re
from pathlib import PurePath

import httpx

from .clients import TMDB, ClientError

TIMEOUT = httpx.Timeout(300.0, connect=5.0)
MAX_EPISODE_LINES = 400

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["tv", "movie"]},
        "tmdb_id": {"type": ["integer", "null"]},
        "title": {"type": "string"},
        "year": {"type": ["integer", "null"]},
        "episodes": {"type": "array", "items": {
            "type": "object",
            "properties": {"file": {"type": "string"}, "season": {"type": "integer"},
                           "episode": {"type": "integer"}, "episode_end": {"type": ["integer", "null"]}},
            "required": ["file", "season", "episode"]}},
        "comment": {"type": "string"},
    },
    "required": ["kind", "tmdb_id", "title", "episodes", "comment"],
}

SYSTEM = """You map video files from a torrent to a movie or a TV show on TMDB, for a Jellyfin library.
Rules:
- Use only TMDB ids from the candidates list; if none fits, set tmdb_id to null and give the title and year.
- For a TV show, return one entry per video file with its TMDB season and episode number.
  Russian releases often number by case/story ("3. Название (2 сер.)" = part 2 of story 3) or count
  episodes across seasons; use the TMDB episode titles to find the right season/episode.
  If one file contains several episodes, set episode_end to the last one.
- Copy file paths exactly as given. Do not invent files.
- Keep "comment" short: one sentence on how you decided, in Russian.
Answer with JSON only."""


class LLMError(Exception):
    pass


def configured(s: dict) -> bool:
    return s.get("llm_kind") in ("openai", "ollama") and bool(s.get("llm_url")) and bool(s.get("llm_model"))


def build_prompt(name: str, files: list[str], kind_hint: str, candidates: list[dict], episode_titles: dict) -> str:
    lines = [f"Torrent / folder name: {name}", f"Expected kind: {kind_hint or 'unknown'}", "", "Video files:"]
    lines += [f"- {f}" for f in files]
    lines += ["", "TMDB candidates:"]
    for c in candidates[:6]:
        lines.append(f"- id {c['id']}: {c.get('title')} / {c.get('original_title')} ({c.get('year') or '?'})"
                     + (f" — {c['overview'][:160]}" if c.get("overview") else ""))
    if not candidates:
        lines.append("- (none found)")
    budget = MAX_EPISODE_LINES
    for show_id, seasons in episode_titles.items():
        lines += ["", f"Episode titles of TMDB id {show_id}:"]
        for season, eps in seasons.items():
            if budget <= 0:
                lines.append("… (truncated)")
                break
            lines.append(f"Season {season}: " + "; ".join(f"E{e['episode']} {e['name']}" for e in eps[:budget]))
            budget -= len(eps)
    return "\n".join(lines)


def episode_titles_for(tmdb: TMDB | None, candidates: list[dict], seasons_hint: set[int]) -> dict:
    """Episode titles of the top TV candidates (the seasons the files seem to be in, else the first few)."""
    out: dict[int, dict[int, list[dict]]] = {}
    if tmdb is None:
        return out
    for c in candidates[:2]:
        try:
            sizes = tmdb.season_sizes(c["id"])
        except ClientError:
            continue
        wanted = sorted(n for n in sizes if n >= 1 and (not seasons_hint or n in seasons_hint)) or \
            sorted(n for n in sizes if n >= 1)[:4]
        out[c["id"]] = {}
        for n in wanted[:6]:
            try:
                out[c["id"]][n] = tmdb.season_episodes(c["id"], n)
            except ClientError:
                pass
    return out


def ask(s: dict, prompt: str) -> dict:
    kind, url, model = s["llm_kind"], s["llm_url"].rstrip("/"), s["llm_model"]
    try:
        if kind == "ollama":
            r = httpx.post(f"{url}/api/chat", timeout=TIMEOUT, json={
                "model": model, "stream": False, "format": SCHEMA, "options": {"temperature": 0},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
            r.raise_for_status()
            text = r.json()["message"]["content"]
        else:
            base = url if url.endswith("/v1") else f"{url}/v1"
            headers = {"Authorization": f"Bearer {s['llm_key']}"} if s.get("llm_key") else {}
            r = httpx.post(f"{base}/chat/completions", timeout=TIMEOUT, headers=headers, json={
                "model": model, "temperature": 0,
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "identification", "strict": True, "schema": SCHEMA}},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
    except httpx.HTTPError as e:
        raise LLMError(f"модель недоступна ({url}): {e}") from e
    except (KeyError, IndexError, ValueError) as e:
        raise LLMError(f"странный ответ сервера модели: {e}") from e
    return parse_answer(text)


def parse_answer(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise LLMError("модель ответила не JSON")
    try:
        return json.loads(m[0])
    except json.JSONDecodeError as e:
        raise LLMError(f"модель ответила неверным JSON: {e}") from e


def to_overrides(answer: dict, files: list[str], candidate_ids: set[int]) -> tuple[dict, str]:
    """Validate the model's answer and turn it into review-page overrides."""
    kind = answer.get("kind")
    if kind not in ("tv", "movie"):
        raise LLMError("модель не указала, фильм это или сериал")
    tmdb_id = answer.get("tmdb_id")
    choice: dict = {}
    if isinstance(tmdb_id, int) and tmdb_id in candidate_ids:
        choice = {"tmdb_id": tmdb_id}
    elif answer.get("title"):
        year = answer.get("year")
        choice = {"title": str(answer["title"])[:200], "year": year if isinstance(year, int) else None}
    comment = str(answer.get("comment") or "")[:300]
    if kind == "movie":
        return {"kind": "movie", "movies": {"0": choice}} if choice else {"kind": "movie"}, comment
    known = set(files)
    by_base = {PurePath(f).name: f for f in files}
    eps, seen = {}, set()
    for e in answer.get("episodes") or []:
        f = e.get("file")
        f = f if f in known else by_base.get(PurePath(str(f)).name)
        season, ep, end = e.get("season"), e.get("episode"), e.get("episode_end")
        if not f or not isinstance(season, int) or not isinstance(ep, int) or not (0 <= season < 100 and 0 < ep < 1000):
            continue
        if (season, ep) in seen:
            raise LLMError(f"модель дала два файла на серию S{season:02d}E{ep:02d}")
        seen.add((season, ep))
        eps[f] = [season, ep, end if isinstance(end, int) and end > ep else None]
    if not eps:
        raise LLMError("модель не сопоставила ни одного файла")
    ov = {"kind": "tv", "episodes": eps, "episodes_how": "модель"}
    if choice:
        ov["show"] = choice
    return ov, comment
