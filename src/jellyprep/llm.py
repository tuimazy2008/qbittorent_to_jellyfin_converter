"""Ask a local language model (LM Studio / Ollama / any OpenAI-compatible server) to identify a release.

The model gets only names: file paths inside the torrent, TMDB candidates and their episode titles.
Its answer is validated and turned into the same overrides a person would enter on the review page,
so it always ends up waiting for confirmation.
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import PurePath

import httpx

from .clients import TMDB, ClientError

TIMEOUT = httpx.Timeout(300.0, connect=5.0)
MAX_EPISODE_LINES = 700
MAX_SECONDS = 20 * 60
MAX_REASONING_CHARS = 40_000     # a model that thinks longer than this is going in circles
MAX_TOKENS = 12_000

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["tv", "movie"]},
        "tmdb_id": {"type": ["integer", "null"]},
        "title": {"type": "string"},
        "year": {"type": ["integer", "null"]},
        "episodes": {"type": "array", "items": {
            "type": "object",
            "properties": {"file": {"type": "string"}, "season": {"type": ["integer", "null"]},
                           "episode": {"type": ["integer", "null"]}, "episode_end": {"type": ["integer", "null"]}},
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
- Every TMDB episode can be used only once. If a file has no matching TMDB episode (an extra, or the second
  part of an episode TMDB lists as one), set its season and episode to null and say so in "comment".
  The person will decide; you don't have to resolve every file.
- Copy file paths exactly as given. Do not invent files.
- Decide once and answer. Do not re-check the same lists again and again.
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
        regular = sorted(n for n in sizes if n >= 1 and sizes[n])
        # the seasons the files seem to be in first, then the rest: the guess may be wrong
        wanted = [n for n in regular if n in seasons_hint] + [n for n in regular if n not in seasons_hint]
        out[c["id"]] = {}
        for n in wanted[:8]:
            try:
                out[c["id"]][n] = tmdb.season_episodes(c["id"], n)
            except ClientError:
                pass
    return out


class LLMCancelled(LLMError):
    pass


@dataclass
class Conversation:
    """One exchange with the model, kept for the "conversation with the model" block."""
    model: str
    prompt: str
    system: str = SYSTEM
    reasoning: str = ""
    content: str = ""
    chunks: int = 0
    started: float = field(default_factory=time.time)
    finished: float | None = None
    status: str = "queued"           # queued | running | ok | error | cancelled
    error: str = ""
    usage: dict = field(default_factory=dict)
    images: list = field(default_factory=list)   # names of frames shown to the model
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    done: threading.Event = field(default_factory=threading.Event, repr=False)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("cancel", "done")}


FRAMES_NOTE = """
Frames from the main video file are attached (opening, middle, end credits). On-screen text — the title card,
the credits, studio and fund logos — is the most reliable clue: read it carefully. Put the decisive text you read
into "comment". If the title you read isn't among the candidates, give it with tmdb_id null (and the year only if
you are sure)."""


def supports_vision(s: dict) -> bool:
    """Ask the model server whether the configured model accepts images (cached per model)."""
    if s.get("llm_vision", "auto") == "off" or not configured(s):
        return False
    key = (s["llm_kind"], s["llm_url"], s["llm_model"])
    if key not in _VISION:
        _VISION[key] = _detect_vision(*key)
    return _VISION[key]


_VISION: dict[tuple, bool] = {}


def _detect_vision(kind: str, url: str, model: str) -> bool:
    url = url.rstrip("/")
    try:
        if kind == "ollama":
            r = httpx.post(f"{url}/api/show", json={"model": model}, timeout=10)
            return "vision" in (r.json().get("capabilities") or [])
        root = url[:-3] if url.endswith("/v1") else url
        r = httpx.get(f"{root}/api/v0/models/{model}", timeout=10)   # LM Studio's own API
        return r.json().get("type") == "vlm"
    except (httpx.HTTPError, ValueError):
        return False


VISION_SYSTEM = """You look at frames taken from one video (a film, or one episode of a TV series): the opening,
the middle and the end credits. Read the on-screen text exactly: the title card, credits (names, roles), studio,
fund and channel logos. Report what you READ in "title"; if no title is readable, leave it empty.
Separately, the file name may be an abbreviated title (vowels dropped, e.g. "LF_R_SMTHNG_LK_T" = "Life or
Something Like It", or transliterated): put your decoding of the file name into "title_from_name" ("" if none).
If the episode file names carry episode titles and you know which series they belong to, put the series title
into "title_from_name". Give the year only if you are sure. Put names, studios and other credits into "credits".
"comment": one sentence in Russian on what you saw. Answer with JSON only."""

VISION_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "original_title": {"type": "string"},
                   "title_from_name": {"type": "string"},
                   "year": {"type": ["integer", "null"]}, "credits": {"type": "array", "items": {"type": "string"}},
                   "comment": {"type": "string"}},
    "required": ["title", "original_title", "title_from_name", "year", "credits", "comment"],
}

CHOOSE_SYSTEM = """You pick which TMDB film or TV series a video is, using what was read from its frames, the file
names and the TMDB cards (year, runtime, director, creators, producers, cast, studios, networks, countries). Names and studios read in the credits are the
strongest evidence: a producer or studio from the credits found in a card decides it. TMDB runtimes are often
missing or wrong (especially for Russian films): never reject a card only because of the runtime.
Credits read from frames may contain OCR mistakes in names. If none of the cards fits, answer tmdb_id null.
"comment": one sentence in Russian explaining the choice. Answer with JSON only."""

CHOOSE_SCHEMA = {
    "type": "object",
    "properties": {"tmdb_id": {"type": ["integer", "null"]}, "comment": {"type": "string"}},
    "required": ["tmdb_id", "comment"],
}


def build_choose_prompt(file_name: str, duration: float, seen: dict, cards: list[dict],
                        other_files: list[str] | None = None) -> str:
    lines = [f"File: {file_name}, duration {int(duration // 60)} min."]
    if other_files:
        lines.append("Episode files: " + "; ".join(other_files[:15]))
    lines += [f"Read from the frames: title «{seen.get('title') or '?'}» / «{seen.get('original_title') or '?'}»,"
              f" from the file name «{seen.get('title_from_name') or '?'}», year {seen.get('year') or '?'}.",
              "Credits read: " + ("; ".join(seen.get("credits") or []) or "—"), "", "TMDB cards:"]
    for c in cards:
        bits = [f"{c['title']} / {c['original_title']} ({c['year'] or '?'}"
                + (f"–{c['last_year']}" if c.get("last_year") and c.get("last_year") != c.get("year") else "") + ")"]
        for key, label in (("runtime", "min"), ("seasons", "seasons"), ("episodes", "episodes")):
            if c.get(key):
                bits.append(f"{c[key]} {label}")
        for key, label in (("directors", "director"), ("creators", "created by"), ("producers", "producers"),
                           ("writers", "writers"), ("cast", "cast"), ("companies", "studios"),
                           ("networks", "networks"), ("countries", "countries")):
            vals = [v for v in c.get(key) or [] if v]
            if vals:
                bits.append(f"{label} {', '.join(vals)}")
        lines.append(f"- id {c['id']}: " + ", ".join(bits) + f". {c.get('overview') or ''}")
    return "\n".join(lines)


def ask(s: dict, prompt: str, conv: Conversation | None = None, images: list[bytes] | None = None,
        schema: dict | None = None, system: str | None = None) -> dict:
    """Stream the answer (so it can be cancelled and its progress shown); return the parsed JSON."""
    kind, url, model = s["llm_kind"], s["llm_url"].rstrip("/"), s["llm_model"]
    conv = conv or Conversation(model=model, prompt=prompt)
    user: dict = {"role": "user", "content": prompt}
    if images:
        b64 = [base64.b64encode(i).decode() for i in images]
        if kind == "ollama":
            user["images"] = b64
        else:
            user["content"] = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in b64]
    schema = schema or SCHEMA
    conv.system = system or SYSTEM
    messages = [{"role": "system", "content": conv.system}, user]
    try:
        if kind == "ollama":
            req = {"url": f"{url}/api/chat", "json": {"model": model, "stream": True, "format": schema,
                                                     "options": {"temperature": 0, "num_predict": MAX_TOKENS},
                                                     "messages": messages}}
            if s.get("llm_reasoning", "off") == "off":
                req["json"]["think"] = False
        else:
            base = url if url.endswith("/v1") else f"{url}/v1"
            req = {"url": f"{base}/chat/completions",
                   "headers": {"Authorization": f"Bearer {s['llm_key']}"} if s.get("llm_key") else {},
                   "json": {"model": model, "temperature": 0, "stream": True,
                            "stream_options": {"include_usage": True}, "max_tokens": MAX_TOKENS,
                            "response_format": {"type": "json_schema", "json_schema": {
                                "name": "identification", "strict": True, "schema": schema}},
                            "messages": messages}}
            if s.get("llm_reasoning", "off") == "off":
                req["json"]["reasoning_effort"] = "none"   # LM Studio: no thinking phase
        with httpx.stream("POST", timeout=TIMEOUT, **req) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if conv.cancel.is_set():
                    raise LLMCancelled("отменено")
                if time.time() - conv.started > MAX_SECONDS:
                    raise LLMError(f"модель думает дольше {MAX_SECONDS // 60} мин, прервано")
                if len(conv.reasoning) > MAX_REASONING_CHARS and not conv.content.strip():
                    raise LLMError("модель зациклилась: рассуждает по кругу и не отвечает. "
                                   "Посмотрите её рассуждения — обычно там видно, что её смутило")
                _feed(conv, kind, line)
    except LLMError as e:
        conv.status, conv.error, conv.finished = ("cancelled" if isinstance(e, LLMCancelled) else "error"), str(e), time.time()
        raise
    except httpx.HTTPError as e:
        conv.status, conv.error, conv.finished = "error", f"модель недоступна ({url}): {e}", time.time()
        raise LLMError(conv.error) from e
    conv.finished = time.time()
    if not conv.reasoning and (m := re.search(r"<think>(.*?)</think>", conv.content, re.S)):
        conv.reasoning = m[1].strip()   # models that think inside the answer text
    if not re.sub(r"<think>.*?</think>", "", conv.content, flags=re.S).strip():
        used = (conv.usage or {}).get("completion_tokens") or 0
        conv.status = "error"
        conv.error = ("модель исчерпала лимит ответа на рассуждения и не успела ответить — выключите "
                      "рассуждения модели в настройках или выберите другую модель"
                      if conv.reasoning and used >= MAX_TOKENS * 0.95 else "модель вернула пустой ответ")
        raise LLMError(conv.error)
    try:
        answer = parse_answer(conv.content)
    except LLMError as e:
        conv.status, conv.error = "error", str(e)
        raise
    conv.status = "ok"
    return answer


def _feed(conv: Conversation, kind: str, line: str) -> None:
    line = line.strip()
    if not line:
        return
    if kind != "ollama":
        if not line.startswith("data:"):
            return
        line = line[5:].strip()
        if line == "[DONE]":
            return
    try:
        obj = json.loads(line)
    except ValueError:
        return
    if kind == "ollama":
        msg = obj.get("message") or {}
        conv.content += msg.get("content") or ""
        conv.reasoning += msg.get("thinking") or ""
        if obj.get("done"):
            conv.usage = {"prompt_tokens": obj.get("prompt_eval_count"), "completion_tokens": obj.get("eval_count")}
    else:
        for ch in obj.get("choices") or []:
            delta = ch.get("delta") or {}
            conv.content += delta.get("content") or ""
            conv.reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
        if obj.get("usage"):
            conv.usage = obj["usage"]
    conv.chunks += 1


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
            continue  # the model left this file unmatched on purpose (or gave junk): the person decides
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
