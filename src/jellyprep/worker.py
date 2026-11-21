"""Background work.

Two threads:
- the main loop syncs torrents, identifies them, does the light work (hardlinks) and cleans up after seeding;
- the encoder takes heavy jobs (re-encode, audio processing, remux) one at a time. Re-encoding waits for
  the night window unless the user pressed "run now".
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import os
import threading
import time
import traceback
from pathlib import Path, PurePath
from typing import Any

from . import config, executor, identify, llm, media, parse, planner, transcode
from .clients import QBit, TMDB, ClientError, Jellyfin, is_complete
from .db import APPROVED, DB, DONE, ERROR, NEW, REMOVED, REVIEW, RUNNING, SKIPPED

log = logging.getLogger("jellyprep")

KIND_HINT = {"movies": "movie", "tv": "tv", "auto": "auto"}
LLM_THINKING = "модель думает…"
LLM_QUEUED = "в очереди к модели"
FRAMES_DIR = config.CONFIG_DIR / "frames"
CATEGORY_PROBLEM = "Категория"


def _show_is_weak(ident: dict) -> bool:
    """Nobody really knows which show this is (generic folder name, no year and several matches, …)."""
    chosen = (ident.get("show") or {}).get("chosen") or {}
    weak = (identify.SHOW_UNKNOWN_PROBLEM, identify.NO_YEAR_PROBLEM, "В TMDB у сериала нет сезона")
    return not chosen.get("id") or any(p.startswith(weak) for p in ident.get("problems") or [])


STUDIO_WORDS = re.compile(r"(?i)pictures|entertainment|production|productions|films?|studios?|fox|warner|universal|"
                          r"paramount|columbia|metro|disney|dreamworks|miramax|lionsgate|regency|мосфильм|ленфильм|"
                          r"ств|киностудия|кинокомпания")


def _by_studios(tmdb: TMDB, credits: list[str], year: int | None) -> list:
    """Films of the studios read in the credits ("20th Century Fox", "Davis Entertainment") released that year."""
    ids: list[int] = []
    for line in credits:
        if not STUDIO_WORDS.search(line):
            continue
        for part in re.split(r"[/,]| A | AN ", line):
            name = re.sub(r"(?i)\b(a|an|the|presents?|in association with|production|productions|company)\b", " ", part)
            name = " ".join(name.split())
            if len(name) >= 4:
                try:
                    ids += tmdb.companies(name)
                except ClientError:
                    pass
    ids = list(dict.fromkeys(ids))[:6]
    if not ids:
        return []
    try:
        return tmdb.discover_movies(ids, year)
    except ClientError:
        return []


def category_hint(kind: str, cat: dict, cats: dict[str, dict]) -> str | None:
    """A series in a films category (or the other way round): which category would fit better."""
    want = {"tv": "tv", "movie": "movies"}.get(kind)
    if not want or cat.get("kind") not in ("movies", "tv") or cat["kind"] == want:
        return None
    words = set(re.split(r"[\W_]+", cat["name"].lower())) - {"movies", "series", "tv", ""}

    def score(c: dict) -> tuple:
        cw = set(re.split(r"[\W_]+", c["name"].lower()))
        return (len(cw & words), bool(c["enabled"]))

    fits = [c for c in cats.values() if c["kind"] == want]
    return max(fits, key=score)["name"] if fits else None


def set_category(db: DB, item_id: int, category: str) -> None:
    item = db.item(item_id)
    if not item or category not in db.categories() or item["state"] not in (NEW, REVIEW, ERROR, APPROVED):
        return
    db.update_item(item_id, category=category, state=REVIEW if item["state"] == APPROVED else item["state"])
    db.log(f"Категория изменена: {item['category']} → {category}", item_id=item_id)


INFO_PROBLEMS = ("Серии сопоставлены по названиям", "Сквозная нумерация", "Нет года в имени")


def map_path(path: str, path_map: list[dict]) -> str:
    """Translate a path as qBittorrent sees it into a path inside this container."""
    best = None
    for m in path_map or []:
        src = m.get("from", "").rstrip("/")
        if src and (path == src or path.startswith(src + "/")) and (best is None or len(src) > len(best["from"])):
            best = {**m, "from": src}
    if best is None:
        return path
    return best["to"].rstrip("/") + path[len(best["from"]):]


def dest_root(settings: dict, cat: dict) -> Path:
    d = cat.get("library_dir") or cat["name"]
    p = Path(d)
    return p if p.is_absolute() else Path(settings["library_root"]) / d


def deletion_due(cat: dict | None, ratio: float, seeding_time: int) -> tuple[bool, str]:
    if not cat or not cat.get("delete_enabled"):
        return False, "удаление выключено"
    r, d = cat.get("seed_ratio"), cat.get("seed_days")
    if r is None and d is None:
        return False, "условия удаления не заданы"
    days = seeding_time / 86400
    if r is not None and ratio >= r:
        return True, f"рейтинг {ratio:.2f} ≥ {r:g}"
    if d is not None and days >= d:
        return True, f"раздаётся {days:.0f} дн. ≥ {d:g}"
    parts = []
    if r is not None:
        parts.append(f"рейтинг {ratio:.2f} из {r:g}")
    if d is not None:
        parts.append(f"{days:.0f} из {d:g} дн.")
    return False, ", ".join(parts)


def in_window(s: dict, now: dt.datetime | None = None) -> bool:
    if not s.get("night_only"):
        return True
    now = now or dt.datetime.now()
    try:
        start = dt.time.fromisoformat(s["night_start"])
        end = dt.time.fromisoformat(s["night_end"])
    except (ValueError, KeyError):
        return True
    t = now.time()
    return start <= t < end if start <= end else (t >= start or t < end)


def is_heavy(plan: dict | None) -> bool:
    return bool(plan) and any(a["action"] in planner.HEAVY for a in plan["actions"])


def needs_window(plan: dict | None) -> bool:
    return bool(plan) and any(a["action"] == "transcode" for a in plan["actions"])


def tmdb_client(s: dict) -> TMDB | None:
    return TMDB(s["tmdb_key"], s.get("tmdb_language") or "ru-RU") if s.get("tmdb_key") else None


def qbit_client(s: dict) -> QBit:
    return QBit(s["qbit_url"], s.get("qbit_user", ""), s.get("qbit_pass", ""))


def safe_probe(path: str) -> dict | None:
    try:
        return transcode.probe(path)
    except Exception:  # noqa: BLE001 — unreadable file: the planner reports it
        return None


def compare_with_rules(ident: dict, overrides: dict) -> tuple[bool, list[str]]:
    """Does the model's answer say exactly what the rules found? Returns (agree, differences)."""
    diff: list[str] = []
    kind = ident.get("kind")
    if overrides.get("kind") != kind:
        return False, [f"модель считает это {'сериалом' if overrides.get('kind') == 'tv' else 'фильмом'}"]
    if kind == "movie":
        movies = ident.get("movies") or []
        rules_id = ((movies[0].get("chosen") or {}).get("id")) if len(movies) == 1 else None
        model_id = ((overrides.get("movies") or {}).get("0") or {}).get("tmdb_id")
        if not rules_id or rules_id != model_id:
            diff.append(f"фильм: TMDB {rules_id}, модель {model_id}")
        return not diff, diff
    rules_id = ((ident.get("show") or {}).get("chosen") or {}).get("id")
    model_id = (overrides.get("show") or {}).get("tmdb_id")
    if not rules_id or rules_id != model_id:
        diff.append(f"сериал: TMDB {rules_id}, модель {model_id}")
    rules = {e["file"]: (e["season"], e["episode"], e.get("episode_end")) for e in ident.get("episodes") or []}
    model = {f: tuple(v) for f, v in (overrides.get("episodes") or {}).items()}
    for f in sorted(set(rules) | set(model)):
        if rules.get(f) != model.get(f):
            show = lambda v: "нет" if not v else f"S{v[0]:02d}E{v[1]:02d}" + (f"-E{v[2]:02d}" if v[2] else "")  # noqa: E731
            diff.append(f"{PurePath(f).name}: TMDB {show(rules.get(f))}, модель {show(model.get(f))}")
    if ident.get("unmatched"):
        diff.append(f"без номера у правил: {len(ident['unmatched'])}")
    return not diff, diff


class Worker:
    def __init__(self, db: DB):
        self.db = db
        self._wake = threading.Event()
        self._enc_wake = threading.Event()
        self._lock = threading.Lock()
        self.status: dict = {"last_run": None, "last_error": None, "running": False, "job": None,
                             "job_started": None}
        self._downloads: tuple[float, list[dict]] = (0.0, [])
        self._llm_lock = threading.Lock()
        # queued and running conversations (insertion order = queue order); one request to the model at a time
        self.llm_jobs: dict[int, llm.Conversation] = {}
        self._llm_queue: list[tuple[int, bool]] = []
        self._llm_cv = threading.Condition(self._llm_lock)
        self._llm_thread: threading.Thread | None = None

    # --- threads ----------------------------------------------------------
    def start(self) -> None:
        self.recover()
        threading.Thread(target=self._loop, daemon=True, name="worker").start()
        threading.Thread(target=self._encoder_loop, daemon=True, name="encoder").start()

    def trigger(self) -> None:
        self._wake.set()
        self._enc_wake.set()

    def recover(self) -> None:
        """After a restart, jobs that were running go back to the queue; half-written files are removed."""
        for item in self.db.query("SELECT id FROM items WHERE note IN (?, ?)", (LLM_THINKING, LLM_QUEUED)):
            self.db.execute("UPDATE items SET note=? WHERE id=?",
                            ("модель: запрос прерван перезапуском сервиса", item["id"]))
        for item in self.db.items(RUNNING):
            for a in (item.get("plan") or {}).get("actions", []):
                Path(a["dst"]).with_name(Path(a["dst"]).stem + ".partial.mkv").unlink(missing_ok=True)
            self.db.update_item(item["id"], state=APPROVED)
            self.db.log("Прервано перезапуском, задача снова в очереди", "warning", item["id"])

    def _loop(self) -> None:
        while True:
            s = config.load(self.db)
            if s["setup_done"]:
                self.run_cycle()
            self._wake.wait(timeout=max(1, float(s["poll_minutes"])) * 60)
            self._wake.clear()

    def _encoder_loop(self) -> None:
        while True:
            try:
                while self.run_next_heavy():
                    pass
            except Exception as e:  # noqa: BLE001
                self.db.log(f"Ошибка очереди обработки: {e}", "error")
                log.exception("encoder failed")
            self._enc_wake.wait(timeout=60)
            self._enc_wake.clear()

    # --- main cycle -------------------------------------------------------
    def run_cycle(self) -> None:
        if not self._lock.acquire(blocking=False):
            return
        self.status["running"] = True
        try:
            s = config.load(self.db)
            qb = qbit_client(s)
            torrents = {t["hash"]: t for t in qb.torrents()}
            self.sync(s, torrents)
            for item in self.db.items(NEW):
                self.analyze(item["id"])
            if llm.configured(s):
                for item in self.db.items(REVIEW):  # model was offline last time: try again
                    if (item.get("trace") or {}).get("llm_pending"):
                        self.ask_llm(item["id"], auto=True)
            changed = False
            if not s["dry_run"]:
                for item in self.db.items(APPROVED):
                    if not is_heavy(item["plan"]) and self.db.claim(item["id"], APPROVED, RUNNING):
                        changed |= self.run_item(item["id"])
            if changed:
                self.refresh_jellyfin(s)
            self.cleanup(s, qb, torrents)
            self.status["last_error"] = None
        except Exception as e:  # keep the loop alive
            self.status["last_error"] = str(e)
            self.db.log(f"Ошибка цикла: {e}", "error")
            log.exception("cycle failed")
        finally:
            self.status["running"] = False
            self.status["last_run"] = time.time()
            self._lock.release()
            self._enc_wake.set()

    def run_next_heavy(self) -> bool:
        """Run one heavy job if one is allowed now. Returns True if something ran."""
        s = config.load(self.db)
        if not s["setup_done"] or s["dry_run"]:
            return False
        window = in_window(s)
        for item in reversed(self.db.items(APPROVED)):  # oldest first
            plan = item["plan"]
            if not is_heavy(plan):
                continue
            if needs_window(plan) and not window and not plan.get("run_now"):
                continue
            if not self.db.claim(item["id"], APPROVED, RUNNING):
                continue
            ok = self.run_item(item["id"])
            if ok:
                self.refresh_jellyfin(s)
            return True
        return False

    def probe_cached(self, path: Path) -> dict | None:
        """ffprobe summary, cached per file (path + size + mtime)."""
        try:
            st = path.stat()
        except OSError:
            return None
        cached = self.db.probe_get(str(path), st.st_size, st.st_mtime)
        if cached is not False:
            return cached
        info = safe_probe(str(path))
        self.db.probe_set(str(path), st.st_size, st.st_mtime, info)
        return info

    def media_hints(self, sc: media.Scan) -> list[str]:
        """Names written inside the main video file (tags), for releases with useless file names."""
        if not sc.videos:
            return []
        main = max(sc.videos, key=lambda v: sc.sizes.get(v, 0))
        return parse.hints_from_info(self.probe_cached(sc.base / main))

    def downloads(self) -> list[dict]:
        """Torrents still downloading in enabled categories (cached for a few seconds)."""
        ts, cached = self._downloads
        if time.time() - ts < 5:
            return cached
        s = config.load(self.db)
        enabled = {n for n, c in self.db.categories().items() if c["enabled"]}
        try:
            out = [t for t in qbit_client(s).torrents()
                   if t.get("category") in enabled and not is_complete(t) and t.get("progress", 0) < 1]
        except Exception:  # noqa: BLE001 — the board must render without qBittorrent
            out = cached
        self._downloads = (time.time(), out)
        return out

    # --- steps ----------------------------------------------------------
    def sync(self, s: dict, torrents: dict[str, dict]) -> None:
        cats = self.db.categories()
        known = self.db.known_hashes()
        for h, t in torrents.items():
            cat = cats.get(t.get("category") or "")
            if h in known or not cat or not cat["enabled"] or not is_complete(t):
                continue
            path = map_path(t["content_path"], s["path_map"])
            item_id = self.db.add_item(hash=h, name=t["name"], category=t.get("category") or "",
                                       content_path=path, ratio=t.get("ratio"),
                                       seeding_time=t.get("seeding_time"))
            self.db.log(f"Новый торрент: {t['name']}", item_id=item_id)
        for item in self.db.items():
            if t := torrents.get(item["hash"] or ""):
                if (t.get("ratio"), t.get("seeding_time")) != (item["ratio"], item["seeding_time"]):
                    self.db.execute("UPDATE items SET ratio=?, seeding_time=? WHERE id=?",
                                    (t.get("ratio"), t.get("seeding_time"), item["id"]))

    def analyze(self, item_id: int, overrides: dict | None = None) -> None:
        item = self.db.item(item_id)
        s = config.load(self.db)
        cat = self.db.categories().get(item["category"]) or {"name": item["category"], "kind": "auto",
                                                              "approval": "always_confirm", "library_dir": ""}
        if overrides is None:
            overrides = (item.get("ident") or {}).get("overrides") or {}
        try:
            sc = media.scan(item["content_path"])
            ident = identify.identify(sc, KIND_HINT.get(cat["kind"], "auto"), tmdb_client(s), overrides,
                                      hints=self.media_hints(sc))
            ident["overrides"] = overrides
            hint = category_hint(ident["kind"], cat, self.db.categories())
            if hint and not overrides.get("category_ok"):
                kind_word = "сериал" if ident["kind"] == "tv" else "фильм"
                for_word = "фильмов" if cat["kind"] == "movies" else "сериалов"
                ident["problems"].insert(0, f"{CATEGORY_PROBLEM}: похоже, это {kind_word}, а категория "
                                            f"«{cat['name']}» — для {for_word}. Подходит «{hint}»?")
                ident["category_hint"] = hint
            plan = planner.build(ident, sc, dest_root(s, cat), s, probe=safe_probe)
        except Exception as e:
            self.db.update_item(item_id, state=ERROR, error=str(e))
            self.db.log(f"Не удалось разобрать: {e}", "error", item_id)
            log.debug(traceback.format_exc())
            return
        problems = ident["problems"] + plan["problems"]
        conf = ident["confidence"]
        manual = bool(overrides)
        if not plan["actions"]:
            state, why = REVIEW, "нечего делать"
        elif problems and not manual:
            state, why = REVIEW, "есть вопросы"
        elif manual:
            state, why = REVIEW, "исправлено вручную, нужно подтвердить"
        elif cat["approval"] == "auto":
            state, why = APPROVED, "режим «автоматически»"
        elif cat["approval"] == "auto_if_confident" and conf >= s["confidence"]:
            state, why = APPROVED, f"уверенность {conf:.0%}"
        elif cat["approval"] == "auto_if_confident":
            state, why = REVIEW, f"уверенность {conf:.0%} < {s['confidence']:.0%}"
        else:
            state, why = REVIEW, "категория требует подтверждения"
        self.db.update_item(item_id, state=state, ident=ident, plan=plan, error=None, note=None)
        self.db.trace(item_id, identified_at=time.time(), decision=why,
                      approved_by="auto" if state == APPROVED else None,
                      approved_at=time.time() if state == APPROVED else None)
        self.db.log(f"Распознано ({why}): {_summary(ident)}; {_plan_summary(plan)} → "
                    f"{'одобрено' if state == APPROVED else 'на проверку'}", item_id=item_id)
        tr = self.db.item(item_id).get("trace") or {}
        # the model either proposes an answer (rules failed) or double-checks the rules' answer
        model_can_help = [p for p in problems if not p.startswith(CATEGORY_PROBLEM)] or conf < s["confidence"]
        if (state == REVIEW and not manual and plan["actions"] and s.get("llm_auto") and llm.configured(s)
                and not tr.get("llm_tried") and model_can_help
                and not (cat["approval"] == "always_confirm" and not problems)):
            self.ask_llm(item_id, auto=True)

    def approve(self, item_id: int, run_now: bool = False) -> None:
        item = self.db.item(item_id)
        if not item or not item.get("plan") or not item["plan"]["actions"]:
            return
        if run_now:
            self.db.update_item(item_id, plan={**item["plan"], "run_now": True})
        if item["state"] in (REVIEW, ERROR):
            self.db.update_item(item_id, state=APPROVED, error=None)
            self.db.trace(item_id, approved_by="manual", approved_at=time.time())
            self.db.log("Одобрено вручную" + (", выполнить не дожидаясь ночи" if run_now else ""), item_id=item_id)
        elif run_now:
            self.db.log("Выполнить не дожидаясь ночи", item_id=item_id)
        self.trigger()

    def ask_llm(self, item_id: int, auto: bool = False, wait: bool = False) -> None:
        """Put the item into the model queue. One request runs at a time; the answer waits for confirmation."""
        s = config.load(self.db)
        with self._llm_cv:
            if item_id in self.llm_jobs:
                conv = self.llm_jobs[item_id]
            else:
                conv = llm.Conversation(model=s.get("llm_model") or "", prompt="")
                self.llm_jobs[item_id] = conv
                self._llm_queue.append((item_id, auto))
                self.db.execute("UPDATE items SET note=? WHERE id=?", (LLM_QUEUED, item_id))
                if self._llm_thread is None or not self._llm_thread.is_alive():
                    self._llm_thread = threading.Thread(target=self._llm_loop, daemon=True, name="llm")
                    self._llm_thread.start()
                self._llm_cv.notify()
        if wait:
            conv.done.wait()

    def llm_queue_position(self, item_id: int) -> int | None:
        """How many requests are ahead (0 = next), None if not queued."""
        with self._llm_lock:
            ids = [i for i, _ in self._llm_queue]
            if item_id not in ids:
                return None
            running = any(c.status == "running" for c in self.llm_jobs.values())
            return ids.index(item_id) + (1 if running else 0)

    def _llm_loop(self) -> None:
        while True:
            with self._llm_cv:
                while not self._llm_queue:
                    self._llm_cv.wait()
                item_id, auto = self._llm_queue.pop(0)
                conv = self.llm_jobs.get(item_id)
            if conv is None:
                continue
            conv.status, conv.started = "running", time.time()
            self.db.execute("UPDATE items SET note=? WHERE id=?", (LLM_THINKING, item_id))
            try:
                self._ask_llm(item_id, auto)
            except Exception as e:  # noqa: BLE001 — the queue must keep going
                self.db.log(f"Модель: {e}", "error", item_id)
                with self._llm_lock:
                    self.llm_jobs.pop(item_id, None)
            finally:
                conv.done.set()

    def _frames_for(self, item_id: int, ident: dict, sc: media.Scan, s: dict, conv: llm.Conversation) -> list[bytes]:
        """Frames of the main video for a film (or a show nobody could name), when the model can see images."""
        named_show = ident.get("kind") == "tv" and ((ident.get("show") or {}).get("chosen") or {}).get("id")
        if named_show or not sc.videos or not llm.supports_vision(s):
            return []
        main = max(sc.videos, key=lambda v: sc.sizes.get(v, 0))
        info = self.probe_cached(sc.base / main) or {}
        frames = media.extract_frames(sc.base / main, info.get("duration") or 0, FRAMES_DIR / str(item_id))
        conv.images = [f.name for f in frames]
        conv.prompt += llm.FRAMES_NOTE + f"\n({len(frames)} frames of «{PurePath(main).name}» attached.)"
        return [f.read_bytes() for f in frames]

    def _movie_by_frames(self, item_id: int, item: dict, sc: media.Scan, s: dict, tmdb: TMDB | None,
                         conv: llm.Conversation) -> tuple[dict, str]:
        """Two steps, so the model isn't anchored on candidates found from a useless file name:
        A) it reads the frames (title card, credits) without any list;
        B) what it read is looked up in TMDB, and it picks among the films by their cards (director, cast,
           studios, runtime). The result waits for confirmation like any model answer.
        """
        main = max(sc.videos, key=lambda v: sc.sizes.get(v, 0))
        info = self.probe_cached(sc.base / main) or {}
        duration = info.get("duration") or 0
        frames = media.extract_frames(sc.base / main, duration, FRAMES_DIR / str(item_id))
        if not frames:
            raise llm.LLMError("не удалось вырезать кадры из файла")
        conv.images = [f.name for f in frames]
        conv.prompt = (f"File «{PurePath(main).name}», {int(duration // 60)} min. "
                       f"{len(frames)} frames attached: opening, middle, end credits.")
        seen = llm.ask(s, conv.prompt, conv, images=[f.read_bytes() for f in frames],
                       schema=llm.VISION_SCHEMA, system=llm.VISION_SYSTEM)
        self._save_conversation(item_id, conv)
        screen = [n for n in dict.fromkeys([seen.get("title"), seen.get("original_title")]) if n and n.strip()]
        guess = (seen.get("title_from_name") or "").strip()
        names = screen + ([guess] if guess and guess not in screen else [])
        # a year in the file name is a fact; the model's year may be bent to fit its guess
        year = parse.guess_title(PurePath(main).name)[1] or (seen.get("year") if isinstance(seen.get("year"), int) else None)
        read = f"на экране: {seen.get('comment') or ', '.join(names)}"
        found: dict[int, Any] = {}
        if tmdb is not None:
            for n in names:
                for c in tmdb.search("movie", n, year) + tmdb.search("movie", n):
                    found.setdefault(c.id, c)
            if year:   # films of the studios read in the credits, that year
                for c in _by_studios(tmdb, seen.get("credits") or [], year):
                    found.setdefault(c.id, c)
        if year:       # a year read on screen rules out candidates from other years (models invent titles)
            found = {k: c for k, c in found.items() if not c.year or abs(c.year - year) <= 1}
        skel = parse.disemvoweled(PurePath(main).name)
        if skel:       # "LF_R_SMTHNG_LK_T": no model needed if exactly one candidate has these consonants
            same = [c for c in found.values()
                    if parse.skeleton(skel) in (parse.skeleton(c.original_title), parse.skeleton(c.title))]
            if len(same) == 1:
                return ({"kind": "movie", "hints": [same[0].original_title],
                         "movies": {"0": {"tmdb_id": same[0].id}}},
                        f"решено без модели: имя файла без гласных совпадает с «{same[0].original_title}» "
                        f"({same[0].year}), а в титрах — студии этого фильма. Модель видела: "
                        f"{seen.get('comment') or '—'}")
        if not names and not found:
            raise llm.LLMError("модель не смогла прочитать название на кадрах: " + (seen.get("comment") or ""))
        studio_hits = not screen or not any(identify.score(screen, year, c) >= 0.6 for c in found.values())
        ranked = sorted(found.values(), key=lambda c: identify.score(names, year, c), reverse=True)
        ranked = ranked[:6] if studio_hits else [c for c in ranked if identify.score(names, year, c) >= 0.6][:6]
        overrides = {"kind": "movie", "hints": names}
        manual = {"title": screen[0], "year": year} if screen else {}   # never a decoded guess as a title
        if not ranked:
            overrides["movies"] = {"0": manual}
            return overrides, f"{read}; в TMDB такого не нашлось"
        best = identify.score(names, year, ranked[0])
        second = identify.score(names, year, ranked[1]) if len(ranked) > 1 else 0
        if not studio_hits and (len(ranked) == 1 or (year and best >= 0.95 and second < best - 0.1)):
            overrides["movies"] = {"0": {"tmdb_id": ranked[0].id}}
            return overrides, f"{read}; в TMDB это «{ranked[0].title}» ({ranked[0].year})"
        cards = [tmdb.movie_card(c.id) for c in ranked]
        step2 = llm.Conversation(model=s["llm_model"], prompt=llm.build_choose_prompt(
            PurePath(main).name, duration, seen, cards), status="running")
        with self._llm_lock:
            self.llm_jobs[item_id] = step2
        pick = llm.ask(s, step2.prompt, step2, schema=llm.CHOOSE_SCHEMA, system=llm.CHOOSE_SYSTEM)
        tid = pick.get("tmdb_id")
        overrides["movies"] = {"0": {"tmdb_id": tid} if tid in {c["id"] for c in cards} else manual}
        return overrides, f"{read}; {pick.get('comment') or ''}".strip("; ")

    def _show_by_frames(self, item_id: int, item: dict, sc: media.Scan, s: dict, tmdb: TMDB | None,
                        conv: llm.Conversation) -> tuple[dict, str]:
        """Same two steps for a series nobody could name ("Season 01/S01E01 - The Pants Tent.mkv"):
        A) read the opening of the first episode, B) pick among TMDB shows by their cards."""
        files = sorted(sc.videos)
        first = files[0]
        info = self.probe_cached(sc.base / first) or {}
        duration = info.get("duration") or 0
        frames = media.extract_frames(sc.base / first, duration, FRAMES_DIR / str(item_id))
        if not frames:
            raise llm.LLMError("не удалось вырезать кадры из файла")
        names_list = [PurePath(f).name for f in files]
        conv.images = [f.name for f in frames]
        conv.prompt = (f"One episode of a TV series: «{PurePath(first).name}», {int(duration // 60)} min. "
                       f"All episode files: {'; '.join(names_list[:15])}. Folder: «{sc.name}». "
                       f"{len(frames)} frames attached: opening, middle, end credits.")
        seen = llm.ask(s, conv.prompt, conv, images=[f.read_bytes() for f in frames],
                       schema=llm.VISION_SCHEMA, system=llm.VISION_SYSTEM)
        self._save_conversation(item_id, conv)
        episode_titles = {parse._norm(re.sub(r"(?i)^s\d+e\d+\W*", "", PurePath(f).stem)) for f in files}
        names = [n for n in dict.fromkeys([seen.get("title"), seen.get("original_title"),
                                            seen.get("title_from_name")])
                 if n and n.strip() and parse._norm(n) not in episode_titles]   # an episode title isn't the show
        year = seen.get("year") if isinstance(seen.get("year"), int) else None
        read = f"на экране: {seen.get('comment') or ', '.join(names)}"
        if not names:
            raise llm.LLMError("модель не узнала сериал (назвала только серию). Укажите сериал вручную")
        found: dict[int, Any] = {}
        if tmdb is not None:
            for n in names:
                for c in tmdb.search("tv", n, year) + tmdb.search("tv", n):
                    found.setdefault(c.id, c)
        ranked = sorted(found.values(), key=lambda c: identify.score(names, year, c), reverse=True)
        ranked = [c for c in ranked if identify.score(names, year, c) >= 0.6][:6]
        overrides = {"kind": "tv", "hints": names}
        if not ranked:
            raise llm.LLMError(f"модель назвала «{names[0]}», но в TMDB такого сериала нет — похоже, выдумала. "
                               f"Укажите сериал вручную")
        if len(ranked) == 1 and identify.score(names, year, ranked[0]) >= 0.9:
            overrides["show"] = {"tmdb_id": ranked[0].id}
            return overrides, f"{read}; в TMDB это «{ranked[0].title}» ({ranked[0].year})"
        cards = [tmdb.tv_card(c.id) for c in ranked]
        step2 = llm.Conversation(model=s["llm_model"], prompt=llm.build_choose_prompt(
            PurePath(first).name, duration, seen, cards, names_list), status="running")
        with self._llm_lock:
            self.llm_jobs[item_id] = step2
        pick = llm.ask(s, step2.prompt, step2, schema=llm.CHOOSE_SCHEMA, system=llm.CHOOSE_SYSTEM)
        tid = pick.get("tmdb_id")
        if tid not in {c["id"] for c in cards}:
            raise llm.LLMError("модель не уверена, какой это сериал. Укажите сериал вручную: " + (pick.get("comment") or ""))
        overrides["show"] = {"tmdb_id": tid}
        return overrides, f"{read}; {pick.get('comment') or ''}".strip("; ")

    @staticmethod
    def _resolve_movie_title(overrides: dict, answer: dict, tmdb: TMDB | None) -> None:
        """The model read a title (e.g. from the credits) that wasn't among the candidates: look it up in TMDB.

        One clear match → that TMDB id. Several → keep the title and offer them as candidates to choose from.
        """
        choice = (overrides.get("movies") or {}).get("0") or {}
        if tmdb is None or "tmdb_id" in choice or not answer.get("title"):
            return
        names = [n for n in dict.fromkeys([answer.get("title"), answer.get("original_title")]) if n]
        overrides["hints"] = names
        year = answer.get("year") if isinstance(answer.get("year"), int) else None
        found: dict[int, Any] = {}
        for n in names:
            try:
                for c in tmdb.search("movie", n, year) + tmdb.search("movie", n):
                    found.setdefault(c.id, c)
            except ClientError:
                return
        ranked = sorted(found.values(), key=lambda c: identify.score(names, year, c), reverse=True)
        overrides.setdefault("movies", {})
        if ranked:
            best = identify.score(names, year, ranked[0])
            second = identify.score(names, year, ranked[1]) if len(ranked) > 1 else 0
            if best >= 0.9 and second < best - 0.1:
                overrides["movies"]["0"] = {"tmdb_id": ranked[0].id}

    def cancel_llm(self, item_id: int) -> None:
        with self._llm_lock:
            conv = self.llm_jobs.get(item_id)
            queued = [q for q in self._llm_queue if q[0] == item_id]
            if conv and queued:                       # not started yet: just take it out of the queue
                self._llm_queue.remove(queued[0])
                self.llm_jobs.pop(item_id, None)
        if conv and queued:
            conv.status = "cancelled"
            conv.done.set()
            self.db.execute("UPDATE items SET note=NULL WHERE id=?", (item_id,))
            self.db.log("Убрано из очереди к модели", item_id=item_id)
        elif conv:
            conv.cancel.set()
        elif (self.db.item(item_id) or {}).get("note") in (LLM_THINKING, LLM_QUEUED):   # stale label
            self.db.execute("UPDATE items SET note=NULL WHERE id=?", (item_id,))

    def _save_conversation(self, item_id: int, conv: llm.Conversation) -> None:
        old = (self.db.item(item_id).get("trace") or {}).get("llm_log") or []
        self.db.trace(item_id, llm_log=([conv.as_dict()] + old)[:5])

    def _ask_llm(self, item_id: int, auto: bool) -> None:
        conv = self.llm_jobs[item_id]
        try:
            s = config.load(self.db)
            item = self.db.item(item_id)
            if not llm.configured(s):
                raise llm.LLMError("модель не настроена (Настройки → Локальная модель)")
            ident = item.get("ident") or {}
            sc = media.scan(item["content_path"])
            if ident.get("kind") == "movie":
                cands = (ident.get("movies") or [{}])[0].get("candidates") or []
            else:
                cands = (ident.get("show") or {}).get("candidates") or []
            tmdb = tmdb_client(s)
            seasons = {e["season"] for e in ident.get("episodes") or []}
            titles = llm.episode_titles_for(tmdb, cands, seasons) if ident.get("kind") != "movie" else {}
            conv.prompt = llm.build_prompt(item["name"], sc.videos, ident.get("kind", ""), cands, titles)
            conv.model = s["llm_model"]
            started = time.time()
            if ident.get("kind") == "movie" and len(ident.get("movies") or []) == 1 and llm.supports_vision(s):
                overrides, comment = self._movie_by_frames(item_id, item, sc, s, tmdb, conv)
                conv = self.llm_jobs.get(item_id, conv)
            elif ident.get("kind") == "tv" and _show_is_weak(ident) and llm.supports_vision(s):
                overrides, comment = self._show_by_frames(item_id, item, sc, s, tmdb, conv)
                conv = self.llm_jobs.get(item_id, conv)
            else:
                images = self._frames_for(item_id, ident, sc, s, conv)
                answer = llm.ask(s, conv.prompt, conv, images=images)
                overrides, comment = llm.to_overrides(answer, sc.videos, {c["id"] for c in cands})
                if overrides.get("kind") == "movie":
                    self._resolve_movie_title(overrides, answer, tmdb)
        except (llm.LLMError, OSError) as e:
            offline = "недоступна" in str(e)
            conv = self.llm_jobs.get(item_id, conv)
            if conv.status == "running":
                conv.status, conv.error, conv.finished = "error", str(e), time.time()
            self._save_conversation(item_id, conv)
            self.db.execute("UPDATE items SET note=? WHERE id=?", (f"модель: {e}", item_id))
            self.db.trace(item_id, llm_tried=True, llm_pending=bool(auto and offline))
            self.db.log(f"Модель: {e}", "warning", item_id)
            return
        finally:
            with self._llm_lock:
                self.llm_jobs.pop(item_id, None)
        self._save_conversation(item_id, conv)
        self.db.trace(item_id, llm_tried=True, llm_pending=False, llm_comment=comment,
                      llm_seconds=round(time.time() - started))
        took = round(time.time() - started)
        agree, diff = compare_with_rules(ident, overrides)
        plan = item.get("plan") or {}
        rules_sure = bool(plan.get("actions")) and all(
            p.startswith(INFO_PROBLEMS) for p in (ident.get("problems") or []) + (plan.get("problems") or []))
        if agree and rules_sure:
            cat = self.db.categories().get(item["category"]) or {}
            if s.get("llm_confirm_auto", True) and cat.get("approval") != "always_confirm" \
                    and self.db.item(item_id)["state"] == REVIEW:
                self.db.update_item(item_id, state=APPROVED, error=None,
                                    note=f"модель согласна с TMDB — продолжено без подтверждения: {comment}")
                self.db.trace(item_id, approved_by="model", approved_at=time.time(),
                              decision="модель подтвердила результат правил")
                self.db.log(f"Модель подтвердила результат правил ({took} с) → одобрено: {comment}", item_id=item_id)
                self.trigger()
            else:
                self.db.execute("UPDATE items SET note=? WHERE id=?",
                                (f"модель согласна с TMDB — подтвердите: {comment}", item_id))
                self.db.log(f"Модель подтвердила результат правил ({took} с): {comment}", item_id=item_id)
            return
        if diff and rules_sure:
            comment = "ответ модели расходится с TMDB: " + "; ".join(diff[:3]) + (" …" if len(diff) > 3 else "") \
                + (f". Модель: {comment}" if comment else "")
        self.analyze(item_id, overrides)
        self.db.execute("UPDATE items SET note=? WHERE id=?",
                        (f"предложение модели — проверьте и подтвердите: {comment}", item_id))
        self.db.log(f"Модель предложила ({round(time.time() - started)} с): {comment}", item_id=item_id)

    def skip(self, item_id: int) -> None:
        self.db.update_item(item_id, state=SKIPPED)
        self.db.log("Пропущено", item_id=item_id)

    def run_item(self, item_id: int) -> bool:
        """Execute an item that was claimed (state RUNNING)."""
        item = self.db.item(item_id)
        s = config.load(self.db)
        plan = item["plan"]

        def progress(n, total, name, frac, speed):
            overall = ((n - 1) + frac) / max(total, 1)
            prev = self.status.get("job") or {}
            # files finished before a restart are skipped without progress reports, so the first report
            # of this run marks where real work started: the ETA is based on progress since then
            first = prev.get("item_id") != item_id
            self.status["job"] = {"item_id": item_id, "name": item["name"], "file": name, "n": n,
                                  "total": total, "percent": round(frac * 100, 1), "speed": speed,
                                  "overall": overall,
                                  "frac0": overall if first else prev["frac0"],
                                  "since": time.time() if first else prev["since"]}

        self.db.trace(item_id, started_at=time.time())
        self.status["job_started"] = time.time()
        try:
            executor.check_space(plan, s["min_free_gb"], s["link_fallback"])
            results = executor.execute(plan, s["link_fallback"], log=lambda m: self.db.log(m, item_id=item_id),
                                       settings=s, progress=progress)
            problems = executor.verify(results)
            if problems:
                raise RuntimeError("; ".join(problems))
        except executor.NotEnoughSpace as e:
            self.db.update_item(item_id, state=APPROVED, note=f"ждёт места на диске: {e}")
            self.db.log(f"Ждёт места на диске: {e}", "warning", item_id)
            return False
        except Exception as e:
            self.db.update_item(item_id, state=ERROR, error=str(e))
            self.db.log(f"Ошибка выполнения: {e}", "error", item_id)
            log.debug(traceback.format_exc())
            return False
        finally:
            self.status["job"] = None
            self.status["job_started"] = None
        failed = [r for r in results if r["method"] == "failed"]
        note = (f"не обработано файлов: {len(failed)} (оставлены как были): "
                + ", ".join(Path(r["src"]).name for r in failed[:3]) + (" …" if len(failed) > 3 else "")) if failed else None
        self.db.update_item(item_id, state=DONE, result={"files": results}, processed_at=time.time(), note=note)
        self.db.log(f"Готово: {plan['target']}", item_id=item_id)
        if item["source"] == "folder":
            self.finish_folder_item(item_id, results)
        return True

    def finish_folder_item(self, item_id: int, results: list[dict]) -> None:
        """Imported from a folder (no torrent): the move ends by removing the original files."""
        item = self.db.item(item_id)
        if not (item.get("options") or {}).get("delete_source", True):
            return
        produced = {os.path.realpath(r["dst"]) for r in results}
        removed = 0
        for r in results:
            src = r["src"]
            if os.path.realpath(src) in produced or not os.path.exists(src):
                continue  # re-processed in place, or already gone
            os.unlink(src)
            removed += 1
        root = Path(item["content_path"])
        if root.is_dir():
            for d in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)) + [root]:
                try:
                    d.rmdir()  # only empty folders
                except OSError:
                    pass
        if removed == 0:
            self.db.log("Обработано на месте", item_id=item_id)
            return
        self.db.update_item(item_id, state=REMOVED, removed_at=time.time())
        self.db.trace(item_id, removed_reason=f"перенос из папки: удалено исходных файлов {removed}")
        self.db.log(f"Перенос завершён: удалено исходных файлов {removed}", item_id=item_id)

    def cleanup(self, s: dict, qb: QBit, torrents: dict[str, dict]) -> None:
        cats = self.db.categories()
        for item in self.db.items(DONE):
            if not item["hash"]:
                continue
            t = torrents.get(item["hash"])
            if t is None:
                self.db.update_item(item["id"], state=REMOVED, removed_at=time.time())
                self.db.trace(item["id"], removed_reason="торрент удалён вне jellyprep")
                self.db.log("Торрент удалён вне сервиса", item_id=item["id"])
                continue
            due, reason = deletion_due(cats.get(item["category"]), t.get("ratio", 0), t.get("seeding_time", 0))
            if not due:
                continue
            files = (item.get("result") or {}).get("files", [])
            problems = executor.verify(files)
            content = item["content_path"].rstrip("/")
            if not files:
                problems.append("нет записей о файлах в библиотеке")
            if any(f["dst"] == content or f["dst"].startswith(content + "/") for f in files):
                problems.append("файлы библиотеки лежат внутри папки торрента")
            if problems:
                self.db.execute("UPDATE items SET note=? WHERE id=?",
                                ("удаление отложено: " + "; ".join(problems), item["id"]))
                self.db.log(f"Не удаляю торрент, библиотека не в порядке: {'; '.join(problems)}", "warning", item["id"])
                continue
            if s["dry_run"]:
                continue
            qb.delete(item["hash"], delete_files=True)
            self.db.update_item(item["id"], state=REMOVED, removed_at=time.time())
            self.db.trace(item["id"], removed_reason=f"условие раздачи выполнено: {reason}")
            self.db.log(f"Торрент удалён вместе с файлами ({reason})", item_id=item["id"])

    def refresh_jellyfin(self, s: dict) -> None:
        if not (s.get("jellyfin_url") and s.get("jellyfin_key")):
            return
        try:
            Jellyfin(s["jellyfin_url"], s["jellyfin_key"]).refresh()
            self.db.log("Jellyfin: запущено обновление библиотеки")
        except ClientError as e:
            self.db.log(str(e), "warning")

    # --- import from existing folders ---------------------------------------
    def import_folder(self, folder: str, category: str, delete_source: bool = True,
                      only: set[str] | None = None, in_place: bool = False) -> dict:
        """Queue every movie/show in `folder` that isn't part of a torrent.

        Each top-level entry (a video file or a sub-folder with videos) becomes one item. Entries that
        belong to a torrent are skipped: the torrent side handles them and removes them after seeding.
        """
        s = config.load(self.db)
        root = Path(folder)
        if not root.is_dir():
            raise ValueError(f"Нет такой папки: {folder}")
        if category not in self.db.categories():
            raise ValueError(f"Нет категории {category}")
        torrent_paths = [Path(map_path(t["content_path"], s["path_map"])) for t in qbit_client(s).torrents()]
        torrent_inodes = _inodes(torrent_paths)
        # in the new library, files finished earlier may need work again (e.g. the voice track was off)
        known = self.db.active_paths() if in_place else self.db.known_paths()
        res = {"added": [], "torrent": [], "known": [], "empty": []}
        for entry in sorted(root.iterdir()):
            if entry.name.startswith(".") or (only is not None and entry.name not in only):
                continue
            if str(entry) in known:
                res["known"].append(entry.name)
                continue
            if any(_related(entry, tp) for tp in torrent_paths) or _inodes([entry]) & torrent_inodes:
                res["torrent"].append(entry.name)   # seeding, possibly through a hardlink (Sonarr/Radarr layouts)
                continue
            try:
                if not media.scan(entry).videos:
                    res["empty"].append(entry.name)
                    continue
            except OSError:
                res["empty"].append(entry.name)
                continue
            item_id = self.db.add_item(source="folder", name=entry.name, category=category,
                                       content_path=str(entry), options={"delete_source": delete_source})
            self.db.log(f"Импорт из папки {folder}", item_id=item_id)
            res["added"].append(entry.name)
        self.trigger()
        return res


def _inodes(paths: list[Path]) -> set[tuple[int, int]]:
    """(device, inode) of every file under the paths: hardlinks share them."""
    out = set()
    for p in paths:
        files = [p] if p.is_file() else (p.rglob("*") if p.is_dir() else [])
        for f in files:
            try:
                st = f.stat()
            except OSError:
                continue
            if f.is_file() and st.st_nlink > 1:
                out.add((st.st_dev, st.st_ino))
    return out


def _related(entry: Path, torrent_path: Path) -> bool:
    """True when the entry is the torrent's content, lies inside it, or contains it."""
    e, t = os.path.normpath(entry), os.path.normpath(torrent_path)
    return e == t or e.startswith(t + os.sep) or t.startswith(e + os.sep)


def _summary(ident: dict) -> str:
    if ident["kind"] == "tv":
        c = ident["show"].get("chosen") or {}
        n = len(ident["episodes"])
        return f"сериал «{c.get('title', '?')}» ({c.get('year') or '?'}), {n} серий"
    titles = [f"«{(m.get('chosen') or {}).get('title', '?')}»" for m in ident.get("movies", [])]
    return "фильм " + ", ".join(titles[:3]) + (" …" if len(titles) > 3 else "")


ACTION_LABELS = {"hardlink": "хардлинк", "transcode": "перекодирование", "process": "обработка звука",
                 "remux": "перепаковка", "exists": "уже в библиотеке", "copy": "копия",
                 "failed": "не обработан"}


def _plan_summary(plan: dict) -> str:
    counts: dict[str, int] = {}
    for a in plan["actions"]:
        if a["type"] == "video":
            counts[a["action"]] = counts.get(a["action"], 0) + 1
    return ", ".join(f"{ACTION_LABELS.get(k, k)}: {v}" for k, v in counts.items()) or "нет видео"
