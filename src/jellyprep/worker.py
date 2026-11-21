"""Background work.

Two threads:
- the main loop syncs torrents, identifies them, does the light work (hardlinks) and cleans up after seeding;
- the encoder takes heavy jobs (re-encode, audio processing, remux) one at a time. Re-encoding waits for
  the night window unless the user pressed "run now".
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time
import traceback
from pathlib import Path

from . import config, executor, identify, media, planner, transcode
from .clients import QBit, TMDB, ClientError, Jellyfin, is_complete
from .db import APPROVED, DB, DONE, ERROR, NEW, REMOVED, REVIEW, RUNNING, SKIPPED

log = logging.getLogger("jellyprep")

KIND_HINT = {"movies": "movie", "tv": "tv", "auto": "auto"}


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


class Worker:
    def __init__(self, db: DB):
        self.db = db
        self._wake = threading.Event()
        self._enc_wake = threading.Event()
        self._lock = threading.Lock()
        self.status: dict = {"last_run": None, "last_error": None, "running": False, "job": None,
                             "job_started": None}
        self._downloads: tuple[float, list[dict]] = (0.0, [])

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
            ident = identify.identify(sc, KIND_HINT.get(cat["kind"], "auto"), tmdb_client(s), overrides)
            ident["overrides"] = overrides
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
    def import_folder(self, folder: str, category: str, delete_source: bool = True) -> dict:
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
        known = self.db.known_paths()
        res = {"added": [], "torrent": [], "known": [], "empty": []}
        for entry in sorted(root.iterdir()):
            if entry.name.startswith("."):
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
