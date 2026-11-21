"""Web UI and API."""
from __future__ import annotations

import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from . import board, config, pipeline, survey, transcode, worker as worker_mod
from .clients import ClientError, Jellyfin, TMDB, llm_models
from .db import APPROVED, DB, DONE, ERROR, NEW, REMOVED, REVIEW, RUNNING, SKIPPED
from .media import free_bytes

HERE = Path(__file__).parent
STEPS = [
    ("qbit", "qBittorrent"),
    ("paths", "Папки"),
    ("categories", "Категории"),
    ("tmdb", "TMDB"),
    ("llm", "Локальная модель"),
    ("video", "Видео"),
    ("audio", "Звук и языки"),
    ("cleanup", "Удаление"),
    ("limits", "Ограничения"),
    ("jellyfin", "Jellyfin"),
    ("finish", "Готово"),
]
STEP_IDS = [s for s, _ in STEPS]
STATE_LABELS = {NEW: "новый", REVIEW: "на проверке", APPROVED: "в очереди", RUNNING: "выполняется",
                DONE: "в библиотеке",
                REMOVED: "оригинал удалён", SKIPPED: "пропущен", ERROR: "ошибка"}
OFF_BY_DEFAULT = re.compile(r"(?i)radarr|sonarr|lidarr|readarr|music|audio|book")
CLEANUP_PRESETS = {
    "r2d30": (True, 2.0, 30.0),
    "r1d14": (True, 1.0, 14.0),
    "never": (False, None, None),
}


def create_app(db: DB | None = None, start_worker: bool = True) -> FastAPI:
    db = db or DB(config.CONFIG_DIR / "jellyprep.db")
    worker = worker_mod.Worker(db)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_worker:
            worker.start()
        yield

    app = FastAPI(title="jellyprep", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.db, app.state.worker = db, worker
    worker.surveys = survey.Surveys(worker)
    app.state.flash = {}
    app.add_middleware(SameOriginMiddleware)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    tpl = Jinja2Templates(directory=HERE / "templates")
    tpl.env.filters.update(size=_size, ts=_ts, days=lambda s: f"{(s or 0) / 86400:.0f}",
                           pretty_json=_pretty_json, mmss=_mmss)
    tpl.env.globals.update(STATE_LABELS=STATE_LABELS, STEPS=STEPS, ACTION_LABELS=worker_mod.ACTION_LABELS, VOICE_TITLE=transcode.VOICE_TITLE,
                           is_heavy=worker_mod.is_heavy, needs_window=worker_mod.needs_window)

    def render(request: Request, name: str, **ctx) -> HTMLResponse:
        s = config.load(db)
        counts = {r["state"]: r["n"] for r in db.query("SELECT state, count(*) n FROM items GROUP BY state")}
        return tpl.TemplateResponse(request, name, {"s": config.public(s), "status": worker.status,
                                                    "counts": counts, **ctx})

    def back(url: str) -> RedirectResponse:
        return RedirectResponse(url, status_code=303)

    # --- pages ------------------------------------------------------------
    @app.get("/")
    def index():
        return back("/items" if config.load(db)["setup_done"] else "/setup/qbit")

    def board_data() -> dict:
        s = config.load(db)
        return board.build(db.items(), worker.downloads(), s, db.categories(), lambda i: db.logs(i, limit=50),
                           _queue_positions(db), worker.status.get("job"), worker.status.get("job_started"),
                           worker.llm_jobs, {i: worker.llm_queue_position(i) for i in list(worker.llm_jobs)})

    @app.get("/items", response_class=HTMLResponse)
    def items(request: Request):
        return render(request, "board.html", b=board_data())

    @app.get("/api/board")
    def api_board():
        return board_data()

    @app.get("/items/{item_id}/panel", response_class=HTMLResponse)
    def item_panel(request: Request, item_id: int):
        ctx = _item_context(item_id)
        if ctx is None:
            return HTMLResponse("<p class='muted'>Задача не найдена</p>", status_code=404)
        return render(request, "_item_panel.html", **ctx)

    @app.get("/items/{item_id}", response_class=HTMLResponse)
    def item_page(request: Request, item_id: int):
        ctx = _item_context(item_id)
        if ctx is None:
            return back("/items")
        return render(request, "item.html", **ctx)

    def _item_context(item_id: int) -> dict | None:
        item = db.item(item_id)
        if not item:
            return None
        cat = db.categories().get(item["category"])
        due = worker_mod.deletion_due(cat, item["ratio"] or 0, item["seeding_time"] or 0)
        logs = db.logs(item_id)
        pipe = pipeline.build(item, config.load(db), cat, logs, _queue_positions(db).get(item_id, 0),
                              worker.status.get("job"))
        live = worker.llm_jobs.get(item_id)
        convs = ([{**live.as_dict(), "live": True}] if live else []) + ((item.get("trace") or {}).get("llm_log") or [])
        cats = db.categories()
        return {"item": item, "cat": cat, "due": due, "logs": logs, "pipe": pipe, "convs": convs, "now": time.time(),
                "summary": pipeline.confirm_summary(item, config.load(db), cat), "all_cats": list(cats.values()),
                "cat_hint": (item.get("ident") or {}).get("category_hint")}

    def back_item(request: Request, item_id: int) -> RedirectResponse:
        """After an action, return to the board (with the row open) if that's where it came from."""
        ref = urlparse(request.headers.get("referer") or "")
        if ref.path in ("/items", "/"):
            return back(f"/items#item-{item_id}")
        return back(f"/items/{item_id}")

    @app.post("/items/{item_id}/approve")
    def item_approve(request: Request, item_id: int):
        worker.approve(item_id)
        return back_item(request, item_id)

    @app.post("/items/{item_id}/run-now")
    def item_run_now(request: Request, item_id: int):
        worker.approve(item_id, run_now=True)
        return back_item(request, item_id)

    @app.post("/items/{item_id}/ask-llm")
    def item_ask_llm(request: Request, item_id: int):
        worker.ask_llm(item_id)
        return back_item(request, item_id)

    @app.get("/items/{item_id}/frames/{name}")
    def item_frame(item_id: int, name: str):
        if not re.fullmatch(r"f_\d{6}\.jpg", name):
            return JSONResponse({"error": "bad name"}, status_code=400)
        path = worker_mod.FRAMES_DIR / str(item_id) / name
        if not path.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/items/{item_id}/category")
    async def item_category(request: Request, item_id: int):
        form = await request.form()
        item = db.item(item_id)
        if item:
            ov = (item.get("ident") or {}).get("overrides") or {}
            if form.get("keep") == "1":
                worker.analyze(item_id, {**ov, "category_ok": True})
                db.log(f"Категория «{item['category']}» оставлена", item_id=item_id)
            else:
                worker_mod.set_category(db, item_id, form.get("category", ""))
                ov = {k: v for k, v in ov.items() if k not in ("kind", "category_ok")}
                worker.analyze(item_id, ov)
        return back_item(request, item_id)

    @app.post("/items/{item_id}/llm-cancel")
    def item_llm_cancel(request: Request, item_id: int):
        worker.cancel_llm(item_id)
        return back_item(request, item_id)

    @app.post("/items/{item_id}/skip")
    def item_skip(request: Request, item_id: int):
        worker.skip(item_id)
        return back_item(request, item_id)

    @app.post("/items/{item_id}/reanalyze")
    def item_reanalyze(request: Request, item_id: int):
        worker.analyze(item_id, overrides={})
        return back_item(request, item_id)

    @app.post("/items/{item_id}/override")
    async def item_override(request: Request, item_id: int):
        form = await request.form()
        item = db.item(item_id)
        if not item or not item.get("ident"):
            return back(f"/items/{item_id}")
        worker.analyze(item_id, overrides=_overrides_from_form(form, item["ident"]))
        return back_item(request, item_id)

    @app.get("/categories", response_class=HTMLResponse)
    def categories_page(request: Request):
        return render(request, "categories.html", cats=_category_rows(db), qbit_error=None)

    @app.post("/categories")
    async def categories_save(request: Request):
        _save_categories(db, await request.form(), with_cleanup=True)
        return back("/categories")

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request):
        s = config.load(db)
        return render(request, "settings.html", path_map_text=_path_map_text(s["path_map"]),
                      jf_path_map_text=_path_map_text(s["jellyfin_path_map"]))

    @app.post("/settings")
    async def settings_save(request: Request):
        form = await request.form()
        values = {k: v for k, v in form.items() if k in config.DEFAULTS}
        values = _coerce(values)
        if "path_map_text" in form:
            values["path_map"] = _parse_path_map(form["path_map_text"])
        if "jf_path_map_text" in form:
            values["jellyfin_path_map"] = _parse_path_map(form["jf_path_map_text"])
        for key in ("keep_langs", "voice_langs"):
            if f"{key}_text" in form:
                values[key] = _lang_list(form[f"{key}_text"])
        if "remux_containers_text" in form:
            values["remux_containers"] = [x.strip().lower().lstrip(".") for x in
                                          re.split(r"[,\s]+", form["remux_containers_text"]) if x.strip()]
        values["dry_run"] = form.get("dry_run") == "1"
        config.save(db, values)
        transcode._vaapi_works.cache_clear()
        worker.trigger()
        return back("/settings")

    @app.post("/settings/rerun-wizard")
    def rerun_wizard():
        return back("/setup/qbit")

    @app.get("/import", response_class=HTMLResponse)
    def import_page(request: Request, refresh: int = 0):
        s = config.load(db)
        cats = db.categories()
        libs, jf_error = [], None
        if s.get("jellyfin_url") and s.get("jellyfin_key"):
            try:
                libs = Jellyfin(s["jellyfin_url"], s["jellyfin_key"]).libraries()
            except Exception as e:  # noqa: BLE001
                jf_error = str(e)
        back_map = [{"from": m["to"], "to": m["from"]} for m in s["jellyfin_path_map"]]
        lib_of = {}  # local path -> Jellyfin library name
        old = []
        root = s["library_root"].rstrip("/")
        for lib in libs:
            for loc in lib.get("Locations") or []:
                local = worker_mod.map_path(loc, back_map).rstrip("/")
                lib_of[local] = lib["Name"]
                if not survey.is_video_library(lib) or local.startswith(root) or not os.path.isdir(local):
                    continue
                old.append({"library": lib["Name"], "path": local, "type": lib.get("CollectionType") or ""})
        cat_list = [c for c in cats.values() if not re.search(r"(?i)music|audio|book", c["name"])]
        for src in old:
            src["suggest"] = _suggest_category(src, cat_list)
            src["sv"] = worker.surveys.get(src["path"], "old", refresh=bool(refresh))
        new = []
        for c in cat_list:
            dest = str(worker_mod.dest_root(s, c))
            if os.path.isdir(dest) and dest not in [n["path"] for n in new]:
                new.append({"category": c["name"], "path": dest, "library": lib_of.get(dest.rstrip("/")),
                            "sv": worker.surveys.get(dest, "new", refresh=bool(refresh))})
        targets = {c["name"]: lib_of.get(str(worker_mod.dest_root(s, c)).rstrip("/")) or c["library_dir"]
                   for c in cat_list}
        running = any(x["sv"]["state"] == "running" for x in old + new)
        return render(request, "import.html", old=old, new=new, cats=cat_list, targets=targets, running=running,
                      jf_error=jf_error, result=app.state.flash.pop("import", None))

    @app.post("/import")
    async def import_run(request: Request):
        form = await request.form()
        path, cat, mode = (form.get("path") or "").strip(), form.get("category", ""), form.get("mode", "manual")
        try:
            if mode in ("old", "new"):
                sv = worker.surveys.get(path, mode)
                rows = (sv.get("result") or {}).get("entries", [])
                want = {r["name"] for r in rows if r["status"] == ("move" if mode == "old" else "work")}
                res = worker.import_folder(path, cat, delete_source=True, only=want, in_place=(mode == "new"))
            else:
                res = worker.import_folder(path, cat, delete_source=form.get("delete_source") == "1")
            worker.surveys.invalidate(path)
            app.state.flash["import"] = {"ok": True, "path": path, "category": cat, "mode": mode, **res}
        except Exception as e:  # noqa: BLE001
            app.state.flash["import"] = {"ok": False, "path": path, "message": str(e)}
        return back("/import")

    @app.get("/log", response_class=HTMLResponse)
    def log_page(request: Request):
        return render(request, "log.html", logs=db.logs())

    @app.post("/run")
    def run_now(request: Request):
        worker.trigger()
        return back(request.headers.get("referer") or "/items")

    @app.get("/api/status")
    def api_status():
        return {"status": worker.status, "job": worker.status.get("job"),
                "counts": {r["state"]: r["n"] for r in db.query("SELECT state, count(*) n FROM items GROUP BY state")}}

    # --- connection tests (use the values currently typed in the form) ---
    @app.post("/api/test/{service}")
    async def api_test(request: Request, service: str):
        form = dict(await request.form())
        s = config.load(db)
        val = lambda k: form.get(k) or s.get(k) or ""  # noqa: E731 — empty secret field = saved one
        try:
            if service == "qbit":
                qb = worker_mod.QBit(val("qbit_url"), form.get("qbit_user", s["qbit_user"]), val("qbit_pass"))
                v = qb.version()
                n = len(qb.torrents())
                return _ok(f"Подключено: qBittorrent {v}, торрентов: {n}")
            if service == "tmdb":
                t = TMDB(val("tmdb_key"), val("tmdb_language") or "ru-RU")
                t.test()
                hit = t.search("movie", "Love Actually", 2003)
                return _ok("Ключ работает" + (f": найдено «{hit[0].title}» ({hit[0].year})" if hit else ""))
            if service == "llm":
                kind = form.get("llm_kind") or s["llm_kind"]
                if kind == "none":
                    return _ok("Модель не используется")
                models = llm_models(kind, val("llm_url"), val("llm_key"))
                return _ok(f"Доступно моделей: {len(models)}", models=models)
            if service == "hw":
                transcode._vaapi_works.cache_clear()
                kind, msg = transcode.hardware({**s, "hw_accel": form.get("hw_accel") or s["hw_accel"]})
                return JSONResponse({"ok": kind == "vaapi" or form.get("hw_accel") == "software", "message": msg})
            if service == "jellyfin":
                j = Jellyfin(val("jellyfin_url"), val("jellyfin_key"))
                name = j.test()
                libs = ", ".join(f'{l["Name"]} ({l.get("CollectionType") or "смешанная"})' for l in j.libraries())
                return _ok(f"Подключено: {name}. Библиотеки: {libs or 'нет'}")
        except ClientError as e:
            return _fail(str(e))
        except Exception as e:  # noqa: BLE001 — show any connection problem to the user
            return _fail(f"{type(e).__name__}: {e}")
        return _fail("Неизвестная проверка")

    # --- setup wizard -----------------------------------------------------
    @app.get("/setup")
    def setup_index():
        return back("/setup/qbit")

    @app.get("/setup/{step}", response_class=HTMLResponse)
    def setup_get(request: Request, step: str):
        if step not in STEP_IDS:
            return back("/setup/qbit")
        s = config.load(db)
        ctx = {"step": step, "idx": STEP_IDS.index(step),
               "prev": STEP_IDS[STEP_IDS.index(step) - 1] if STEP_IDS.index(step) else None}
        if step == "paths":
            ctx.update(_paths_context(s))
            ctx["path_map_text"] = _path_map_text(s["path_map"])
        if step == "categories":
            ctx["cats"], ctx["qbit_error"] = _sync_categories(db, s)
        if step == "cleanup":
            ctx["preset"] = _current_cleanup_preset(db)
        if step == "jellyfin":
            ctx.update(_jellyfin_context(db, s))
            ctx["jf_path_map_text"] = _path_map_text(s["jellyfin_path_map"])
            ctx["jf_results"] = app.state.flash.pop("jellyfin", None)
        if step == "finish":
            ctx["cats"] = [c for c in db.categories().values() if c["enabled"]]
        return render(request, "setup.html", **ctx)

    @app.post("/setup/jellyfin/libraries")
    async def setup_jellyfin_libraries(request: Request):
        form = await request.form()
        app.state.flash["jellyfin"] = _apply_jellyfin_libraries(db, config.load(db), form)
        return back("/setup/jellyfin#libraries")

    @app.post("/setup/{step}")
    async def setup_post(request: Request, step: str):
        form = await request.form()
        s = config.load(db)
        values = _coerce({k: v for k, v in form.items() if k in config.DEFAULTS})
        if step == "paths":
            values["path_map"] = _parse_path_map(form.get("path_map_text", ""))
        if step == "jellyfin":
            values["jellyfin_path_map"] = _parse_path_map(form.get("jf_path_map_text", ""))
        if step == "categories":
            _save_categories(db, form, with_cleanup=False)
        if step == "cleanup":
            _apply_cleanup_preset(db, form)
        if step == "audio":
            values["keep_langs"] = _lang_list(form.get("keep_langs_text", ""))
            values["voice_langs"] = _lang_list(form.get("voice_langs_text", "")) or values["keep_langs"][:1]
        if step == "video":
            transcode._vaapi_works.cache_clear()
        if step == "limits" and form.get("min_free_choice") != "custom":
            values["min_free_gb"] = float(form.get("min_free_choice", s["min_free_gb"]))
        if step == "finish":
            values["dry_run"] = form.get("dry_run", "1") == "1"
            values["setup_done"] = True
        config.save(db, values)
        if step == "finish":
            db.log("Настройка завершена" + (" (dry-run)" if values["dry_run"] else ""))
            worker.trigger()
            return back("/items")
        if form.get("action") == "stay":
            return back(f"/setup/{step}#libraries")
        nxt = STEP_IDS[min(STEP_IDS.index(step) + 1, len(STEP_IDS) - 1)]
        return back(f"/setup/{nxt}")

    return app


# --- middleware --------------------------------------------------------------
class SameOriginMiddleware(BaseHTTPMiddleware):
    """No login in v1 (LAN only), so at least refuse cross-site form posts (CSRF)."""

    async def dispatch(self, request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin") or request.headers.get("referer")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"error": "cross-origin request refused"}, status_code=403)
        return await call_next(request)


# --- helpers -----------------------------------------------------------------
def _ok(message: str, **extra) -> JSONResponse:
    return JSONResponse({"ok": True, "message": message, **extra})


def _fail(message: str) -> JSONResponse:
    return JSONResponse({"ok": False, "message": message})


def _pretty_json(text: str) -> str:
    import json as _json
    body = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    try:
        return _json.dumps(_json.loads(body), ensure_ascii=False, indent=2)
    except ValueError:
        return text or ""


def _mmss(sec: float | None) -> str:
    sec = int(sec or 0)
    return f"{sec // 60} мин {sec % 60:02d} с" if sec >= 60 else f"{sec} с"


def _size(n: int | float | None) -> str:
    n = float(n or 0)
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if n < 1024 or unit == "ТБ":
            return f"{n:.0f} {unit}" if unit in ("Б", "КБ") else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def _ts(t: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "—"


def _coerce(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        d = config.DEFAULTS.get(k)
        if isinstance(d, bool):
            out[k] = v in ("1", "true", "on", True)
        elif isinstance(d, (int, float)) and not isinstance(d, bool):
            try:
                out[k] = type(d)(float(str(v).replace(",", ".")))
            except ValueError:
                continue
        elif isinstance(d, list):
            continue
        else:
            out[k] = str(v).strip()
    return out


def _queue_positions(db: DB) -> dict[int, int]:
    """For each approved heavy job, how many heavy jobs are ahead of it (oldest first)."""
    heavy = [i["id"] for i in reversed(db.items(APPROVED)) if worker_mod.is_heavy(i["plan"])]
    return {item_id: n for n, item_id in enumerate(heavy)}


def _suggest_category(src: dict, cats: list[dict]) -> str:
    """Pick the category whose type matches the Jellyfin library and whose name shares words with the path."""
    want = {"movies": "movies", "tvshows": "tv"}.get(src["type"], "auto")
    words = set(re.split(r"[\W_]+", src["path"].lower()))
    last = set(re.split(r"[\W_]+", src["path"].rstrip("/").rsplit("/", 1)[-1].lower()))   # "series", "kids_movies"
    series_like = bool(last & {"series", "tv", "shows", "serials", "сериалы"})

    def score(c):
        cw = set(re.split(r"[\W_]+", c["name"].lower()))
        kind_ok = c["kind"] == ("tv" if series_like else want)   # the folder name beats Jellyfin's library type
        return (kind_ok * 4 + bool(c["enabled"]) + len(cw & last) * 5 + len(cw & words)
                - len(cw - words - {"movies", "series", "tv"}))

    return max(cats, key=score)["name"] if cats else ""


def _lang_list(text: str) -> list[str]:
    return [transcode.norm_lang(x) for x in re.split(r"[,\s]+", str(text)) if transcode.norm_lang(x)]


def _path_map_text(path_map: list[dict]) -> str:
    return "\n".join(f'{m["from"]} -> {m["to"]}' for m in path_map or [])


def _parse_path_map(text: str) -> list[dict]:
    out = []
    for line in str(text).splitlines():
        if "->" in line:
            a, b = (x.strip() for x in line.split("->", 1))
            if a and b:
                out.append({"from": a, "to": b})
    return out


def _paths_context(s: dict) -> dict:
    rows, error = [], None
    try:
        qb = worker_mod.qbit_client(s)
        paths = {"(по умолчанию)": qb.default_save_path()}
        for name, c in qb.categories().items():
            paths[name] = c.get("savePath") or ""
        for name, p in paths.items():
            if not p:
                continue
            mapped = worker_mod.map_path(p, s["path_map"])
            rows.append({"name": name, "qbit": p, "mapped": mapped, "exists": os.path.isdir(mapped)})
    except Exception as e:  # noqa: BLE001 — a wizard page must render whatever qBittorrent does
        error = str(e)
    lib = s["library_root"]
    lib_info = {"path": lib, "exists": os.path.isdir(lib), "same_disk": None, "free": None}
    probe = next((r["mapped"] for r in rows if r["exists"]), None)
    try:
        existing = lib
        while not os.path.exists(existing):
            existing = os.path.dirname(existing)
        lib_info["free"] = free_bytes(existing)
        if probe:
            lib_info["same_disk"] = os.stat(existing).st_dev == os.stat(probe).st_dev
    except OSError:
        pass
    return {"path_rows": rows, "path_error": error, "lib": lib_info}


def _sync_categories(db: DB, s: dict) -> tuple[list[dict], str | None]:
    error = None
    try:
        names = list(worker_mod.qbit_client(s).categories())
    except Exception as e:  # noqa: BLE001
        names, error = [], str(e)
    existing = db.categories()
    for name in names:
        if name not in existing:
            low = name.lower()
            kind = "tv" if re.search(r"series|tv|show|сериал", low) else "movies" if "movie" in low else "auto"
            db.upsert_category(name, enabled=0 if OFF_BY_DEFAULT.search(name) else 1, kind=kind,
                               library_dir=name.replace("/", "_"), approval="auto_if_confident",
                               delete_enabled=1, seed_ratio=2.0, seed_days=30.0)
    return _category_rows(db), error


def _category_rows(db: DB) -> list[dict]:
    rows = list(db.categories().values())
    for r in rows:
        r["hint"] = "уже обрабатывается Radarr/Sonarr/Lidarr" if re.search(
            r"(?i)radarr|sonarr|lidarr|readarr", r["name"]) else ("музыка/книги: не видео" if OFF_BY_DEFAULT.search(r["name"]) else "")
    return rows


def _save_categories(db: DB, form, with_cleanup: bool) -> None:
    for name in form.getlist("cat"):
        fields = {
            "enabled": 1 if form.get(f"enabled::{name}") else 0,
            "kind": form.get(f"kind::{name}", "auto"),
            "library_dir": (form.get(f"dir::{name}") or name.replace("/", "_")).strip(),
            "approval": form.get(f"approval::{name}", "auto_if_confident"),
        }
        if with_cleanup:
            fields["delete_enabled"] = 1 if form.get(f"delete::{name}") else 0
            for key, col in (("ratio", "seed_ratio"), ("days", "seed_days")):
                raw = (form.get(f"{key}::{name}") or "").replace(",", ".").strip()
                fields[col] = float(raw) if raw else None
        db.upsert_category(name, **fields)


def _current_cleanup_preset(db: DB) -> dict:
    cats = [c for c in db.categories().values() if c["enabled"]]
    if not cats:
        return {"name": "r2d30", "ratio": 2.0, "days": 30.0}
    c = cats[0]
    for name, (on, r, d) in CLEANUP_PRESETS.items():
        if bool(c["delete_enabled"]) == on and (not on or (c["seed_ratio"] == r and c["seed_days"] == d)):
            return {"name": name, "ratio": c["seed_ratio"], "days": c["seed_days"]}
    return {"name": "custom", "ratio": c["seed_ratio"], "days": c["seed_days"]}


def _apply_cleanup_preset(db: DB, form) -> None:
    preset = form.get("preset", "r2d30")
    if preset == "custom":
        def num(k):
            raw = (form.get(k) or "").replace(",", ".").strip()
            return float(raw) if raw else None
        on, r, d = True, num("ratio"), num("days")
    else:
        on, r, d = CLEANUP_PRESETS.get(preset, CLEANUP_PRESETS["r2d30"])
    for name in db.categories():
        db.upsert_category(name, delete_enabled=1 if on else 0, seed_ratio=r, seed_days=d)


JF_TYPES = {"movies": "movies", "tv": "tvshows", "auto": ""}
JF_TYPE_LABELS = {"movies": "Фильмы", "tvshows": "Сериалы", "": "Смешанная"}


def _jellyfin_context(db: DB, s: dict) -> dict:
    """Enabled categories with their library folder and the Jellyfin libraries that already use it."""
    ctx = {"jf_libs": [], "jf_rows": [], "jf_error": None, "JF_TYPE_LABELS": JF_TYPE_LABELS}
    if not (s.get("jellyfin_url") and s.get("jellyfin_key")):
        return ctx
    try:
        libs = Jellyfin(s["jellyfin_url"], s["jellyfin_key"]).libraries()
    except Exception as e:  # noqa: BLE001
        ctx["jf_error"] = str(e)
        return ctx
    ctx["jf_libs"] = [{"name": l["Name"], "type": l.get("CollectionType") or "",
                       "paths": l.get("Locations") or []} for l in libs]
    taken_names = {l["name"].lower() for l in ctx["jf_libs"]}
    for c in db.categories().values():
        if not c["enabled"]:
            continue
        folder = str(worker_mod.dest_root(s, c))
        jf_path = worker_mod.map_path(folder, s["jellyfin_path_map"]).rstrip("/")
        linked = [l["name"] for l in ctx["jf_libs"] if jf_path in [p.rstrip("/") for p in l["paths"]]]
        name = c["name"].replace("/", " ").replace("_", " ").strip().capitalize()
        if name.lower() in taken_names:
            name += " (jellyprep)"
        ctx["jf_rows"].append({"cat": c["name"], "folder": folder, "jf_path": jf_path, "linked": linked,
                               "type": JF_TYPES.get(c["kind"], ""), "name": name})
    return ctx


def _apply_jellyfin_libraries(db: DB, s: dict, form) -> list[dict]:
    results = []
    try:
        j = Jellyfin(s["jellyfin_url"], s["jellyfin_key"])
    except Exception as e:  # noqa: BLE001
        return [{"ok": False, "message": str(e)}]
    cats = db.categories()
    for name in form.getlist("jf_cat"):
        action = form.get(f"jf_action::{name}", "skip")
        if action == "skip" or name not in cats:
            continue
        folder = worker_mod.dest_root(s, cats[name])
        jf_path = worker_mod.map_path(str(folder), s["jellyfin_path_map"])
        try:
            folder.mkdir(parents=True, exist_ok=True)  # Jellyfin refuses folders that don't exist
            if action == "create":
                lib = (form.get(f"jf_name::{name}") or name).strip()
                j.add_library(lib, form.get(f"jf_type::{name}", ""), jf_path)
                msg = f"{name}: создана библиотека «{lib}» → {jf_path}"
            elif action.startswith("add:"):
                lib = action[4:]
                j.add_path(lib, jf_path)
                msg = f"{name}: папка {jf_path} добавлена в библиотеку «{lib}»"
            else:
                continue
            db.log("Jellyfin: " + msg)
            results.append({"ok": True, "message": msg})
        except Exception as e:  # noqa: BLE001
            results.append({"ok": False, "message": f"{name}: {e}"})
    return results or [{"ok": True, "message": "Ничего не выбрано"}]


def _overrides_from_form(form, ident: dict) -> dict:
    ov: dict = {}
    old = ident.get("overrides") or {}
    kind = form.get("kind") or ident["kind"]
    ov["kind"] = kind
    if kind != ident["kind"]:
        return ov  # kind changed: re-identify from scratch with the new kind
    if kind == "tv":
        ov["show"] = _choice(form, "show")
        eps = dict(old.get("episodes") or {})
        for key in form:
            if key.startswith("ep_file::"):
                i = key.split("::", 1)[1]
                f = form[key]
                se, ep = form.get(f"ep_s::{i}", ""), form.get(f"ep_e::{i}", "")
                os_, oe = form.get(f"ep_os::{i}", ""), form.get(f"ep_oe::{i}", "")
                if se.isdigit() and ep.isdigit() and (se, ep) != (os_, oe):
                    eps[f] = [int(se), int(ep)]
        if eps:
            ov["episodes"] = eps
    else:
        ov["movies"] = {str(i): _choice(form, f"movie{i}") for i in range(len(ident.get("movies", [])))}
    return ov


def _choice(form, prefix: str) -> dict:
    pick = form.get(f"{prefix}_pick", "")
    if pick == "manual":
        title = (form.get(f"{prefix}_title") or "").strip()
        year = (form.get(f"{prefix}_year") or "").strip()
        return {"title": title, "year": int(year) if year.isdigit() else None} if title else {}
    if pick == "tmdb":
        raw = re.sub(r"\D", "", form.get(f"{prefix}_tmdb") or "")
        return {"tmdb_id": int(raw)} if raw else {}
    if pick.isdigit():
        return {"tmdb_id": int(pick)}
    return {}


app = None


def run() -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=os.environ.get("JELLYPREP_HOST", "0.0.0.0"),
                port=int(os.environ.get("JELLYPREP_PORT", "8765")), log_level="warning")


if __name__ == "__main__":
    run()
