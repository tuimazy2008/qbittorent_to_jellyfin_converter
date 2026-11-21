"""One-page board: every item as a row of five stage cells with a status line.

Stages: download → identify (incl. review) → process → library → seeding (incl. removal of the original).
"""
from __future__ import annotations

import time
from typing import Any

from . import llm, pipeline, worker as w
from .db import APPROVED, DONE, ERROR, NEW, REMOVED, REVIEW, RUNNING, SKIPPED

STAGES = ["Загрузка", "Распознавание", "Обработка", "Библиотека", "Раздача"]
GROUPS = [("decision", "Нужно твоё решение"), ("work", "В работе"), ("seeding", "Раздаются"), ("done", "Готово")]
RANK = {"error": 5, "wait": 4, "current": 3, "todo": 2, "done": 1, "skip": 0}


def _cell(status: str, progress: float | None = None) -> dict:
    return {"status": status, "progress": None if progress is None else round(max(0.0, min(progress, 1.0)), 3)}


def _merge(*stages: dict) -> dict:
    """Combine pipeline stages into one cell: the most attention-worthy status wins."""
    st = max(stages, key=lambda s: RANK[s["status"]])
    if all(s["status"] in ("done", "skip") for s in stages):
        status = "skip" if all(s["status"] == "skip" for s in stages) else "done"
        return _cell(status)
    return _cell(st["status"], st.get("progress"))


def _dur(sec: float) -> str:
    sec = int(max(sec, 0))
    if sec >= 86400:
        return f"{sec // 86400} дн."
    if sec >= 3600:
        return f"{sec // 3600} ч {sec % 3600 // 60} мин"
    return f"{max(sec // 60, 1)} мин"


def title_of(item: dict) -> tuple[str, str]:
    ident = item.get("ident") or {}
    if ident.get("kind") == "tv":
        c = (ident.get("show") or {}).get("chosen") or {}
        if c.get("title"):
            return (f"{c['title']} ({c['year']})" if c.get("year") else c["title"],
                    f"{len(ident.get('episodes') or [])} сер.")
    if ident.get("kind") == "movie":
        movies = ident.get("movies") or []
        c = (movies[0].get("chosen") if movies else None) or {}
        if c.get("title"):
            extra = f" и ещё {len(movies) - 1}" if len(movies) > 1 else ""
            return (f"{c['title']} ({c['year']})" if c.get("year") else c["title"]) + extra, "фильм"
    return item["name"], ""


def item_row(item: dict, s: dict, cat: dict | None, logs: list[dict], queue_ahead: int, job: dict | None,
             job_started: float | None = None, conv: Any = None) -> dict[str, Any]:
    p = pipeline.build(item, s, cat, logs, queue_ahead, job)
    st = {x["key"]: x for x in p["stages"]}
    state = item["state"]
    is_torrent = bool(item.get("hash")) and item.get("source", "torrent") == "torrent"

    cells = [
        _cell("done") if is_torrent else _cell("skip"),
        _merge(st["ident"], st["review"]),
        _merge(st["prepare"]),
        _merge(st["library"]),
        _merge(st["seeding"], st["removed"]),
    ]
    title, sub = title_of(item)
    actions: list[dict] = []
    cur = p["current"]
    reason = cur["lines"][-1] if cur["lines"] else ""

    if state == REVIEW:
        icon, group = "❓", "decision"
        problems = (item.get("ident") or {}).get("problems", []) + (item.get("plan") or {}).get("problems", [])
        line = "нужно решение: " + (problems[0] if problems else (item.get("trace") or {}).get("decision") or "проверьте распознавание")
        if (item.get("note") or "").startswith(("модель", "предложение модели")):
            icon, line = "🤖", item["note"]
        if conv is not None:
            took = time.time() - conv.started
            line = (f"модель {conv.model} думает {_mmss(took)}"
                    + (f" · получено фрагментов ответа: {conv.chunks}" if conv.chunks else " · читает вопрос"))
        actions.append({"label": "Открыть", "open": True})
        if conv is not None:
            actions.append({"label": "Отменить", "post": f"/items/{item['id']}/llm-cancel"})
        elif item.get("note") == w.LLM_THINKING:
            actions.append({"label": "Сбросить", "post": f"/items/{item['id']}/llm-cancel"})
        elif llm.configured(s):
            actions.append({"label": "Спросить модель", "post": f"/items/{item['id']}/ask-llm"})
        if (item.get("plan") or {}).get("actions"):
            actions.append({"label": "Подтвердить", "post": f"/items/{item['id']}/approve"})
    elif state == ERROR:
        icon, group, line = "✗", "decision", f"ошибка: {item.get('error') or reason}"
        actions += [{"label": "Открыть", "open": True}, {"label": "Повторить", "post": f"/items/{item['id']}/approve"}]
    elif state == NEW:
        icon, group, line = "…", "work", "распознаётся"
    elif state == APPROVED:
        icon, group, line = "⏸", "work", reason
        if w.needs_window(item.get("plan")) and s.get("night_only") and not (item.get("plan") or {}).get("run_now"):
            actions.append({"label": "Сейчас", "post": f"/items/{item['id']}/run-now"})
    elif state == RUNNING:
        icon, group, line = "⚙", "work", reason
        if job and job.get("item_id") == item["id"] and job.get("since"):
            done_frac = job.get("overall", 0) - job.get("frac0", 0)
            elapsed = time.time() - job["since"]
            if done_frac > 0.002 and elapsed > 30:
                line += f" · осталось ~{_dur(elapsed * (1 - job['overall']) / done_frac)}"
    elif state == DONE and is_torrent:
        icon, group = "✓", "seeding"
        line = "в библиотеке · " + "; ".join(st["seeding"]["lines"]) + _seed_eta(cat, item)
        if item.get("note"):
            line += f" · {item['note']}"
    elif state == SKIPPED:
        icon, group, line = "⤼", "done", "пропущено"
    else:  # removed, or done without a torrent
        icon, group = "✓", "done"
        line = "; ".join(x for x in (st["prepare"]["lines"][:1] + st["removed"]["lines"][:1]) if x) or "готово"
        if item.get("note"):
            line += f" · {item['note']}"

    return {"id": item["id"], "key": f"item-{item['id']}", "title": title, "sub": sub, "name": item["name"],
            "category": item["category"], "folder": item.get("source") == "folder", "cells": cells,
            "icon": icon, "line": line, "group": group, "actions": actions,
            "updated": item.get("updated_at") or item["created_at"]}


def _mmss(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60} мин {sec % 60:02d} с" if sec >= 60 else f"{sec} с"


def _seed_eta(cat: dict | None, item: dict) -> str:
    if not cat or not cat.get("delete_enabled") or not cat.get("seed_days"):
        return ""
    left = cat["seed_days"] * 86400 - (item.get("seeding_time") or 0)
    return f" · удалится не позже чем через {_dur(left)}" if left > 0 else ""


def download_row(t: dict) -> dict[str, Any]:
    eta = t.get("eta") or 0
    speed = (t.get("dlspeed") or 0) / 1e6
    state = t.get("state", "")
    if state in ("pausedDL", "stoppedDL"):
        line = f"на паузе · {t['progress']:.0%}"
    elif state in ("stalledDL", "metaDL", "queuedDL", "checkingDL", "forcedMetaDL"):
        line = f"ждёт пиров · {t['progress']:.0%}" if state != "queuedDL" else f"в очереди qBittorrent · {t['progress']:.0%}"
    else:
        line = f"{t['progress']:.0%} · {speed:.1f} МБ/с" + (f" · осталось ~{_dur(eta)}" if 0 < eta < 8640000 else "")
    return {"id": None, "key": f"dl-{t['hash'][:12]}", "title": t["name"], "sub": "загружается", "name": t["name"],
            "category": t.get("category", ""), "folder": False,
            "cells": [_cell("current", t.get("progress", 0))] + [_cell("todo")] * 4,
            "icon": "↓", "line": line, "group": "work", "actions": [], "updated": time.time()}


def build(items: list[dict], downloads: list[dict], s: dict, cats: dict, logs_for, ahead: dict, job: dict | None,
          job_started: float | None, llm_jobs: dict | None = None) -> dict[str, Any]:
    rows = [item_row(i, s, cats.get(i["category"]), logs_for(i["id"]), ahead.get(i["id"], 0), job, job_started,
                     (llm_jobs or {}).get(i["id"])) for i in items]
    rows += [download_row(t) for t in downloads]
    order = {"⚙": 0, "↓": 1, "⏸": 2, "…": 3}
    groups = []
    for key, label in GROUPS:
        g = [r for r in rows if r["group"] == key]
        if key == "work":
            g.sort(key=lambda r: (order.get(r["icon"], 9), -r["updated"]))
        else:
            g.sort(key=lambda r: -r["updated"])
        groups.append({"key": key, "label": label, "rows": g, "collapsed": key == "done"})
    counts = [0] * 5  # how many items are currently at each stage
    for r in rows:
        if r["group"] in ("work", "decision", "seeding"):
            idx = next((n for n, c in enumerate(r["cells"]) if c["status"] in ("current", "wait", "error")), None)
            if idx is not None:
                counts[idx] += 1
    return {"stages": STAGES, "counts": counts, "groups": groups, "dry_run": bool(s.get("dry_run"))}
