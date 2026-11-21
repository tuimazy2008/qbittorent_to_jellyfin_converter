"""Where an item is in the pipeline, and why: found → identified → review → prepare → library → seeding → removed."""
from __future__ import annotations

import re
from typing import Any

from . import worker as w
from .db import APPROVED, DONE, ERROR, NEW, REMOVED, REVIEW, RUNNING, SKIPPED

DONE_, CURRENT, WAIT, TODO, SKIP, FAIL = "done", "current", "wait", "todo", "skip", "error"
ACTION_LABELS = {"hardlink": "хардлинк", "transcode": "перекодирование", "process": "обработка звука",
                 "remux": "перепаковка", "exists": "уже в библиотеке", "copy": "копия", "failed": "не обработан"}


def _stage(key: str, title: str, status: str, *lines: str, when: float | None = None,
           progress: float | None = None) -> dict:
    return {"key": key, "title": title, "status": status, "lines": [l for l in lines if l],
            "when": when, "progress": progress}


def _from_log(logs: list[dict], pattern: str) -> dict | None:
    rx = re.compile(pattern)
    return next((l for l in reversed(logs) if rx.search(l["message"])), None)  # logs are newest first


def describe_identity(ident: dict) -> str:
    if not ident:
        return ""
    if ident.get("kind") == "tv":
        c = (ident.get("show") or {}).get("chosen") or {}
        eps = ident.get("episodes") or []
        seasons = sorted({e["season"] for e in eps})
        s_txt = f"сезон {seasons[0]}" if len(seasons) == 1 else f"сезоны {seasons[0]}–{seasons[-1]}" if seasons else ""
        return (f"Сериал «{c.get('title', '?')}» ({c.get('year') or '?'}), {len(eps)} сер., {s_txt}"
                + (f" · TMDB {c['id']}" if c.get("id") else " · без TMDB"))
    movies = ident.get("movies") or []
    names = [f"«{(m.get('chosen') or {}).get('title', '?')}» ({(m.get('chosen') or {}).get('year') or '?'})" for m in movies]
    ids = [str((m.get("chosen") or {}).get("id")) for m in movies if (m.get("chosen") or {}).get("id")]
    return ("Фильм " if len(names) == 1 else f"Фильмы ({len(names)}): ") + ", ".join(names[:3]) + \
        (" …" if len(names) > 3 else "") + (f" · TMDB {', '.join(ids[:3])}" if ids else "")


def plan_counts(plan: dict | None) -> str:
    counts: dict[str, int] = {}
    for a in (plan or {}).get("actions", []):
        if a["type"] == "video":
            counts[a["action"]] = counts.get(a["action"], 0) + 1
    return ", ".join(f"{ACTION_LABELS.get(k, k)} ×{v}" if v > 1 else ACTION_LABELS.get(k, k)
                     for k, v in counts.items())


def build(item: dict, s: dict, cat: dict | None, logs: list[dict], queue_ahead: int = 0,
          job: dict | None = None) -> dict[str, Any]:
    st = item["state"]
    tr = item.get("trace") or {}
    ident = item.get("ident") or {}
    plan = item.get("plan") or {}
    result = item.get("result") or {}
    is_torrent = item.get("source", "torrent") == "torrent" and item.get("hash")
    stages = []

    # 1. found
    src = (f"Торрент в qBittorrent, категория «{item['category']}»" if is_torrent
           else f"Папка из «Переноса», категория «{item['category']}»")
    stages.append(_stage("found", "Найден", DONE_, src, when=item["created_at"]))

    # 2. identified
    if st == NEW:
        stages.append(_stage("ident", "Распознан", CURRENT, "ждёт распознавания в ближайшем цикле"))
    elif not ident:
        stages.append(_stage("ident", "Распознан", FAIL, item.get("error") or "не удалось разобрать"))
    else:
        conf = ident.get("confidence") or 0
        ev = _from_log(logs, r"^Распознано")
        stages.append(_stage("ident", "Распознан", DONE_, describe_identity(ident), f"уверенность {conf:.0%}",
                             when=tr.get("identified_at") or (ev or {}).get("ts")))

    # 3. review / decision
    problems = (ident.get("problems") or []) + (plan.get("problems") or [])
    manual = _from_log(logs, r"^Одобрено вручную")
    approved_by = tr.get("approved_by") or ("manual" if manual else None)
    decision = tr.get("decision") or ""
    if not ident:
        stages.append(_stage("review", "Проверка", TODO))
    elif st == REVIEW:
        why = f"нужно ваше решение: {decision}" if decision else "нужно ваше решение"
        stages.append(_stage("review", "Проверка", WAIT, why, *problems[:4],
                             "… и ещё" if len(problems) > 4 else ""))
    elif st == SKIPPED:
        stages.append(_stage("review", "Проверка", SKIP, "пропущено вами"))
    elif approved_by == "manual":
        stages.append(_stage("review", "Проверка", DONE_, "подтверждено вручную",
                             when=tr.get("approved_at") or (manual or {}).get("ts")))
    else:
        stages.append(_stage("review", "Проверка", DONE_,
                             f"подтверждено автоматически: {decision}" if decision else "подтверждено автоматически",
                             when=tr.get("approved_at") or tr.get("identified_at")))

    # 4. prepare
    actions = plan_counts(plan)
    heavy, window = w.is_heavy(plan), w.needs_window(plan)
    if st in (NEW, REVIEW, SKIPPED) or not ident:
        stages.append(_stage("prepare", "Подготовка", SKIP if st == SKIPPED else TODO,
                             f"план: {actions}" if actions and st != SKIPPED else ""))
    elif st == APPROVED:
        if s.get("dry_run"):
            why = "dry-run: ничего не выполняется, пока режим не выключен в настройках"
        elif item.get("note"):
            why = item["note"]
        elif window and s.get("night_only") and not plan.get("run_now") and not w.in_window(s):
            why = f"перекодирование ждёт ночного окна {s['night_start']}–{s['night_end']} (можно «Выполнить сейчас»)"
        elif heavy and queue_ahead:
            why = f"в очереди обработки, впереди задач: {queue_ahead}"
        elif heavy:
            why = "следующая в очереди обработки"
        else:
            why = "выполнится в ближайшем цикле (через несколько минут)"
        stages.append(_stage("prepare", "Подготовка", WAIT, f"план: {actions}", why))
    elif st == RUNNING:
        prog, line = None, "выполняется"
        if job and job.get("item_id") == item["id"]:
            prog = ((job["n"] - 1) + job["percent"] / 100) / max(job["total"], 1)
            line = (f"файл {job['n']} из {job['total']}: {job['file']} — {job['percent']:.0f}%"
                    + (f", скорость {job['speed']}" if job.get("speed") else ""))
        stages.append(_stage("prepare", "Подготовка", CURRENT, f"план: {actions}", line,
                             when=tr.get("started_at"), progress=prog))
    elif st == ERROR:
        stages.append(_stage("prepare", "Подготовка", FAIL, item.get("error") or "ошибка"))
    else:
        files = result.get("files", [])
        before = sum(f.get("size", 0) for f in files if f.get("type") == "video")
        after = sum(f.get("out_size", f.get("size", 0)) for f in files if f.get("type") == "video")
        took = sum(f.get("seconds", 0) for f in files)
        size = f"{_gb(before)} → {_gb(after)}" if before and after and abs(before - after) > 1e6 else ""
        stages.append(_stage("prepare", "Подготовка", DONE_, f"сделано: {actions}" if actions else "",
                             size, f"заняло {_dur(took)}" if took > 60 else "", item.get("note") or "",
                             when=item.get("processed_at")))

    # 5. library
    if st in (DONE, REMOVED):
        stages.append(_stage("library", "В библиотеке", DONE_, plan.get("target") or "",
                             "файлы проверены", when=item.get("processed_at")))
    else:
        stages.append(_stage("library", "В библиотеке", TODO, plan.get("target") or ""))

    # 6. seeding
    if not is_torrent:
        stages.append(_stage("seeding", "Раздача", SKIP, "без торрента"))
    elif st == REMOVED:
        stages.append(_stage("seeding", "Раздача", DONE_, f"рейтинг {item.get('ratio') or 0:.2f}, "
                             f"{(item.get('seeding_time') or 0) / 86400:.0f} дн."))
    else:
        due, reason = w.deletion_due(cat, item.get("ratio") or 0, item.get("seeding_time") or 0)
        prog = _seed_progress(cat, item)
        status = CURRENT if st == DONE else TODO
        line = (f"рейтинг {item.get('ratio') or 0:.2f}, раздаётся {(item.get('seeding_time') or 0) / 86400:.0f} дн.")
        target = reason if not due else f"условие выполнено: {reason}"
        stages.append(_stage("seeding", "Раздача", status, line, target if st == DONE else "",
                             progress=prog if st == DONE else None))

    # 7. original removed
    if st == REMOVED:
        ev = _from_log(logs, r"^Торрент удалён|^Перенос завершён")
        reason = tr.get("removed_reason") or (ev or {}).get("message", "")
        stages.append(_stage("removed", "Оригинал удалён", DONE_, reason, when=item.get("removed_at")))
    elif not is_torrent and not (item.get("options") or {}).get("delete_source", True):
        stages.append(_stage("removed", "Оригинал удалён", SKIP, "оригинал оставлен по вашему выбору"))
    elif st == DONE and is_torrent:
        due, _ = w.deletion_due(cat, item.get("ratio") or 0, item.get("seeding_time") or 0)
        why = item.get("note") or ("dry-run: торрент не удаляется" if s.get("dry_run") and due
                                    else "в ближайшем цикле" if due else "после выполнения условия раздачи")
        stages.append(_stage("removed", "Оригинал удалён", WAIT if due else TODO, why))
    elif st == DONE:
        stages.append(_stage("removed", "Оригинал удалён", SKIP, "обработано на месте"))
    else:
        stages.append(_stage("removed", "Оригинал удалён", TODO))

    current = next((x for x in stages if x["status"] in (CURRENT, WAIT, FAIL)), None)
    if st == SKIPPED:
        current = stages[2]
    if current is None:
        current = stages[-1] if st == REMOVED else next((x for x in reversed(stages) if x["status"] == DONE_), stages[0])
    return {"stages": stages, "current": current}


def _seed_progress(cat: dict | None, item: dict) -> float | None:
    if not cat or not cat.get("delete_enabled"):
        return None
    parts = []
    if cat.get("seed_ratio"):
        parts.append((item.get("ratio") or 0) / cat["seed_ratio"])
    if cat.get("seed_days"):
        parts.append((item.get("seeding_time") or 0) / 86400 / cat["seed_days"])
    return min(max(parts), 1.0) if parts else None


def _gb(n: float) -> str:
    return f"{n / 1024 ** 3:.1f} ГБ"


def _dur(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600} ч {sec % 3600 // 60} мин" if sec >= 3600 else f"{sec // 60} мин"
