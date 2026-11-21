"""SQLite storage: settings, categories, items, log."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS categories (
    name        TEXT PRIMARY KEY,
    enabled     INTEGER NOT NULL DEFAULT 0,
    kind        TEXT NOT NULL DEFAULT 'auto',      -- movies | tv | auto
    library_dir TEXT NOT NULL DEFAULT '',
    approval    TEXT NOT NULL DEFAULT 'auto_if_confident',  -- auto | auto_if_confident | always_confirm
    delete_enabled INTEGER NOT NULL DEFAULT 1,
    seed_ratio  REAL,
    seed_days   REAL
);
CREATE TABLE IF NOT EXISTS items (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL DEFAULT 'torrent',
    hash         TEXT UNIQUE,
    name         TEXT NOT NULL,
    category     TEXT NOT NULL DEFAULT '',
    content_path TEXT NOT NULL,
    state        TEXT NOT NULL DEFAULT 'new',
    ident        TEXT,
    plan         TEXT,
    result       TEXT,
    error        TEXT,
    ratio        REAL,
    seeding_time INTEGER,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    processed_at REAL,
    removed_at   REAL
);
CREATE TABLE IF NOT EXISTS probe_cache (
    path  TEXT PRIMARY KEY,
    size  INTEGER NOT NULL,
    mtime REAL NOT NULL,
    info  TEXT
);
CREATE TABLE IF NOT EXISTS log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    level   TEXT NOT NULL,
    item_id INTEGER,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS log_item ON log(item_id);
"""

JSON_COLS = ("ident", "plan", "result", "options", "trace")
# Columns added after v0.1; applied to existing databases on start.
MIGRATIONS = [("options", "options TEXT"),
              ("trace", "trace TEXT"),   # when and why each pipeline step happened
              ("note", "note TEXT")]     # why the item is waiting right now (space, …)

# Item states
NEW, REVIEW, APPROVED, RUNNING, DONE, REMOVED, SKIPPED, ERROR, DELETED = (
    "new", "review", "approved", "running", "done", "removed", "skipped", "error", "deleted")


class DB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._migrate()

    def _migrate(self) -> None:
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(items)")}
        for name, ddl in MIGRATIONS:
            if name not in cols:
                self._conn.execute(f"ALTER TABLE items ADD COLUMN {ddl}")
        self._conn.commit()

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def query(self, sql: str, params: tuple | dict = ()) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_decode(dict(r)) for r in rows]

    # --- settings -------------------------------------------------------
    def get_settings(self) -> dict[str, Any]:
        return {r["key"]: json.loads(r["value"]) for r in self.query("SELECT * FROM settings")}

    def set_setting(self, key: str, value: Any) -> None:
        self.execute("INSERT INTO settings(key, value) VALUES(?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, json.dumps(value, ensure_ascii=False)))

    # --- categories -----------------------------------------------------
    def categories(self) -> dict[str, dict]:
        return {r["name"]: r for r in self.query("SELECT * FROM categories ORDER BY name")}

    def upsert_category(self, name: str, **fields: Any) -> None:
        existing = self.query("SELECT name FROM categories WHERE name=?", (name,))
        if not existing:
            self.execute("INSERT INTO categories(name) VALUES(?)", (name,))
        if fields:
            cols = ", ".join(f"{k}=?" for k in fields)
            self.execute(f"UPDATE categories SET {cols} WHERE name=?", (*fields.values(), name))

    # --- items ----------------------------------------------------------
    def add_item(self, **fields: Any) -> int:
        now = time.time()
        fields = {**fields, "created_at": now, "updated_at": now}
        fields = {k: _encode(k, v) for k, v in fields.items()}
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        return self.execute(f"INSERT INTO items({cols}) VALUES({marks})", tuple(fields.values())).lastrowid

    def update_item(self, item_id: int, **fields: Any) -> None:
        fields = {**fields, "updated_at": time.time()}
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = tuple(_encode(k, v) for k, v in fields.items())
        self.execute(f"UPDATE items SET {cols} WHERE id=?", (*vals, item_id))

    def trace(self, item_id: int, **fields: Any) -> None:
        """Merge fields into the item's pipeline trace."""
        with self._lock:
            row = self._conn.execute("SELECT trace FROM items WHERE id=?", (item_id,)).fetchone()
            current = json.loads(row["trace"]) if row and row["trace"] else {}
            current.update(fields)
            self._conn.execute("UPDATE items SET trace=? WHERE id=?",
                               (json.dumps(current, ensure_ascii=False), item_id))
            self._conn.commit()

    def item(self, item_id: int) -> dict | None:
        rows = self.query("SELECT * FROM items WHERE id=?", (item_id,))
        return rows[0] if rows else None

    def items(self, state: str | None = None) -> list[dict]:
        if state:
            return self.query("SELECT * FROM items WHERE state=? ORDER BY id DESC", (state,))
        return self.query("SELECT * FROM items ORDER BY id DESC")

    def claim(self, item_id: int, from_state: str, to_state: str) -> bool:
        """Atomically move an item between states; False if someone else changed it first."""
        cur = self.execute("UPDATE items SET state=?, updated_at=? WHERE id=? AND state=?",
                           (to_state, time.time(), item_id, from_state))
        return cur.rowcount == 1

    def known_paths(self) -> set[str]:
        return {r["content_path"] for r in self.query(
            "SELECT content_path FROM items WHERE state NOT IN ('skipped', 'error')")}

    def active_paths(self) -> set[str]:
        """Content paths of items that are still in progress (not finished, skipped or failed)."""
        return {r["content_path"] for r in self.query(
            "SELECT content_path FROM items WHERE state IN ('new', 'review', 'approved', 'running')")}

    def probe_get(self, path: str, size: int, mtime: float) -> dict | None | bool:
        """Cached ffprobe summary; False when not cached (None = cached as unreadable)."""
        rows = self.query("SELECT size, mtime, info FROM probe_cache WHERE path=?", (path,))
        if not rows or rows[0]["size"] != size or abs(rows[0]["mtime"] - mtime) > 1:
            return False
        return json.loads(rows[0]["info"]) if rows[0]["info"] else None

    def probe_set(self, path: str, size: int, mtime: float, info: dict | None) -> None:
        self.execute("INSERT INTO probe_cache(path, size, mtime, info) VALUES(?, ?, ?, ?) "
                     "ON CONFLICT(path) DO UPDATE SET size=excluded.size, mtime=excluded.mtime, info=excluded.info",
                     (path, size, mtime, json.dumps(info, ensure_ascii=False) if info else None))

    def known_hashes(self) -> set[str]:
        return {r["hash"] for r in self.query("SELECT hash FROM items WHERE hash IS NOT NULL")}

    # --- log ------------------------------------------------------------
    def log(self, message: str, level: str = "info", item_id: int | None = None) -> None:
        self.execute("INSERT INTO log(ts, level, item_id, message) VALUES(?, ?, ?, ?)",
                     (time.time(), level, item_id, message))

    def logs(self, item_id: int | None = None, limit: int = 300) -> list[dict]:
        if item_id is None:
            return self.query("SELECT * FROM log ORDER BY id DESC LIMIT ?", (limit,))
        return self.query("SELECT * FROM log WHERE item_id=? ORDER BY id DESC LIMIT ?", (item_id, limit))


def _encode(key: str, value: Any) -> Any:
    if key in JSON_COLS and value is not None:
        return json.dumps(value, ensure_ascii=False)
    return value


def _decode(row: dict) -> dict:
    for k in JSON_COLS:
        if row.get(k):
            row[k] = json.loads(row[k])
    return row
