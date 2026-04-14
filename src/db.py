from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Optional


DATA_DIR = Path("./data")
DB_PATH = DATA_DIR / "app.db"

_init_lock = threading.Lock()
_initialized = False


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH.as_posix(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def get_conn() -> sqlite3.Connection:
    init_db()
    return _connect()


def init_db() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return

        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT NOT NULL UNIQUE,
                  password_hash BLOB NOT NULL,
                  password_salt BLOB NOT NULL,
                  role TEXT NOT NULL CHECK(role IN ('admin','merchant')),
                  created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS merchants (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  slug TEXT NOT NULL UNIQUE,
                  name TEXT NOT NULL,
                  personality TEXT NOT NULL,
                  owner_user_id INTEGER NOT NULL,
                  enabled INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL DEFAULT (datetime('now')),
                  FOREIGN KEY(owner_user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS kb_versions (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  merchant_id INTEGER NOT NULL,
                  qa_pairs_count INTEGER NOT NULL DEFAULT 0,
                  persist_directory TEXT NOT NULL,
                  mode TEXT NOT NULL,
                  created_at TEXT NOT NULL DEFAULT (datetime('now')),
                  FOREIGN KEY(merchant_id) REFERENCES merchants(id) ON DELETE CASCADE
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

        _initialized = True


def count_users(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(1) AS c FROM users;")
    row = cur.fetchone()
    return int(row["c"]) if row else 0


def fetch_one(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
    cur = conn.execute(sql, tuple(params))
    return cur.fetchone()


def fetch_all(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    cur = conn.execute(sql, tuple(params))
    return list(cur.fetchall())

