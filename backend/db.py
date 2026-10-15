"""SQLite persistence for sessions, alerts, attention history and enquiries."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "vigidriver.db")

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_label  TEXT    NOT NULL DEFAULT 'Driver_01',
                started_at    REAL    NOT NULL,
                ended_at      REAL,
                duration_s    REAL,
                model_used    TEXT    NOT NULL DEFAULT 'mobilenetv2+vigcnn',
                threshold     REAL    NOT NULL DEFAULT 0.62,
                avg_drowsiness REAL,
                peak_drowsiness REAL,
                alert_count   INTEGER NOT NULL DEFAULT 0,
                critical_count INTEGER NOT NULL DEFAULT 0,
                frames_total  INTEGER NOT NULL DEFAULT 0,
                frames_drowsy INTEGER NOT NULL DEFAULT 0,
                attention_hist TEXT,          -- JSON list of downsampled probabilities
                outcome       TEXT    NOT NULL DEFAULT 'safe'  -- safe|caution|unsafe
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id),
                ts          REAL    NOT NULL,
                tier        TEXT    NOT NULL,      -- warning | critical
                p_drowsy    REAL    NOT NULL,
                note        TEXT
            );

            CREATE TABLE IF NOT EXISTS attention (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id),
                ts          REAL    NOT NULL,
                p_drowsy    REAL    NOT NULL,
                p_mobilenet REAL    NOT NULL,
                p_vigcnn    REAL    NOT NULL,
                tier        TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS contact_messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL,
                email      TEXT    NOT NULL,
                subject    TEXT    NOT NULL DEFAULT 'General enquiry',
                message    TEXT    NOT NULL,
                created_at REAL    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id);
            CREATE INDEX IF NOT EXISTS idx_attention_session ON attention(session_id);
            """
        )


# -- sessions ------------------------------------------------------------------

def create_session(driver_label: str, threshold: float) -> int:
    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (driver_label, started_at, threshold) VALUES (?, ?, ?)",
            (driver_label, time.time(), threshold),
        )
        return int(cur.lastrowid)


def end_session(session_id: int) -> dict[str, Any] | None:
        with _lock, _connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None or row["ended_at"] is not None:
                return None
            att = conn.execute(
                "SELECT p_drowsy FROM attention WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
            probs = [r["p_drowsy"] for r in att]
            counts = conn.execute(
                "SELECT tier, COUNT(*) AS n FROM alerts WHERE session_id = ? GROUP BY tier",
                (session_id,),
            ).fetchall()
            warning_count = sum(c["n"] for c in counts if c["tier"] == "warning")
            critical_count = sum(c["n"] for c in counts if c["tier"] == "critical")
            avg_p = sum(probs) / len(probs) if probs else 0.0
            peak_p = max(probs) if probs else 0.0
            frames_total = len(probs)
            frames_drowsy = sum(1 for p in probs if p >= row["threshold"])
            # Audit outcome: unsafe on any critical alert, caution on warnings or
            # a high sustained average, safe otherwise.
            if critical_count > 0 or avg_p >= 0.55:
                outcome = "unsafe"
            elif warning_count > 0 or avg_p >= 0.35:
                outcome = "caution"
            else:
                outcome = "safe"
            # Downsample the attention trace to at most 240 points for storage.
            hist = probs
            if len(hist) > 240:
                step = len(hist) / 240.0
                hist = [hist[int(i * step)] for i in range(240)]
            ended = time.time()
            conn.execute(
                """
                UPDATE sessions SET ended_at = ?, duration_s = ?, avg_drowsiness = ?,
                    peak_drowsiness = ?, alert_count = ?, critical_count = ?,
                    frames_total = ?, frames_drowsy = ?, attention_hist = ?, outcome = ?
                WHERE id = ?
                """,
                (
                    ended,
                    round(ended - row["started_at"], 1),
                    round(avg_p, 4),
                    round(peak_p, 4),
                    warning_count,
                    critical_count,
                    frames_total,
                    frames_drowsy,
                    json.dumps([round(p, 4) for p in hist]),
                    outcome,
                    session_id,
                ),
            )
        # Read back OUTSIDE the write transaction: a fresh connection inside
        # the with-block would see the pre-commit WAL snapshot and return
        # stale ended_at/duration fields.
        return get_session(session_id)


def close_stale_sessions() -> int:
    """Finalise sessions left open by a previous server run (restart cleanup).

    Runs at startup so a crash/restart never leaves ghost "Running" rows in
    the audit table. Returns the number of sessions closed.
    """
    with _connect() as conn:
        rows = conn.execute("SELECT id FROM sessions WHERE ended_at IS NULL").fetchall()
    closed = 0
    for r in rows:
        if end_session(int(r["id"])) is not None:
            closed += 1
    return closed


def get_session(session_id: int) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return dict(row) | {"attention_hist": json.loads(row["attention_hist"] or "[]")}


def list_sessions(limit: int = 100) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) | {"attention_hist": json.loads(r["attention_hist"] or "[]")} for r in rows]


def session_stats() -> dict[str, Any]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n_sessions,
                   COALESCE(SUM(alert_count), 0)      AS warnings,
                   COALESCE(SUM(critical_count), 0)   AS criticals,
                   COALESCE(AVG(duration_s), 0)       AS avg_duration,
                   COALESCE(SUM(frames_total), 0)     AS frames
            FROM sessions WHERE ended_at IS NOT NULL
            """
        ).fetchone()
        outcomes = conn.execute(
            "SELECT outcome, COUNT(*) AS n FROM sessions WHERE ended_at IS NOT NULL GROUP BY outcome"
        ).fetchall()
        by_outcome = {r["outcome"]: r["n"] for r in outcomes}
        safe = by_outcome.get("safe", 0)
        caution = by_outcome.get("caution", 0)
        unsafe = by_outcome.get("unsafe", 0)
        total = safe + caution + unsafe
        return {
            "sessions": row["n_sessions"],
            "warnings": row["warnings"],
            "criticals": row["criticals"],
            "avg_duration_s": round(row["avg_duration"] or 0, 1),
            "frames_analyzed": row["frames"],
            "outcome_counts": {"safe": safe, "caution": caution, "unsafe": unsafe},
            "safety_score": round(100.0 * (safe + 0.5 * caution) / total) if total else 100,
        }


# -- alerts / attention ---------------------------------------------------------

def add_alert(session_id: int, tier: str, p_drowsy: float, note: str | None = None) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO alerts (session_id, ts, tier, p_drowsy, note) VALUES (?, ?, ?, ?, ?)",
            (session_id, time.time(), tier, p_drowsy, note),
        )


def add_attention(
    session_id: int,
    p_drowsy: float,
    p_mobilenet: float,
    p_vigcnn: float,
    tier: str,
) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO attention (session_id, ts, p_drowsy, p_mobilenet, p_vigcnn, tier) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, time.time(), p_drowsy, p_mobilenet, p_vigcnn, tier),
        )


def attention_distribution() -> list[dict[str, Any]]:
    """Histogram of attention weights (drowsiness probability) over all sessions."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT p_drowsy FROM attention"
        ).fetchall()
    bins = [round(0.05 * i, 2) for i in range(21)]  # 0.00 .. 1.00
    edges = bins + [1.01]
    counts = [0] * 20
    for r in rows:
        p = min(max(r["p_drowsy"], 0.0), 0.999)
        counts[min(int(p * 20), 19)] += 1
    return [
        {"bin_start": edges[i], "bin_end": edges[i + 1], "count": counts[i]}
        for i in range(20)
    ]


def alerts_for_session(session_id: int) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE session_id = ? ORDER BY ts", (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# -- contact ---------------------------------------------------------------------

def add_contact_message(name: str, email: str, subject: str, message: str) -> int:
    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO contact_messages (name, email, subject, message, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, email, subject, message, time.time()),
        )
        return int(cur.lastrowid)


def list_contact_messages(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM contact_messages ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
