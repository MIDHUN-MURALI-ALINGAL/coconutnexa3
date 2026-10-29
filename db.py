import os
import sqlite3
import time
from contextlib import contextmanager

DB_PATH = os.getenv("DB_PATH", "coconexa.db")


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    with connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            message TEXT NOT NULL,
            remind_at REAL NOT NULL,
            sent INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS prefs (
            phone TEXT PRIMARY KEY,
            lang TEXT DEFAULT 'en',
            voice INTEGER DEFAULT 1
        );
        """)


def add_scan(phone, label, confidence):
    with connect() as c:
        c.execute("INSERT INTO scans (phone, label, confidence, created_at) VALUES (?, ?, ?, ?)",
                  (phone, label, confidence, time.time()))


def total(phone):
    with connect() as c:
        return c.execute("SELECT COUNT(*) FROM scans WHERE phone = ?", (phone,)).fetchone()[0]


def history(phone, limit=200):
    with connect() as c:
        return c.execute("SELECT * FROM scans WHERE phone = ? ORDER BY created_at DESC LIMIT ?",
                         (phone, limit)).fetchall()


def counts(phone):
    with connect() as c:
        rows = c.execute("SELECT label, COUNT(*) AS n FROM scans WHERE phone = ? GROUP BY label",
                         (phone,)).fetchall()
        return {r["label"]: r["n"] for r in rows}


def clear_scans(phone):
    with connect() as c:
        c.execute("DELETE FROM scans WHERE phone = ?", (phone,))


def add_reminder(phone, message, remind_at):
    with connect() as c:
        c.execute("INSERT INTO reminders (phone, message, remind_at) VALUES (?, ?, ?)",
                  (phone, message, remind_at))


def reminders(phone):
    with connect() as c:
        return c.execute("SELECT * FROM reminders WHERE phone = ? ORDER BY remind_at",
                         (phone,)).fetchall()


def due_reminders(now):
    with connect() as c:
        return c.execute("SELECT * FROM reminders WHERE sent = 0 AND remind_at <= ?",
                         (now,)).fetchall()


def mark_sent(reminder_id):
    with connect() as c:
        c.execute("UPDATE reminders SET sent = 1 WHERE id = ?", (reminder_id,))


def get_pref(phone):
    with connect() as c:
        row = c.execute("SELECT lang, voice FROM prefs WHERE phone = ?", (phone,)).fetchone()
    if row is None:
        return {"lang": "en", "voice": 1}
    return {"lang": row["lang"], "voice": row["voice"]}


def set_pref(phone, lang=None, voice=None):
    current = get_pref(phone)
    if lang is not None:
        current["lang"] = lang
    if voice is not None:
        current["voice"] = int(voice)
    with connect() as c:
        c.execute("""INSERT INTO prefs (phone, lang, voice) VALUES (?, ?, ?)
                     ON CONFLICT(phone) DO UPDATE SET lang = excluded.lang, voice = excluded.voice""",
                  (phone, current["lang"], current["voice"]))
