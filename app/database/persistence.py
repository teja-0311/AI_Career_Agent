"""SQLite persistence service for users, sessions, preferences, conversation turns, and saved jobs."""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from app.config import settings


class PersistenceService:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or settings.sqlite_db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    target_roles TEXT,
                    preferred_locations TEXT,
                    preferred_work_mode TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS saved_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, job_id),
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(thread_id) REFERENCES threads(thread_id),
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );
            """)
            conn.commit()

    def ensure_user_and_thread(self, user_id: str, thread_id: str):
        """Register user and bind thread enforcing ownership."""
        with self._get_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
            cur = conn.execute("SELECT user_id FROM threads WHERE thread_id = ?", (thread_id,))
            row = cur.fetchone()
            if row:
                if row["user_id"] != user_id:
                    raise PermissionError(f"Security Alert: Thread '{thread_id}' does not belong to user '{user_id}'.")
            else:
                conn.execute("INSERT INTO threads (thread_id, user_id) VALUES (?, ?)", (thread_id, user_id))
            conn.commit()

    def update_preferences(
        self,
        user_id: str,
        target_roles: Optional[List[str]] = None,
        preferred_locations: Optional[List[str]] = None,
        preferred_work_mode: Optional[str] = None,
    ):
        """Update persistent user preferences."""
        current = self.get_preferences(user_id) or {}
        new_roles = target_roles if target_roles is not None else current.get("target_roles", [])
        new_locs = preferred_locations if preferred_locations is not None else current.get("preferred_locations", [])
        new_wm = preferred_work_mode if preferred_work_mode is not None else current.get("preferred_work_mode")

        with self._get_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
            conn.execute("""
                INSERT INTO user_preferences (user_id, target_roles, preferred_locations, preferred_work_mode, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    target_roles=excluded.target_roles,
                    preferred_locations=excluded.preferred_locations,
                    preferred_work_mode=excluded.preferred_work_mode,
                    updated_at=CURRENT_TIMESTAMP;
            """, (user_id, json.dumps(new_roles), json.dumps(new_locs), new_wm))
            conn.commit()

    def get_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "user_id": row["user_id"],
                "target_roles": json.loads(row["target_roles"] or "[]"),
                "preferred_locations": json.loads(row["preferred_locations"] or "[]"),
                "preferred_work_mode": row["preferred_work_mode"],
                "updated_at": row["updated_at"],
            }

    def save_job_for_user(self, user_id: str, job_id: str) -> bool:
        """Cumulative saved job addition."""
        with self._get_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
            try:
                conn.execute("INSERT INTO saved_jobs (user_id, job_id) VALUES (?, ?)", (user_id, job_id))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False  # Already saved

    def get_saved_jobs(self, user_id: str) -> List[str]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT job_id FROM saved_jobs WHERE user_id = ? ORDER BY id DESC", (user_id,))
            return [r["job_id"] for r in cur.fetchall()]

    def record_message(self, thread_id: str, user_id: str, role: str, content: str):
        self.ensure_user_and_thread(user_id, thread_id)
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO conversation_messages (thread_id, user_id, role, content) VALUES (?, ?, ?, ?)",
                (thread_id, user_id, role, content)
            )
            conn.commit()

    def get_conversation_history(self, thread_id: str, user_id: str, limit: int = 20) -> List[Dict[str, str]]:
        self.ensure_user_and_thread(user_id, thread_id)
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT role, content FROM conversation_messages WHERE thread_id = ? AND user_id = ? ORDER BY id DESC LIMIT ?",
                (thread_id, user_id, limit)
            )
            rows = cur.fetchall()
            return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def clear_thread(self, thread_id: str, user_id: str):
        self.ensure_user_and_thread(user_id, thread_id)
        with self._get_connection() as conn:
            conn.execute("DELETE FROM conversation_messages WHERE thread_id = ? AND user_id = ?", (thread_id, user_id))
            conn.commit()
