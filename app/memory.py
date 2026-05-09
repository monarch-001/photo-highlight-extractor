import aiosqlite
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class MemoryManager:
    def __init__(self, db_path: str = "data/photo_memory.db"):
        self.db_path = db_path

    async def initialize(self):
        async with aiosqlite.connect(self.db_path) as db:
            # Table for image metadata and analysis results
            await db.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    path TEXT PRIMARY KEY,
                    filename TEXT,
                    blur_score REAL,
                    exposure_score REAL,
                    face_count INTEGER,
                    emotions TEXT,
                    scene_label TEXT,
                    is_duplicate INTEGER DEFAULT 0,
                    highlight_score REAL DEFAULT 0.0,
                    is_highlight INTEGER DEFAULT 0,
                    cluster_id TEXT,
                    explanation TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Table for session/user preferences
            await db.execute("""
                CREATE TABLE IF NOT EXISTS session_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            await db.commit()

    async def save_image_metadata(self, metadata: Dict[str, Any]):
        async with aiosqlite.connect(self.db_path) as db:
            keys = list(metadata.keys())
            values = list(metadata.values())
            placeholders = ", ".join(["?"] * len(keys))
            sql = f"INSERT OR REPLACE INTO images ({', '.join(keys)}) VALUES ({placeholders})"
            await db.execute(sql, values)
            await db.commit()

    async def get_image_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM images WHERE path = ?", (path,)) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def get_all_highlights(self) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM images WHERE is_highlight = 1 ORDER BY highlight_score DESC") as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def set_preference(self, key: str, value: Any):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("INSERT OR REPLACE INTO session_state (key, value) VALUES (?, ?)", (key, json.dumps(value)))
            await db.commit()

    async def get_preference(self, key: str) -> Optional[Any]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT value FROM session_state WHERE key = ?", (key,)) as cursor:
                row = await cursor.fetchone()
                return json.loads(row[0]) if row else None
