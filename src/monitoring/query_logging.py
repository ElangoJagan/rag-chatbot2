import sqlite3
from datetime import datetime,timezone

DB_PATH = "data/query_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            confidence REAL,
            latency_ms REAL,
            sources TEXT,
            was_gated INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_query(question, answer, confidence, latency_ms, sources, was_gated):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO query_logs
        (timestamp, question, answer, confidence, latency_ms, sources, was_gated)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            question, 
            answer,
            confidence, 
            latency_ms,
            ','.join(sources),
            1 if was_gated else 0
            
        )
    )
    conn.commit()
    conn.close()