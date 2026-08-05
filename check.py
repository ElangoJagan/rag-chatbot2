import sqlite3
conn = sqlite3.connect('data/query_logs.db')

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", tables)

rows = conn.execute("SELECT * FROM query_logs").fetchall()
print("Rows:", rows)