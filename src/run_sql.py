# -*- coding: utf-8 -*-
"""
Загружает синтетические CSV в базу SQLite (build/analytics.db) и выполняет
именованные запросы из sql/analysis_queries.sql, печатая результаты и
сохраняя их в reports/sql_results.md.

Запуск:
    python3 src/run_sql.py
"""

from pathlib import Path
import re
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
REF = ROOT / "data" / "reference"
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)
DB_PATH = BUILD / "analytics.db"


def build_db():
    """Создаёт свежую базу и загружает в неё все таблицы."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)

    tickets = pd.read_csv(RAW / "tickets.csv")
    # булевы поля -> целые 1/0 для переносимых SQL-условий
    for col in ["is_critical_for_flights", "sla_response_met",
                "sla_resolution_met", "sla_breached", "reopened"]:
        tickets[col] = tickets[col].astype("Int64").astype(int)
    tickets.to_sql("tickets", conn, index=False)

    flights = pd.read_csv(RAW / "flights_daily.csv")
    flights.to_sql("flights_daily", conn, index=False)
    pd.read_csv(REF / "sla_policy.csv").to_sql("sla_policy", conn, index=False)
    pd.read_csv(REF / "systems.csv").to_sql("systems", conn, index=False)
    pd.read_csv(REF / "agents.csv").to_sql("agents", conn, index=False)

    # индексы для наглядности (демонстрация оптимизации)
    cur = conn.cursor()
    cur.execute("CREATE INDEX idx_tickets_priority ON tickets(priority)")
    cur.execute("CREATE INDEX idx_tickets_system   ON tickets(system)")
    cur.execute("CREATE INDEX idx_tickets_created  ON tickets(created_at)")
    conn.commit()
    return conn


def parse_named_queries(sql_text):
    """Разбивает .sql на именованные блоки по маркеру -- @name: ..."""
    blocks = []
    name, buff = None, []
    for line in sql_text.splitlines():
        m = re.match(r"^--\s*@name:\s*(.+)$", line.strip())
        if m:
            if name and any(s.strip() for s in buff):
                blocks.append((name, "\n".join(buff).strip()))
            name, buff = m.group(1).strip(), []
        elif name is not None:
            buff.append(line)
    if name and any(s.strip() for s in buff):
        blocks.append((name, "\n".join(buff).strip()))
    # убираем ведущие строки-комментарии внутри блока
    cleaned = []
    for n, q in blocks:
        q = "\n".join(l for l in q.splitlines()
                      if not l.strip().startswith("--")).strip()
        if q:
            cleaned.append((n, q))
    return cleaned


def main():
    conn = build_db()
    sql_text = (ROOT / "sql" / "analysis_queries.sql").read_text(encoding="utf-8")
    queries = parse_named_queries(sql_text)

    md_lines = ["# Результаты SQL-запросов\n",
                "Автоматически сгенерировано `src/run_sql.py` "
                "(база `build/analytics.db`, SQLite).\n"]

    print(f"Загружено таблиц в {DB_PATH.name}. Запросов: {len(queries)}\n")
    for name, q in queries:
        df = pd.read_sql_query(q, conn)
        head = df.head(15)
        print("=" * 70)
        print(name)
        print("-" * 70)
        print(head.to_string(index=False))
        print()
        md_lines.append(f"\n## {name}\n")
        md_lines.append("```\n" + head.to_string(index=False) + "\n```\n")

    (ROOT / "reports" / "sql_results.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("Результаты сохранены в reports/sql_results.md")
    conn.close()


if __name__ == "__main__":
    main()
