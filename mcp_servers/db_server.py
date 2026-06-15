"""数据库 MCP 服务器 — 提供 SQLite 只读查询工具"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WYF Database Server", json_response=True)

DEFAULT_DB_PATH = str(Path(__file__).parent.parent / "data" / "app.db")


def _get_connection(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


@mcp.tool()
def query_database(sql: str, db_path: str = DEFAULT_DB_PATH) -> str:
    """执行只读 SQL 查询

    Args:
        sql: SELECT 查询语句
        db_path: 数据库文件路径, 默认为 data/app.db
    """
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT") and not sql_stripped.startswith("WITH"):
        return "错误: 仅支持 SELECT 查询（只读模式）"

    try:
        conn = _get_connection(db_path)
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "查询结果为空"

        lines = [", ".join(columns)]
        for row in rows:
            lines.append(", ".join(str(v) for v in row))
        return "\n".join(lines)
    except Exception as e:
        return f"查询错误: {e}"


@mcp.tool()
def list_tables(db_path: str = DEFAULT_DB_PATH) -> str:
    """列出数据库中的所有表

    Args:
        db_path: 数据库文件路径, 默认为 data/app.db
    """
    try:
        conn = _get_connection(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not tables:
            return "数据库中没有表"
        return "表列表:\n" + "\n".join(f"  - {t}" for t in tables)
    except Exception as e:
        return f"错误: {e}"


@mcp.tool()
def get_schema(table: str, db_path: str = DEFAULT_DB_PATH) -> str:
    """获取指定表的结构信息

    Args:
        table: 表名
        db_path: 数据库文件路径, 默认为 data/app.db
    """
    try:
        conn = _get_connection(db_path)
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        conn.close()

        if not columns:
            return f"表 '{table}' 不存在或没有列信息"

        lines = [f"表 '{table}' 结构:"]
        for col in columns:
            _cid, name, col_type, notnull, default_val, pk = col
            attrs = []
            if pk:
                attrs.append("主键")
            if notnull:
                attrs.append("NOT NULL")
            if default_val is not None:
                attrs.append(f"默认={default_val}")
            attr_str = f" ({', '.join(attrs)})" if attrs else ""
            lines.append(f"  {name}: {col_type}{attr_str}")
        return "\n".join(lines)
    except Exception as e:
        return f"错误: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
