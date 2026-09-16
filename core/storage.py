"""数据持久化层。

基于标准库 ``sqlite3``，通过 ``asyncio.to_thread`` 把同步 IO 调度到线程池，
避免阻塞 AstrBot 事件循环。设计要点：

- **零第三方依赖**：只用标准库，不引入 ``aiosqlite``，少一个安装失败点。
- **并发安全**：对外方法均为 ``async``，内部用 ``asyncio.Lock`` 串行化。
- **落盘位置**：默认 ``<data>/plugin_data/astrbot_plugin_group_distiller/distiller.db``，
  遵循 AstrBot 规范（数据放 ``data`` 目录）。

导入不到 AstrBot 时回退标准库 ``logging``，可被单元测试直接导入。
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

try:  # 允许在未安装 AstrBot 的环境（如单元测试）下导入
    from astrbot.api import logger
except ImportError:  # pragma: no cover - 仅在无 AstrBot 环境触发
    import logging

    logger = logging.getLogger("astrbot_plugin_group_distiller")

PLUGIN_NAME = "astrbot_plugin_group_distiller"
DB_FILENAME = "distiller.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id      TEXT UNIQUE,
    group_id        TEXT NOT NULL,
    speaker_qq      TEXT NOT NULL,
    speaker_name    TEXT NOT NULL DEFAULT '',
    content         TEXT NOT NULL,
    raw_type        TEXT NOT NULL DEFAULT 'text',
    timestamp       INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL DEFAULT 0,
    is_context      INTEGER NOT NULL DEFAULT 0,
    context_for_qq  TEXT NOT NULL DEFAULT '',
    distilled       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_msg_target   ON messages(group_id, speaker_qq);
CREATE INDEX IF NOT EXISTS idx_msg_ctx      ON messages(group_id, context_for_qq);
CREATE INDEX IF NOT EXISTS idx_msg_distilled ON messages(distilled);

CREATE TABLE IF NOT EXISTS persona (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id        TEXT NOT NULL,
    target_qq       TEXT NOT NULL,
    snapshot        TEXT NOT NULL,
    distill_round   INTEGER NOT NULL DEFAULT 0,
    last_distill_at INTEGER NOT NULL DEFAULT 0,
    updated_at      INTEGER NOT NULL DEFAULT 0,
    UNIQUE(group_id, target_qq)
);

CREATE TABLE IF NOT EXISTS corrections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id   TEXT NOT NULL,
    target_qq  TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


@dataclass
class MessageRecord:
    """一条待入库的群聊消息。

    Attributes:
        message_id: 平台消息 ID，用于去重。
        group_id: 所在群号。
        speaker_qq: 发言人 QQ。
        speaker_name: 发言人昵称。
        content: 消息文本（非文本消息已被转成 ``[图片]`` 之类的占位符）。
        raw_type: 原始消息类型标记，保留给后续扩展。
        timestamp: 消息时间戳（秒）。
        created_at: 入库时间戳（秒）。
        is_context: True 表示这是「目标消息的相邻上下文」，不计入目标语料统计。
        context_for_qq: 该上下文属于哪个目标（仅 ``is_context=True`` 时有意义）。

            多目标场景下这是**必需**的：同一个群里若有两位蒸馏目标，
            上下文行必须归属到具体目标，否则 A 目标蒸馏时会混进 B 目标旁边
            的闲聊，污染人格判断。
    """

    message_id: str
    group_id: str
    speaker_qq: str
    speaker_name: str = ""
    content: str = ""
    raw_type: str = "text"
    timestamp: int = 0
    created_at: int = 0
    is_context: bool = False
    context_for_qq: str = ""


# 第三级兜底（写入插件目录内）的告警只触发一次，避免每次调用都刷屏
_FALLBACK_WARNED = False

# 回退到插件目录内时写入的说明文件名
DATA_MARKER_FILENAME = "README_DATA_HERE.txt"


def _warn_plugin_local_fallback(plugin_dir: Path) -> None:
    """第三级兜底告警：数据落在插件目录内，插件更新/重装会丢数据。

    仅在本进程内第一次触发时打印 warning，并落一个说明文件作为肉眼可见的提示。
    """
    global _FALLBACK_WARNED
    if not _FALLBACK_WARNED:
        _FALLBACK_WARNED = True
        logger.warning(
            "[%s] 未能定位 AstrBot 数据目录，数据目录已回退到插件目录内：%s；"
            "插件更新/重装会导致数据丢失，请检查 AstrBot 版本兼容性。",
            PLUGIN_NAME,
            plugin_dir,
        )
    try:
        marker = plugin_dir / DATA_MARKER_FILENAME
        if not marker.exists():
            marker.write_text(
                "该目录位于插件目录内，插件更新时会被覆盖，请勿在此长期保存数据。\n",
                encoding="utf-8",
            )
    except OSError as exc:  # pragma: no cover - 磁盘异常兜底
        logger.error("[%s] 写入数据目录说明文件失败: %s", PLUGIN_NAME, exc)


def resolve_plugin_data_dir() -> Path:
    """解析本插件的持久化数据目录，并确保其存在。

    优先使用 AstrBot 官方提供的 ``get_astrbot_data_path``；取不到时依次兜底
    到插件旁可能存在的 ``<AstrBot>/data`` 目录，最后退回插件目录下的
    ``data_local``（此级会显式告警，因为插件更新会覆盖该目录）。

    Returns:
        存在且可写的插件数据目录 ``Path``。
    """
    base: Path
    plugin_local_fallback = False
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path

        base = Path(get_astrbot_data_path())
    except Exception:  # pragma: no cover - 依赖 AstrBot 运行时
        # 兜底 1：假定插件位于 <AstrBot>/data/plugins/<plugin>/core/storage.py
        try:
            guess = Path(__file__).resolve().parents[3] / "data"
        except IndexError:
            guess = Path(__file__).resolve().parent / "data"
        if guess.exists():
            base = guess
        else:
            # 兜底 2：退回插件目录内 —— 有数据丢失风险，走显式告警
            base = Path(__file__).resolve().parents[1] / "data_local"
            plugin_local_fallback = True

    plugin_dir = base / "plugin_data" / PLUGIN_NAME
    plugin_dir.mkdir(parents=True, exist_ok=True)

    if plugin_local_fallback:
        _warn_plugin_local_fallback(plugin_dir)
    return plugin_dir


class Storage:
    """基于 sqlite3 的异步存储层。"""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """构造存储对象（不做任何 IO）。

        Args:
            db_path: 显式指定数据库文件路径；为 None 时在 :meth:`init` 阶段
                解析到 AstrBot 的 data 目录。
        """
        self._db_path: Optional[Path] = Path(db_path) if db_path else None
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    async def init(self) -> None:
        """初始化数据库：解析路径、建表、建立连接。"""
        if self._db_path is None:
            self._db_path = resolve_plugin_data_dir() / DB_FILENAME
        else:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run(self._init_sync)

    def _init_sync(self) -> None:
        assert self._db_path is not None
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        # 先补列再建索引：老库若缺 context_for_qq，_SCHEMA 里的建索引语句会直接报错
        self._migrate_sync(conn)
        conn.executescript(_SCHEMA)
        conn.commit()
        self._conn = conn

    @staticmethod
    def _migrate_sync(conn: sqlite3.Connection) -> None:
        """表结构迁移（幂等）。

        目前只有一条：v0.1.0 → v0.2.0 给 ``messages`` 补 ``context_for_qq`` 列。
        老库里已有的上下文行该列为空串，取语料时会被当成「属于该群任意目标」，
        退化成单目标时的行为，不会丢数据。
        """
        try:
            rows = conn.execute("PRAGMA table_info(messages)").fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - 极端异常
            logger.error("[%s] 读取表结构失败: %s", PLUGIN_NAME, exc)
            return
        if not rows:
            return  # 全新库，_SCHEMA 会直接建出完整结构
        cols = {str(row["name"]) for row in rows}
        if "context_for_qq" not in cols:
            try:
                conn.execute(
                    "ALTER TABLE messages "
                    "ADD COLUMN context_for_qq TEXT NOT NULL DEFAULT ''"
                )
                conn.commit()
                logger.info(
                    "[%s] 数据库已升级：messages 表新增 context_for_qq 列（v0.2.0）。",
                    PLUGIN_NAME,
                )
            except sqlite3.Error as exc:  # pragma: no cover - 磁盘/权限异常
                logger.error("[%s] 升级表结构失败: %s", PLUGIN_NAME, exc)

    async def close(self) -> None:
        """提交并关闭连接（幂等）。"""
        await self._run(self._close_sync)

    def _close_sync(self) -> None:
        conn = self._conn
        if conn is not None:
            try:
                conn.commit()
            except sqlite3.Error as exc:  # pragma: no cover
                logger.error("[%s] 关闭前提交失败: %s", PLUGIN_NAME, exc)
            finally:
                try:
                    conn.close()
                except sqlite3.Error:  # pragma: no cover
                    pass
                self._conn = None

    async def _run(self, func: Callable[..., Any], *args: Any) -> Any:
        """在线程池中执行同步 DB 操作，并用锁串行化。"""
        async with self._lock:
            return await asyncio.to_thread(func, *args)

    # ------------------------------------------------------------------ #
    # 语料写入 / 查询
    # ------------------------------------------------------------------ #

    async def insert_message(self, record: MessageRecord) -> bool:
        """写入一条语料，``message_id`` 去重。

        Returns:
            True 表示新插入，False 表示重复或写入失败。
        """
        return await self._run(self._insert_message_sync, record)

    def _insert_message_sync(self, record: MessageRecord) -> bool:
        if self._conn is None:
            return False
        try:
            cur = self._conn.execute(
                """
                INSERT OR IGNORE INTO messages
                    (message_id, group_id, speaker_qq, speaker_name, content,
                     raw_type, timestamp, created_at, is_context, context_for_qq)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.message_id,
                    record.group_id,
                    record.speaker_qq,
                    record.speaker_name,
                    record.content,
                    record.raw_type,
                    int(record.timestamp or 0),
                    int(record.created_at or time.time()),
                    1 if record.is_context else 0,
                    record.context_for_qq if record.is_context else "",
                ),
            )
            self._conn.commit()
            return cur.rowcount > 0
        except sqlite3.Error as exc:
            logger.error("[%s] 写入语料失败: %s", PLUGIN_NAME, exc)
            return False

    async def count_messages(self, group_id: str, qq: str) -> int:
        """统计目标对象的语料总条数（不含上下文字）。"""
        return await self._run(self._count_messages_sync, group_id, qq)

    def _count_messages_sync(self, group_id: str, qq: str) -> int:
        if self._conn is None:
            return 0
        try:
            cur = self._conn.execute(
                """
                SELECT COUNT(*) AS c FROM messages
                WHERE group_id = ? AND speaker_qq = ? AND is_context = 0
                """,
                (group_id, qq),
            )
            return int(cur.fetchone()["c"])
        except sqlite3.Error as exc:
            logger.error("[%s] 统计语料失败: %s", PLUGIN_NAME, exc)
            return 0

    async def count_undistilled(self, group_id: str, qq: str) -> int:
        """统计尚未蒸馏的目标语料条数（用于进度条/自动触发）。"""
        return await self._run(self._count_undistilled_sync, group_id, qq)

    def _count_undistilled_sync(self, group_id: str, qq: str) -> int:
        if self._conn is None:
            return 0
        try:
            cur = self._conn.execute(
                """
                SELECT COUNT(*) AS c FROM messages
                WHERE group_id = ? AND speaker_qq = ? AND is_context = 0
                      AND distilled = 0
                """,
                (group_id, qq),
            )
            return int(cur.fetchone()["c"])
        except sqlite3.Error as exc:
            logger.error("[%s] 统计未蒸馏语料失败: %s", PLUGIN_NAME, exc)
            return 0

    async def fetch_undistilled(
        self, group_id: str, qq: str, limit: int
    ) -> list[dict[str, Any]]:
        """取最新 ``limit`` 条未蒸馏语料（含目标消息与其相邻上下文）。"""
        return await self._run(self._fetch_undistilled_sync, group_id, qq, limit)

    def _fetch_undistilled_sync(
        self, group_id: str, qq: str, limit: int
    ) -> list[dict[str, Any]]:
        if self._conn is None:
            return []
        try:
            cur = self._conn.execute(
                """
                SELECT id, message_id, speaker_qq, speaker_name, content,
                       timestamp, is_context, context_for_qq
                FROM messages
                WHERE group_id = ? AND distilled = 0
                      AND (
                            speaker_qq = ?
                            OR (
                                is_context = 1
                                AND (context_for_qq = '' OR context_for_qq = ?)
                            )
                          )
                ORDER BY timestamp ASC, id ASC
                LIMIT ?
                """,
                (group_id, qq, qq, max(1, int(limit))),
            )
            return [dict(row) for row in cur.fetchall()]
        except sqlite3.Error as exc:
            logger.error("[%s] 读取未蒸馏语料失败: %s", PLUGIN_NAME, exc)
            return []

    async def mark_distilled(self, ids: list[int]) -> int:
        """把给定 id 的语料标记为已蒸馏。"""
        if not ids:
            return 0
        return await self._run(self._mark_distilled_sync, ids)

    def _mark_distilled_sync(self, ids: list[int]) -> int:
        if self._conn is None:
            return 0
        try:
            self._conn.executemany(
                "UPDATE messages SET distilled = 1 WHERE id = ?",
                [(int(i),) for i in ids],
            )
            self._conn.commit()
            return len(ids)
        except sqlite3.Error as exc:
            logger.error("[%s] 标记已蒸馏失败: %s", PLUGIN_NAME, exc)
            return 0

    async def get_time_span(self, group_id: str, qq: str) -> tuple[int, int]:
        """返回目标语料的时间跨度 (最早时间戳, 最晚时间戳)，无数据时返回 (0, 0)。"""
        return await self._run(self._get_time_span_sync, group_id, qq)

    def _get_time_span_sync(self, group_id: str, qq: str) -> tuple[int, int]:
        if self._conn is None:
            return 0, 0
        try:
            cur = self._conn.execute(
                """
                SELECT MIN(timestamp) AS lo, MAX(timestamp) AS hi FROM messages
                WHERE group_id = ? AND speaker_qq = ? AND is_context = 0
                """,
                (group_id, qq),
            )
            row = cur.fetchone()
            return int(row["lo"] or 0), int(row["hi"] or 0)
        except sqlite3.Error as exc:
            logger.error("[%s] 读取时间跨度失败: %s", PLUGIN_NAME, exc)
            return 0, 0

    async def clear_messages(self, group_id: str, qq: str) -> int:
        """清空某目标在该群的全部语料（含上下文），返回删除条数。"""
        return await self._run(self._clear_messages_sync, group_id, qq)

    def _clear_messages_sync(self, group_id: str, qq: str) -> int:
        if self._conn is None:
            return 0
        try:
            cur = self._conn.execute(
                """
                DELETE FROM messages
                WHERE group_id = ?
                      AND (
                            speaker_qq = ?
                            OR (
                                is_context = 1
                                AND (context_for_qq = '' OR context_for_qq = ?)
                            )
                          )
                """,
                (group_id, qq, qq),
            )
            self._conn.commit()
            return cur.rowcount
        except sqlite3.Error as exc:
            logger.error("[%s] 清空语料失败: %s", PLUGIN_NAME, exc)
            return 0

    # ------------------------------------------------------------------ #
    # Persona
    # ------------------------------------------------------------------ #

    async def get_persona(self, group_id: str, qq: str) -> Optional[dict[str, Any]]:
        """读取 Persona 快照（已解析为 dict），不存在返回 None。"""
        return await self._run(self._get_persona_sync, group_id, qq)

    def _get_persona_sync(self, group_id: str, qq: str) -> Optional[dict[str, Any]]:
        if self._conn is None:
            return None
        try:
            cur = self._conn.execute(
                "SELECT snapshot FROM persona WHERE group_id = ? AND target_qq = ?",
                (group_id, qq),
            )
            row = cur.fetchone()
            if row is None:
                return None
            import json

            return json.loads(row["snapshot"])
        except (sqlite3.Error, ValueError) as exc:
            logger.error("[%s] 读取 Persona 失败: %s", PLUGIN_NAME, exc)
            return None

    async def save_persona(self, group_id: str, qq: str, snapshot: dict[str, Any]) -> bool:
        """以 UPSERT 方式保存 Persona 快照。"""
        return await self._run(self._save_persona_sync, group_id, qq, snapshot)

    def _save_persona_sync(self, group_id: str, qq: str, snapshot: dict[str, Any]) -> bool:
        if self._conn is None:
            return False
        import json

        try:
            meta = snapshot.get("meta") if isinstance(snapshot.get("meta"), dict) else {}
            round_no = int(meta.get("distill_round", 0) or 0)
            last_at = int(meta.get("last_distill_at", 0) or 0)
            payload = json.dumps(snapshot, ensure_ascii=False)
            self._conn.execute(
                """
                INSERT INTO persona
                    (group_id, target_qq, snapshot, distill_round, last_distill_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(group_id, target_qq) DO UPDATE SET
                    snapshot = excluded.snapshot,
                    distill_round = excluded.distill_round,
                    last_distill_at = excluded.last_distill_at,
                    updated_at = excluded.updated_at
                """,
                (group_id, qq, payload, round_no, last_at, int(time.time())),
            )
            self._conn.commit()
            return True
        except (sqlite3.Error, TypeError, ValueError) as exc:
            logger.error("[%s] 保存 Persona 失败: %s", PLUGIN_NAME, exc)
            return False

    async def delete_persona(self, group_id: str, qq: str) -> bool:
        """删除某目标的 Persona 快照及其纠正层（用于 ``/zl del <QQ> purge``）。"""
        return await self._run(self._delete_persona_sync, group_id, qq)

    def _delete_persona_sync(self, group_id: str, qq: str) -> bool:
        if self._conn is None:
            return False
        try:
            self._conn.execute(
                "DELETE FROM persona WHERE group_id = ? AND target_qq = ?",
                (group_id, qq),
            )
            self._conn.execute(
                "DELETE FROM corrections WHERE group_id = ? AND target_qq = ?",
                (group_id, qq),
            )
            self._conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("[%s] 删除 Persona 失败: %s", PLUGIN_NAME, exc)
            return False

    # ------------------------------------------------------------------ #
    # 纠正层
    # ------------------------------------------------------------------ #

    async def add_correction(self, group_id: str, qq: str, content: str) -> bool:
        """追加一条人工纠正。"""
        return await self._run(self._add_correction_sync, group_id, qq, content)

    def _add_correction_sync(self, group_id: str, qq: str, content: str) -> bool:
        if self._conn is None:
            return False
        try:
            self._conn.execute(
                """
                INSERT INTO corrections (group_id, target_qq, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (group_id, qq, content, int(time.time())),
            )
            self._conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("[%s] 追加纠正失败: %s", PLUGIN_NAME, exc)
            return False

    async def get_corrections(self, group_id: str, qq: str) -> list[str]:
        """读取某目标的全部人工纠正（按时间升序）。"""
        return await self._run(self._get_corrections_sync, group_id, qq)

    def _get_corrections_sync(self, group_id: str, qq: str) -> list[str]:
        if self._conn is None:
            return []
        try:
            cur = self._conn.execute(
                """
                SELECT content FROM corrections
                WHERE group_id = ? AND target_qq = ?
                ORDER BY id ASC
                """,
                (group_id, qq),
            )
            return [str(row["content"]) for row in cur.fetchall()]
        except sqlite3.Error as exc:
            logger.error("[%s] 读取纠正失败: %s", PLUGIN_NAME, exc)
            return []

    # ------------------------------------------------------------------ #
    # 运行时状态（键值对）
    # ------------------------------------------------------------------ #

    async def get_state(self, key: str) -> Optional[str]:
        """读取运行时状态值。"""
        return await self._run(self._get_state_sync, key)

    def _get_state_sync(self, key: str) -> Optional[str]:
        if self._conn is None:
            return None
        try:
            cur = self._conn.execute("SELECT value FROM state WHERE key = ?", (key,))
            row = cur.fetchone()
            return None if row is None else str(row["value"])
        except sqlite3.Error as exc:
            logger.error("[%s] 读取状态失败: %s", PLUGIN_NAME, exc)
            return None

    async def set_state(self, key: str, value: str) -> bool:
        """写入运行时状态值（UPSERT）。"""
        return await self._run(self._set_state_sync, key, value)

    def _set_state_sync(self, key: str, value: str) -> bool:
        if self._conn is None:
            return False
        try:
            self._conn.execute(
                """
                INSERT INTO state (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )
            self._conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("[%s] 写入状态失败: %s", PLUGIN_NAME, exc)
            return False
