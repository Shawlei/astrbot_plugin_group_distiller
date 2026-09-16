"""语料采集器。

负责静默监听群消息、按目标过滤、写入内存缓冲区并异步落库。

关键设计：

- **绝不阻塞监听器**：``handle_event`` 只做轻量过滤与入 buffer，
  真正的写库放在后台 flush 循环或超阈值时的一次性任务里。
- **计数器内存化**：维护 ``total_messages`` / ``undistilled`` / ``today_new``
  等内存计数，避免每次渲染进度面板都全表扫描。
- **上下文捕获**：可选记录目标消息前后各 1 条相邻消息（``is_context=True``），
  帮助 LLM 理解语境，但不计入目标语料统计。

本模块零 AstrBot 强依赖，可被单元测试直接导入。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional

try:  # 允许在未安装 AstrBot 的环境（如单元测试）下导入
    from astrbot.api import logger
except ImportError:  # pragma: no cover - 仅在无 AstrBot 环境触发
    import logging

    logger = logging.getLogger("astrbot_plugin_group_distiller")

try:
    from .storage import MessageRecord, Storage
except ImportError:  # pragma: no cover - 兼容顶层导入
    from storage import MessageRecord, Storage  # type: ignore

PLUGIN_NAME = "astrbot_plugin_group_distiller"


@dataclass
class RuntimeState:
    """采集/蒸馏的运行时状态（可被指令动态修改，并持久化到 state 表）。

    Attributes:
        enabled: 插件总开关。
        listen_enabled: 静默采集开关。
        group_id: 目标群号。
        qq_id: 目标 QQ 号。
        nickname: 目标昵称（用于提示词与展示）。
        total_messages: 目标语料总数（内存缓存）。
        undistilled: 未蒸馏目标语料数（内存缓存）。
        today_new: 今日新增条数（内存缓存）。
        today_date: 今日计数的日期标记，跨天时重置 ``today_new``。
    """

    enabled: bool = True
    listen_enabled: bool = True
    group_id: str = ""
    qq_id: str = ""
    nickname: str = ""
    total_messages: int = 0
    undistilled: int = 0
    today_new: int = 0
    today_date: str = ""

    def has_target(self) -> bool:
        """是否已设定完整的目标（群 + QQ）。"""
        return bool(self.group_id and self.qq_id)


class Collector:
    """静默采集器。"""

    def __init__(
        self,
        storage: Storage,
        config: Any,
        state: RuntimeState,
        plugin_name: str = PLUGIN_NAME,
    ) -> None:
        """构造采集器（不做 IO）。

        Args:
            storage: 存储层实例。
            config: AstrBot 配置对象（dict 语义）。
            state: 共享的运行时状态对象。
            plugin_name: 用于日志前缀。
        """
        self.storage = storage
        self.config = config
        self.state = state
        self.plugin_name = plugin_name

        self._buffer: list[MessageRecord] = []
        self._flush_task: Optional[asyncio.Task[None]] = None
        self._last_other: dict[str, MessageRecord] = {}
        self._pending_context: dict[str, bool] = {}
        self._flush_interval = 5.0
        self._max_buffer = 40
        self._running = False

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """启动后台 flush 循环。"""
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """停止 flush 循环并做最后一次落盘。"""
        self._running = False
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        await self.flush()

    async def _flush_loop(self) -> None:
        """周期性把 buffer 落盘。"""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001 - 后台任务需吞掉所有异常
                logger.error("[%s] flush 循环异常: %s", self.plugin_name, exc)

    async def flush(self) -> int:
        """把当前缓冲区写入数据库，并更新内存计数。

        Returns:
            本次实际新插入的目标语料条数。
        """
        if not self._buffer:
            return 0
        batch = self._buffer
        self._buffer = []
        inserted = 0
        for rec in batch:
            try:
                ok = await self.storage.insert_message(rec)
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] 写入语料异常: %s", self.plugin_name, exc)
                continue
            if ok and not rec.is_context:
                self.state.total_messages += 1
                self.state.undistilled += 1
                self._bump_today()
                inserted += 1
        return inserted

    def _bump_today(self) -> None:
        """维护"今日新增"计数，跨天自动归零。"""
        today = time.strftime("%Y-%m-%d")
        if self.state.today_date != today:
            self.state.today_date = today
            self.state.today_new = 0
        self.state.today_new += 1

    # ------------------------------------------------------------------ #
    # 事件处理
    # ------------------------------------------------------------------ #

    async def handle_event(self, event: Any) -> None:
        """处理一条群消息（静默、非阻塞）。

        该方法不会回复、不会 yield、不会打断机器人正常聊天；任何异常都被
        拦截并记录日志，绝不影响主链路。

        Args:
            event: AstrBot 的 ``AstrMessageEvent``。
        """
        if not (self.state.enabled and self.state.listen_enabled and self.state.has_target()):
            return

        try:
            group_id = str(event.get_group_id() or "")
        except Exception:  # noqa: BLE001
            return
        if not group_id or group_id != self.state.group_id:
            return

        try:
            sender_id = str(event.get_sender_id() or "")
        except Exception:  # noqa: BLE001
            return
        if not sender_id:
            return

        content = self._extract_content(event)
        if not content:
            return

        ts = self._extract_timestamp(event)
        msg_id = self._extract_message_id(event, group_id, sender_id, ts, content)
        sender_name = self._extract_sender_name(event)

        is_target = sender_id == self.state.qq_id
        record_ctx = bool(self.config.get("record_context", True))
        min_len = self._safe_int(self.config.get("min_message_length", 1), 0)
        ignore_cmd = bool(self.config.get("ignore_commands", True))
        passes = self._passes_filter(content, min_len, ignore_cmd)

        if is_target:
            # 目标发言被过滤（太短 / 是指令）→ 整条丢弃，
            # 且不补录任何上下文，避免产生"没有对应目标发言的孤立上下文行"。
            if not passes:
                return
            # 目标发言确认保留后，才补录"上一条他人消息"作为上文
            if record_ctx:
                prev = self._last_other.pop(group_id, None)
                if prev is not None:
                    self._buffer.append(prev)
            self._buffer.append(
                MessageRecord(
                    message_id=msg_id,
                    group_id=group_id,
                    speaker_qq=sender_id,
                    speaker_name=sender_name,
                    content=content,
                    raw_type="text",
                    timestamp=ts,
                    created_at=int(time.time()),
                    is_context=False,
                )
            )
            self._pending_context[group_id] = True
        else:
            ctx_rec = MessageRecord(
                message_id=msg_id,
                group_id=group_id,
                speaker_qq=sender_id,
                speaker_name=sender_name,
                content=content,
                raw_type="text",
                timestamp=ts,
                created_at=int(time.time()),
                is_context=True,
            )
            # 紧随目标发言之后的他人消息 → 记为下文
            if record_ctx and self._pending_context.get(group_id):
                self._buffer.append(ctx_rec)
                self._pending_context[group_id] = False
            # 缓存为"潜在上文"，等目标下次发言时补记
            if passes:
                self._last_other[group_id] = ctx_rec
            else:
                self._last_other.pop(group_id, None)

        # 缓冲区过大时，立即异步落盘（不阻塞当前处理）
        if len(self._buffer) >= self._max_buffer and self._running:
            asyncio.create_task(self.flush())

    # ------------------------------------------------------------------ #
    # 事件字段提取（全部带兜底，异常不外抛）
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_content(event: Any) -> str:
        """提取消息文本，优先用 ``get_message_outline``（非文本转占位符）。"""
        content = ""
        try:
            content = event.get_message_outline() or ""
        except Exception:  # noqa: BLE001
            content = ""
        if not content:
            try:
                content = getattr(event, "message_str", "") or ""
            except Exception:  # noqa: BLE001
                content = ""
        return str(content).strip()

    @staticmethod
    def _extract_sender_name(event: Any) -> str:
        """提取发送者昵称，失败返回空串。"""
        try:
            return str(event.get_sender_name() or "")
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _extract_timestamp(event: Any) -> int:
        """提取消息时间戳，失败回退到当前时间。"""
        obj = getattr(event, "message_obj", None)
        ts = getattr(obj, "timestamp", None)
        try:
            return int(ts) if ts else int(time.time())
        except (TypeError, ValueError):
            return int(time.time())

    @staticmethod
    def _extract_message_id(
        event: Any, group_id: str, sender_id: str, ts: int, content: str
    ) -> str:
        """提取消息 ID 用于去重；缺失时构造稳定回退 ID。"""
        obj = getattr(event, "message_obj", None)
        mid = getattr(obj, "message_id", None)
        if mid:
            return str(mid)
        return f"{group_id}:{sender_id}:{ts}:{abs(hash(content)) % (10 ** 12)}"

    @staticmethod
    def _passes_filter(content: str, min_len: int, ignore_cmd: bool) -> bool:
        """判断消息是否通过采集过滤（长度 / 指令前缀）。"""
        text = content.strip()
        if not text:
            return False
        if ignore_cmd and text.startswith("/"):
            return False
        if min_len > 0 and len(text) < min_len:
            return False
        return True

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        """把任意配置值安全转换为 int。"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
