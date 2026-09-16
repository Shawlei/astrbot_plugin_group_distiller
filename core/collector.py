"""语料采集器。

负责静默监听群消息、按目标过滤、写入内存缓冲区并异步落库。

关键设计：

- **多目标**：同时支持「一个群多个群友」「多个群各若干群友」。一条消息只
  在「群号匹配 且 发送者就是某个目标」时才计为该目标的语料。
- **上下文归属**：上下文行带 ``context_for_qq``，明确属于哪个目标。否则同一群里
  两个目标的旁听消息会互相污染。
- **绝不阻塞监听器**：``handle_event`` 只做轻量过滤与入 buffer，
  真正的写库放在后台 flush 循环或超阈值时的一次性任务里。
- **计数器内存化**：``overview`` 维护每个目标的 total / undistilled 计数，
  避免每次渲染进度面板或检查自动蒸馏阈值都全表扫描。

本模块零 AstrBot 强依赖，可被单元测试直接导入。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any, Optional

try:  # 允许在未安装 AstrBot 的环境（如单元测试）下导入
    from astrbot.api import logger
except ImportError:  # pragma: no cover - 仅在无 AstrBot 环境触发
    import logging

    logger = logging.getLogger("astrbot_plugin_group_distiller")

try:
    from .storage import MessageRecord, Storage
    from .targets import TargetSpec, dedupe
except ImportError:  # pragma: no cover - 兼容顶层导入
    from storage import MessageRecord, Storage  # type: ignore
    from targets import TargetSpec, dedupe  # type: ignore

PLUGIN_NAME = "astrbot_plugin_group_distiller"


class RuntimeState:
    """采集/蒸馏的运行时状态（可被指令动态修改，并持久化到 state 表）。

    Attributes:
        enabled: 插件总开关。
        listen_enabled: 静默采集开关。
        targets: 全部蒸馏目标（权威来源）。
        active_key: 当前「选中」的目标键（``群号:QQ号``），面板/档案/导出都以它为准。
        group_id / qq_id / nickname: 当前目标的镜像字段，便于渲染与外部读取；
            当 :attr:`targets` 为空时会退化成「单目标」来源（兼容旧调用方式）。
        total_messages: 当前目标的语料总数（内存缓存）。
        undistilled: 当前目标的未蒸馏语料数（内存缓存）。
        today_new: 今日新增条数（内存缓存）。
        today_date: 今日计数的日期标记，跨天时重置 ``today_new``。
        overview: 每个目标的计数缓存 ``{目标键: {"total": n, "undistilled": m}}``，
            供自动蒸馏阈值判断与目标列表渲染使用。
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        listen_enabled: bool = True,
        group_id: str = "",
        qq_id: str = "",
        nickname: str = "",
        targets: Optional[list[TargetSpec]] = None,
    ) -> None:
        """构造运行时状态（不做任何 IO）。

        兼容两种用法：新代码传 ``targets``；旧代码/测试传 ``group_id`` +
        ``qq_id``，会被自动包装成单个目标。
        """
        self.enabled = enabled
        self.listen_enabled = listen_enabled
        self.targets: list[TargetSpec] = dedupe(list(targets or []))
        self.active_key: str = self.targets[0].key if self.targets else ""
        self.group_id = str(group_id or "")
        self.qq_id = str(qq_id or "")
        self.nickname = str(nickname or "")
        self.total_messages = 0
        self.undistilled = 0
        self.today_new = 0
        self.today_date = ""
        self.overview: dict[str, dict[str, int]] = {}
        # 群号 -> 该群最近一次出现过的 unified_msg_origin。
        # 每日定时总结要在群里播报时用它，避免自己拼 umo 拼错。
        self.last_umo: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # 目标集合
    # ------------------------------------------------------------------ #

    def _legacy_target(self) -> Optional[TargetSpec]:
        """把镜像字段（group_id / qq_id / nickname）拼成一个目标。

        仅用于兼容「只填了单目标」的旧配置与旧调用方式。
        """
        if self.group_id and self.qq_id:
            return TargetSpec(self.group_id, self.qq_id, self.nickname)
        return None

    def active_target(self) -> Optional[TargetSpec]:
        """返回当前选中的目标；没有目标时返回 None。"""
        if self.active_key:
            for spec in self.targets:
                if spec.key == self.active_key:
                    return spec
        if self.targets:
            return self.targets[0]
        # 兼容模式：targets 为空时用镜像字段拼一个单目标
        return self._legacy_target()

    def collect_targets(self) -> list[TargetSpec]:
        """返回采集/蒸馏时应当生效的全部目标。

        ``targets`` 非空时以它为准；为空时退化为镜像字段表示的单个目标。
        """
        if self.targets:
            return list(self.targets)
        spec = self.active_target()
        return [spec] if spec is not None else []

    def has_target(self) -> bool:
        """是否已设定至少一个完整的目标（群 + QQ）。"""
        return bool(self.collect_targets())

    def find_target(self, qq: str) -> Optional[TargetSpec]:
        """按 QQ 号查找目标（跨群也会返回；同名多个时取第一个）。"""
        qq = str(qq or "").strip()
        for spec in self.collect_targets():
            if spec.qq_id == qq:
                return spec
        return None

    def targets_in_group(self, group_id: str) -> list[TargetSpec]:
        """返回某群内需要采集的全部目标。"""
        group_id = str(group_id or "")
        return [t for t in self.collect_targets() if t.group_id == group_id]

    def add_target(self, spec: TargetSpec) -> bool:
        """新增一个目标（已存在则只更新昵称）。返回是否为新增。

        注意：如果当前处于「只有镜像字段的单目标」状态（旧配置/旧调用方式），
        会先把那个目标正式收编进 :attr:`targets`，否则新加一个目标会把老目标
        悄悄挤掉。
        """
        if spec is None:
            return False
        if not self.targets:
            legacy = self._legacy_target()
            if legacy is not None and legacy.key != spec.key:
                self.targets.append(legacy)
        for i, old in enumerate(self.targets):
            if old.key == spec.key:
                if spec.nickname and spec.nickname != old.nickname:
                    self.targets[i] = spec
                return False
        self.targets.append(spec)
        if not self.active_key:
            self.set_active(spec.key)
        return True

    def remove_target(self, qq: str) -> int:
        """删除某 QQ 对应的全部目标，返回删除条数。"""
        qq = str(qq or "").strip()
        before = len(self.targets)
        self.targets = [t for t in self.targets if t.qq_id != qq]
        removed = before - len(self.targets)
        if removed and self.active_key not in {t.key for t in self.targets}:
            self.active_key = self.targets[0].key if self.targets else ""
            self.sync_active_fields()
        return removed

    def clear_targets(self) -> None:
        """清空目标列表。"""
        self.targets = []
        self.active_key = ""
        self.sync_active_fields()

    def set_active(self, key: str) -> bool:
        """把某个目标设为当前选中项（同时刷新镜像字段）。

        Args:
            key: 目标键（``群号:QQ号``），或直接给 QQ 号。

        Returns:
            是否设置成功。
        """
        key = str(key or "").strip()
        target: Optional[TargetSpec] = None
        for spec in self.targets:
            if spec.key == key or spec.qq_id == key:
                target = spec
                break
        if target is None:
            return False
        self.active_key = target.key
        self.sync_active_fields()
        return True

    def sync_active_fields(self) -> None:
        """把当前目标的群号/QQ/昵称镜像到平铺字段（供渲染与外部读取）。

        注意这里**只看** :attr:`targets`，不看镜像字段本身 —— 否则目标被删光后
        镜像字段会把自己「复活」成一个幽灵目标，面板永远清不掉。
        """
        spec: Optional[TargetSpec] = None
        if self.active_key:
            for item in self.targets:
                if item.key == self.active_key:
                    spec = item
                    break
        if spec is None and self.targets:
            spec = self.targets[0]
        self.group_id = spec.group_id if spec else ""
        self.qq_id = spec.qq_id if spec else ""
        self.nickname = spec.nickname if spec else ""

    # ------------------------------------------------------------------ #
    # 计数缓存
    # ------------------------------------------------------------------ #

    def note_inserted(self, group_id: str, qq: str) -> None:
        """记一条新语料入库，同步 overview 与当前目标的计数。"""
        key = f"{group_id}:{qq}"
        bucket = self.overview.get(key)
        if bucket is None:
            bucket = {"total": 0, "undistilled": 0}
            self.overview[key] = bucket
        bucket["total"] = int(bucket.get("total", 0)) + 1
        bucket["undistilled"] = int(bucket.get("undistilled", 0)) + 1

        active = self.active_target()
        if active is not None and active.key == key:
            self.total_messages += 1
            self.undistilled += 1
        self._bump_today()

    def set_counts(self, key: str, total: int, undistilled: int) -> None:
        """用数据库统计结果覆盖某个目标的计数缓存。"""
        self.overview[key] = {
            "total": max(0, int(total)),
            "undistilled": max(0, int(undistilled)),
        }
        active = self.active_target()
        if active is not None and active.key == key:
            self.total_messages = self.overview[key]["total"]
            self.undistilled = self.overview[key]["undistilled"]

    def counts_for(self, spec: TargetSpec) -> dict[str, int]:
        """读取某目标的计数缓存（缺失返回 0）。"""
        bucket = self.overview.get(spec.key) or {}
        return {
            "total": int(bucket.get("total", 0) or 0),
            "undistilled": int(bucket.get("undistilled", 0) or 0),
        }

    def _bump_today(self) -> None:
        """维护"今日新增"计数，跨天自动归零。"""
        today = time.strftime("%Y-%m-%d")
        if self.today_date != today:
            self.today_date = today
            self.today_new = 0
        self.today_new += 1

    # ------------------------------------------------------------------ #
    # 会话来源（每日总结播报用）
    # ------------------------------------------------------------------ #

    def note_umo(self, group_id: str, umo: str) -> None:
        """记录某个群最近出现过的会话来源（``unified_msg_origin``）。"""
        if group_id and umo:
            self.last_umo[str(group_id)] = str(umo)

    def umo_for(self, group_id: str, platform: str = "aiocqhttp") -> str:
        """取某个群的会话来源。

        优先用采集时真实见过的；没见过（比如刚重启）就按 AstrBot 的
        ``platform:message_type:session_id`` 格式兜底拼一个。
        """
        known = self.last_umo.get(str(group_id))
        if known:
            return known
        return f"{platform}:GroupMessage:{group_id}"


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
        # 群号 -> 最近一条"他人消息"，作为下一次目标发言的潜在上文
        self._last_other: dict[str, MessageRecord] = {}
        # 群号 -> 还欠哪个目标一条"下文"（值为目标 QQ 号，空串表示不欠）
        self._pending_context: dict[str, str] = {}
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
                self.state.note_inserted(rec.group_id, rec.speaker_qq)
                inserted += 1
        return inserted

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
        if not (self.state.enabled and self.state.listen_enabled):
            return

        try:
            group_id = str(event.get_group_id() or "")
        except Exception:  # noqa: BLE001
            return
        if not group_id:
            return

        # 只关心"本群里存在蒸馏目标"的群，其它群消息在第一步就被挡掉
        group_targets = self.state.targets_in_group(group_id)
        if not group_targets:
            return

        try:
            self.state.note_umo(group_id, getattr(event, "unified_msg_origin", "") or "")
        except Exception:  # noqa: BLE001 - 记来源失败不影响采集
            pass

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

        target = next((t for t in group_targets if t.qq_id == sender_id), None)
        record_ctx = bool(self.config.get("record_context", True))
        min_len = self._safe_int(self.config.get("min_message_length", 1), 0)
        ignore_cmd = bool(self.config.get("ignore_commands", True))
        passes = self._passes_filter(content, min_len, ignore_cmd)

        if target is not None:
            # 目标发言被过滤（太短 / 是指令）→ 整条丢弃，
            # 且不补录任何上下文，避免产生"没有对应目标发言的孤立上下文行"。
            if not passes:
                return
            # 目标发言确认保留后，才补录"上一条他人消息"作为上文。
            # 必须**复制**而不是复用同一个对象：那条消息可能已经作为另一个
            # 目标的下文进了缓冲区，直接改它会把前一个目标的上下文一起改掉。
            if record_ctx:
                prev = self._last_other.pop(group_id, None)
                if prev is not None:
                    self._buffer.append(self._as_context(prev, target.qq_id))
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
            self._pending_context[group_id] = target.qq_id
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
            # 紧随某个目标发言之后的他人消息 → 记为那个目标的下文
            pending_qq = self._pending_context.get(group_id, "")
            if record_ctx and pending_qq:
                self._buffer.append(self._as_context(ctx_rec, pending_qq))
                self._pending_context[group_id] = ""
            # 缓存为"潜在上文"（保存未归属的基对象，附归属时再复制出来），
            # 等目标下次发言时补记
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
    def _as_context(rec: MessageRecord, qq: str) -> MessageRecord:
        """把一条他人消息复制成"归属于目标 ``qq`` 的上下文行"。

        ``message_id`` 会拼上 ``#ctx<QQ>`` 后缀。原因是 ``messages.message_id``
        有唯一约束，而同一句话完全可能同时是两个目标的上下文（同群多目标时
        它既是 A 的下文、又是 B 的上文）—— 不加后缀的话第二条会被去重掉，
        导致某个目标凭空少了一条语境。
        """
        return replace(
            rec,
            context_for_qq=qq,
            message_id=f"{rec.message_id}#ctx{qq}",
        )

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
