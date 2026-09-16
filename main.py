"""AstrBot 插件主入口：我要蒸馏群友（astrbot_plugin_group_distiller）。

静默采集指定群友的聊天语料，调用 LLM 蒸馏成 5 层 AI 人格档案，
并通过 ``/zl`` 指令族在群内查看进度与档案。

作者：Shawlei
版本：v0.1.0
仓库：https://github.com/Shawlei/astrbot_plugin_group_distiller
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp

try:  # 兼容包内导入 / 顶层导入两种加载方式
    from .core import progress, prompts
    from .core.collector import Collector, RuntimeState
    from .core.distiller import (
        Distiller,
        compute_completeness,
        count_formed_layers,
        empty_snapshot,
    )
    from .core.storage import Storage, resolve_plugin_data_dir
except ImportError:  # pragma: no cover
    from core import progress, prompts  # type: ignore
    from core.collector import Collector, RuntimeState  # type: ignore
    from core.distiller import (  # type: ignore
        Distiller,
        compute_completeness,
        count_formed_layers,
        empty_snapshot,
    )
    from core.storage import Storage, resolve_plugin_data_dir  # type: ignore

PLUGIN_NAME = "astrbot_plugin_group_distiller"
PLUGIN_DISPLAY = "我要蒸馏群友"

# 可能出现在 message_str 开头的指令前缀（AstrBot 有时会剥离，有时不会）
_PREFIXES = ("/zl", "zl", "／zl", "/蒸馏", "蒸馏", "／蒸馏")


def _parse_command(raw: str) -> tuple[str, str]:
    """解析 ``/zl`` 指令文本为 ``(子命令, 参数)``。

    兼容 AstrBot 是否剥离指令前缀的情况：``/zl set 1 2``、``zl set 1 2``、
    ``set 1 2`` 三种形态都能正确解析为 ``("set", "1 2")``。

    Args:
        raw: ``event.message_str``。

    Returns:
        ``(head, rest)``：``head`` 已转小写，``rest`` 为去首尾空白的剩余参数。
        ``head`` 为空串或 ``panel`` 表示展示默认面板。
    """
    text = (raw or "").strip()
    if not text:
        return "panel", ""

    for prefix in _PREFIXES:
        if text == prefix:
            return "panel", ""
        if text.startswith(prefix + " ") or text.startswith(prefix + "\u3000"):
            text = text[len(prefix) :].strip()
            break

    if not text:
        return "panel", ""

    parts = text.split(maxsplit=1)
    head = parts[0].strip().lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    return head, rest


async def _write_text(path: Path, text: str) -> None:
    """在线程池里写文件，避免阻塞事件循环。"""
    await asyncio.to_thread(path.write_text, text, "utf-8")


@register(PLUGIN_NAME, "Shawlei", "把群友蒸馏成 AI 人格档案", "v0.1.0")
class GroupDistillerPlugin(Star):
    """群友蒸馏插件主类。"""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        self.storage = Storage()
        self.state = RuntimeState()
        self.collector = Collector(self.storage, self.config, self.state, PLUGIN_NAME)
        self.distiller = Distiller(
            self.storage, self.config, self.state, self.context, PLUGIN_NAME
        )

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """插件实例化后自动调用：初始化存储、加载状态、启动采集。"""
        try:
            await self.storage.init()
            await self._load_runtime_state()
            if self.state.has_target():
                self.state.total_messages = await self.storage.count_messages(
                    self.state.group_id, self.state.qq_id
                )
                self.state.undistilled = await self.storage.count_undistilled(
                    self.state.group_id, self.state.qq_id
                )
            await self.collector.start()
            logger.info("[%s] 初始化完成，目标=%s/%s", PLUGIN_NAME, self.state.group_id, self.state.qq_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 初始化失败: %s", PLUGIN_NAME, exc, exc_info=True)

    async def terminate(self) -> None:
        """插件卸载/停用时调用：落盘、取消后台任务、关闭连接。"""
        try:
            await self.collector.stop()
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 停止采集器失败: %s", PLUGIN_NAME, exc)
        try:
            task = getattr(self.distiller, "_task", None)
            if task is not None and not task.done():
                task.cancel()
                # 等后台蒸馏任务真正结束后再关 DB，避免"关库后任务仍在写"的竞态
                await asyncio.gather(task, return_exceptions=True)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 取消蒸馏任务失败: %s", PLUGIN_NAME, exc)
        try:
            await self.storage.close()
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 关闭存储失败: %s", PLUGIN_NAME, exc)

    # ------------------------------------------------------------------ #
    # 静默监听（核心采集入口）
    # ------------------------------------------------------------------ #

    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    async def on_any_message(self, event: AstrMessageEvent) -> None:
        """静默采集每条群消息（绝不 yield、绝不打断主链路）。"""
        try:
            await self.collector.handle_event(event)
            await self.distiller.maybe_auto(event.unified_msg_origin)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 监听处理异常: %s", PLUGIN_NAME, exc)
        return

    # ------------------------------------------------------------------ #
    # /zl 指令族
    # ------------------------------------------------------------------ #

    @filter.command("zl", alias={"蒸馏"})
    async def zl(self, event: AstrMessageEvent):
        """``/zl`` 指令族入口，手动解析 ``event.message_str`` 分发子命令。"""
        if bool(self.config.get("admin_only", True)) and not self._is_admin(event):
            yield event.plain_result("🔒 该指令仅管理员可用。")
            return

        head, rest = _parse_command(getattr(event, "message_str", "") or "")

        try:
            if head in ("", "panel", "菜单", "进度"):
                yield self._reply(event, await self._build_panel())

            elif head == "on":
                self.state.enabled = True
                self.state.listen_enabled = True
                await self._persist_state()
                yield self._reply(event, "✅ 已开启采集。")

            elif head == "off":
                self.state.listen_enabled = False
                await self._persist_state()
                yield self._reply(event, "⏸️ 已关闭采集（已有语料与档案保留）。")

            elif head == "set":
                async for reply in self._handle_set(event, rest):
                    yield reply

            elif head in ("now", "目标"):
                yield self._reply(event, self._render_target())

            elif head in ("distill", "蒸馏", "开始"):
                result = await self.distiller.trigger(event.unified_msg_origin, manual=True)
                yield self._reply(event, result.message)

            elif head in ("profile", "档案", "画像"):
                snapshot = await self._load_snapshot()
                yield self._reply(
                    event, progress.render_profile(snapshot, self.state.nickname, self.state.qq_id)
                )

            elif head in ("export", "导出"):
                async for reply in self._handle_export(event):
                    yield reply

            elif head in ("correct", "纠正"):
                async for reply in self._handle_correct(event, rest):
                    yield reply

            elif head in ("reset", "重置"):
                async for reply in self._handle_reset(event, rest):
                    yield reply

            elif head in ("help", "帮助", "?", "？"):
                yield self._reply(event, progress.render_help())

            else:
                yield self._reply(
                    event, f"❓ 未知子命令：{head}\n\n" + progress.render_help()
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 指令处理异常: %s", PLUGIN_NAME, exc, exc_info=True)
            yield event.plain_result("😵 指令处理出错，请查看后台日志。")

    # ------------------------------------------------------------------ #
    # 子命令处理
    # ------------------------------------------------------------------ #

    async def _handle_set(self, event: AstrMessageEvent, rest: str):
        """``/zl set <群号> <QQ号>`` 或 ``/zl set <QQ号>``。"""
        tokens = rest.split()
        if not tokens:
            yield self._reply(event, "用法：/zl set <群号> <QQ号>，或 /zl set <QQ号>（当前群）")
            return

        if len(tokens) == 1:
            group_id = str(event.get_group_id() or "")
            qq = tokens[0]
            nickname = ""
            if not group_id:
                yield self._reply(event, "⚠️ 不在群里，请补上群号：/zl set <群号> <QQ号>")
                return
        else:
            group_id = tokens[0]
            qq = tokens[1]
            nickname = tokens[2] if len(tokens) > 2 else ""

        if not group_id.isdigit():
            yield self._reply(event, "⚠️ 群号必须是纯数字。")
            return
        if not qq.isdigit():
            yield self._reply(event, "⚠️ QQ 号必须是纯数字。")
            return

        self.state.group_id = group_id
        self.state.qq_id = qq
        if nickname:
            self.state.nickname = nickname
        await self._persist_state()

        # 切换目标后刷新计数缓存
        self.state.total_messages = await self.storage.count_messages(group_id, qq)
        self.state.undistilled = await self.storage.count_undistilled(group_id, qq)

        yield self._reply(
            event,
            f"🎯 目标已设定：{self.state.nickname or '未知昵称'} "
            f"({qq}) @ 群 {group_id}\n现有语料 {self.state.total_messages} 条。",
        )

    async def _handle_export(self, event: AstrMessageEvent):
        """``/zl export``：导出 Markdown 到数据目录。"""
        if not self.state.has_target():
            yield self._reply(event, "⚠️ 还没有设定目标，先 /zl set。")
            return
        try:
            snapshot = await self._load_snapshot() or empty_snapshot()
            markdown = prompts.render_export_markdown(
                snapshot, self.state.nickname, self.state.qq_id, PLUGIN_DISPLAY
            )
            out_dir = resolve_plugin_data_dir()
            path = out_dir / f"persona_{self.state.qq_id}.md"
            await _write_text(path, markdown)
            preview = markdown[:600]
            yield self._reply(event, f"📄 已导出到：{path}\n\n{preview}\n\n……（预览已截断）")
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 导出失败: %s", PLUGIN_NAME, exc, exc_info=True)
            yield self._reply(event, "😵 导出失败，请查看后台日志。")

    async def _handle_correct(self, event: AstrMessageEvent, rest: str):
        """``/zl correct <内容>`` / ``/zl 纠正 <内容>``。"""
        if not self.state.has_target():
            yield self._reply(event, "⚠️ 还没有设定目标，先 /zl set。")
            return
        content = rest.strip()
        if not content:
            yield self._reply(event, "用法：/zl correct <要纠正的内容>")
            return
        ok = await self.storage.add_correction(self.state.group_id, self.state.qq_id, content)
        if ok:
            yield self._reply(event, f"✅ 已记下纠正（优先级最高）：\n- {content}")
        else:
            yield self._reply(event, "😵 写入纠正失败，请查看后台日志。")

    async def _handle_reset(self, event: AstrMessageEvent, rest: str):
        """``/zl reset confirm``：二次确认后清空语料。"""
        if not self.state.has_target():
            yield self._reply(event, "⚠️ 还没有设定目标，先 /zl set。")
            return
        if rest.strip().lower() not in ("confirm", "确认", "yes", "y"):
            yield self._reply(
                event,
                "⚠️ 该操作会清空该目标的全部语料（档案保留）。\n"
                "确认无误请再次发送：/zl reset confirm",
            )
            return
        removed = await self.storage.clear_messages(self.state.group_id, self.state.qq_id)
        self.state.total_messages = 0
        self.state.undistilled = 0
        yield self._reply(event, f"🧹 已清空 {removed} 条语料。（人格档案与纠正层仍保留）")

    # ------------------------------------------------------------------ #
    # 渲染与工具
    # ------------------------------------------------------------------ #

    async def _build_panel(self) -> str:
        """组装并渲染进度面板。"""
        if not self.state.has_target():
            return progress.render_no_target()

        snapshot = await self._load_snapshot()
        meta = snapshot.get("meta", {}) if isinstance(snapshot, dict) else {}
        min_ts, max_ts = await self.storage.get_time_span(
            self.state.group_id, self.state.qq_id
        )

        data = progress.PanelData(
            plugin_name=PLUGIN_DISPLAY,
            has_target=True,
            nickname=self.state.nickname,
            qq=self.state.qq_id,
            group_id=self.state.group_id,
            distilling=self.distiller.is_running(),
            listen=bool(self.state.enabled and self.state.listen_enabled),
            total=self.state.total_messages,
            saturation=self._safe_int(self.config.get("saturation_messages", 1500), 1500),
            round_no=int(meta.get("distill_round", 0) or 0),
            last_distill_ts=int(meta.get("last_distill_at", 0) or 0),
            queue=self.state.undistilled,
            min_ts=min_ts,
            max_ts=max_ts,
            completeness=compute_completeness(snapshot),
            formed_layers=count_formed_layers(snapshot),
            total_layers=len(prompts.LAYER_KEYS),
        )
        return progress.render_panel(data)

    def _render_target(self) -> str:
        """渲染 ``/zl now`` 的目标信息。"""
        if not self.state.has_target():
            return "❗ 当前还没有设定目标。用 /zl set <群号> <QQ号> 设定。"
        listen = "开启" if (self.state.enabled and self.state.listen_enabled) else "关闭"
        return (
            f"🎯 当前目标：{self.state.nickname or '未知昵称'} ({self.state.qq_id})\n"
            f"🏠 目标群：{self.state.group_id}\n"
            f"📡 采集状态：{listen}"
        )

    async def _load_snapshot(self) -> Optional[dict[str, Any]]:
        """读取 Persona 快照，并注入最新的人工纠正层。"""
        if not self.state.has_target():
            return None
        snapshot = await self.storage.get_persona(self.state.group_id, self.state.qq_id)
        if snapshot is None:
            return None
        corrections = await self.storage.get_corrections(
            self.state.group_id, self.state.qq_id
        )
        snapshot["corrections"] = corrections
        return snapshot

    def _reply(self, event: AstrMessageEvent, text: str):
        """构造回复结果；按配置决定是否 @ 提问者。"""
        if bool(self.config.get("reply_with_at", True)):
            try:
                chain = [Comp.At(qq=event.get_sender_id()), Comp.Plain("\n" + text)]
                return event.chain_result(chain)
            except Exception:  # noqa: BLE001 - 组装富媒体失败则退化纯文本
                pass
        return event.plain_result(text)

    @staticmethod
    def _is_admin(event: AstrMessageEvent) -> bool:
        """安全判断事件是否为管理员。"""
        try:
            return bool(event.is_admin())
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        """把任意配置值安全转换为 int。"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    async def _load_runtime_state(self) -> None:
        """从配置初始化运行时状态，再用持久化覆盖（指令修改优先）。"""
        self.state.enabled = bool(self.config.get("enabled", True))
        self.state.listen_enabled = bool(self.config.get("listen_enabled", True))
        self.state.group_id = str(self.config.get("target_group_id", "") or "")
        self.state.qq_id = str(self.config.get("target_qq_id", "") or "")
        self.state.nickname = str(self.config.get("target_nickname", "") or "")

        for store_key, attr in (
            ("rt_target_group_id", "group_id"),
            ("rt_target_qq_id", "qq_id"),
            ("rt_target_nickname", "nickname"),
        ):
            stored = await self.storage.get_state(store_key)
            if stored is not None:
                setattr(self.state, attr, stored)

        for store_key, attr in (("rt_enabled", "enabled"), ("rt_listen_enabled", "listen_enabled")):
            stored = await self.storage.get_state(store_key)
            if stored is not None:
                setattr(self.state, attr, stored == "1")

    async def _persist_state(self) -> None:
        """把运行时状态写入持久化 state 表。"""
        await self.storage.set_state("rt_target_group_id", self.state.group_id)
        await self.storage.set_state("rt_target_qq_id", self.state.qq_id)
        await self.storage.set_state("rt_target_nickname", self.state.nickname)
        await self.storage.set_state("rt_enabled", "1" if self.state.enabled else "0")
        await self.storage.set_state(
            "rt_listen_enabled", "1" if self.state.listen_enabled else "0"
        )
