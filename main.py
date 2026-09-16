"""AstrBot 插件主入口：我要蒸馏群友（astrbot_plugin_group_distiller）。

静默采集多个「群 + 群友」目标的聊天语料，调用 LLM 蒸馏成 5 层 AI 人格档案，
通过 ``/zl`` 指令族在群内查看进度、管理目标，并把蒸馏结果生成 / 写入
AstrBot 的「人格设定」。

作者：Shawlei
版本：v0.2.0
仓库：https://github.com/Shawlei/astrbot_plugin_group_distiller
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Optional

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp

try:  # MessageChain 用于"主动发消息"（每日总结播报），缺失时该功能优雅降级
    from astrbot.api.event import MessageChain  # type: ignore
except ImportError:  # pragma: no cover - 仅在精简/老版本环境触发
    MessageChain = None  # type: ignore

try:  # 兼容包内导入 / 顶层导入两种加载方式
    from .core import persona_bridge, progress, prompts, schedule
    from .core import targets as targets_mod
    from .core.collector import Collector, RuntimeState
    from .core.distiller import (
        Distiller,
        compute_completeness,
        count_formed_layers,
        empty_snapshot,
    )
    from .core.schedule import DailyDigestScheduler, format_hhmm
    from .core.storage import Storage, resolve_plugin_data_dir
    from .core.targets import TargetSpec
except ImportError:  # pragma: no cover
    from core import persona_bridge, progress, prompts  # type: ignore
    from core import schedule  # type: ignore
    from core import targets as targets_mod  # type: ignore
    from core.collector import Collector, RuntimeState  # type: ignore
    from core.distiller import (  # type: ignore
        Distiller,
        compute_completeness,
        count_formed_layers,
        empty_snapshot,
    )
    from core.schedule import DailyDigestScheduler, format_hhmm  # type: ignore
    from core.storage import Storage, resolve_plugin_data_dir  # type: ignore
    from core.targets import TargetSpec  # type: ignore

PLUGIN_NAME = "astrbot_plugin_group_distiller"
PLUGIN_DISPLAY = "我要蒸馏群友"
PLUGIN_VERSION = "v0.2.0"

# 可能出现在 message_str 开头的指令前缀（AstrBot 有时会剥离，有时不会）
_PREFIXES = ("/zl", "zl", "／zl", "/蒸馏", "蒸馏", "／蒸馏")

# 目标管理的子命令别名
_HEADS_LIST = ("list", "ls", "列表", "目标", "清单")
_HEADS_ADD = ("add", "添加", "新增", "加")
_HEADS_DEL = ("del", "delete", "remove", "rm", "删", "删除")
_HEADS_USE = ("use", "switch", "切换", "选中")
_HEADS_PERSONA = ("persona", "人格", "模板", "人格模板")
_HEADS_PUSH = ("push", "写入", "应用", "装进人格")


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


@register(PLUGIN_NAME, "Shawlei", "把群友蒸馏成 AI 人格档案", PLUGIN_VERSION)
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
        # 每日定时总结：读配置 → 到点回调 main 的 _run_daily_digest
        self.scheduler = DailyDigestScheduler(
            self._daily_cfg, self._run_daily_digest, PLUGIN_NAME
        )
        self.scheduler.set_state_hook(self._persist_digest_date)

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """插件实例化后自动调用：初始化存储、加载目标、启动采集与每日定时总结。"""
        try:
            await self.storage.init()
            await self._load_runtime_state()
            await self._refresh_counts()
            self.distiller.set_on_distilled(self._on_distilled)
            await self.collector.start()

            daily = self._daily_cfg()
            self.scheduler.restore_last_date(
                await self.storage.get_state("rt_digest_last_date")
            )
            if daily["enabled"]:
                await self.scheduler.start()
                logger.info(
                    "[%s] 每日定时总结已启用：每天 %s（当日最少 %d 条；上次：%s）",
                    PLUGIN_NAME,
                    daily["display_time"],
                    daily["min_messages"],
                    self.scheduler.last_date() or "无记录",
                )

            logger.info(
                "[%s] %s 初始化完成：%d 个目标，当前 %s",
                PLUGIN_NAME,
                PLUGIN_VERSION,
                len(self.state.targets),
                self.state.active_target().label()
                if self.state.active_target()
                else "无",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 初始化失败: %s", PLUGIN_NAME, exc, exc_info=True)

    async def terminate(self) -> None:
        """插件卸载/停用时调用：落盘、取消后台任务、关闭连接。"""
        try:
            await self.scheduler.stop()
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 停止每日总结调度器失败: %s", PLUGIN_NAME, exc)
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

            elif head in _HEADS_ADD:
                async for reply in self._handle_add(event, rest):
                    yield reply

            elif head in _HEADS_DEL:
                async for reply in self._handle_del(event, rest):
                    yield reply

            elif head in _HEADS_USE:
                async for reply in self._handle_use(event, rest):
                    yield reply

            elif head in _HEADS_LIST:
                yield self._reply(event, await self._render_target_list())

            elif head in ("now",):
                yield self._reply(event, self._render_target())

            elif head in ("distill", "蒸馏", "开始"):
                all_targets = rest.strip().lower() in ("all", "全部", "所有", "全")
                result = await self.distiller.trigger(
                    event.unified_msg_origin, manual=True, all_targets=all_targets
                )
                yield self._reply(event, result.message)

            elif head in ("digest", "总结", "日报", "每日总结"):
                async for reply in self._handle_digest(event, rest):
                    yield reply

            elif head in ("profile", "档案", "画像"):
                snapshot = await self._load_snapshot()
                active = self.state.active_target()
                yield self._reply(
                    event,
                    progress.render_profile(
                        snapshot,
                        self.state.nickname,
                        self.state.qq_id,
                        active.group_id if active else "",
                    ),
                )

            elif head in _HEADS_PERSONA:
                async for reply in self._handle_persona(event):
                    yield reply

            elif head in _HEADS_PUSH:
                async for reply in self._handle_push(event):
                    yield reply

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
    # 目标管理子命令
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_target_args(rest: str, event: AstrMessageEvent) -> tuple[str, str, str]:
        """把 ``rest`` 解析成 ``(群号, QQ号, 昵称)``。

        支持 ``<群号> <QQ号> [昵称]`` 与 ``<QQ号> [昵称]``（用当前群）两种写法。
        昵称允许带空格，因此第三段取剩余整串。
        """
        parts = rest.split(maxsplit=2)
        if not parts:
            return "", "", ""
        if len(parts) == 1:
            return str(event.get_group_id() or ""), parts[0], ""
        nickname = parts[2] if len(parts) > 2 else ""
        return parts[0], parts[1], nickname

    async def _handle_set(self, event: AstrMessageEvent, rest: str):
        """``/zl set <群号> <QQ号>``：**替换**目标列表（保留原有语料与档案）。"""
        tokens = rest.split()
        if not tokens:
            yield self._reply(event, "用法：/zl set <群号> <QQ号>，或 /zl set <QQ号>（当前群）")
            return

        group_id, qq, nickname = self._split_target_args(rest, event)
        if not group_id:
            yield self._reply(event, "⚠️ 不在群里，请补上群号：/zl set <群号> <QQ号>")
            return
        error = self._validate_ids(group_id, qq)
        if error:
            yield self._reply(event, error)
            return

        spec = TargetSpec(group_id=group_id, qq_id=qq, nickname=nickname)
        self.state.clear_targets()
        self.state.add_target(spec)
        self.state.set_active(spec.key)
        await self._after_target_change()

        yield self._reply(
            event,
            f"🎯 目标已设定（已替换原有清单）：{spec.label()}\n"
            f"现有语料 {self.state.total_messages} 条。\n"
            "想同时蒸馏多个人，请改用 /zl add。",
        )

    async def _handle_add(self, event: AstrMessageEvent, rest: str):
        """``/zl add <群号> <QQ号> [昵称]``：追加一个目标。"""
        if not rest.strip():
            yield self._reply(
                event,
                "用法：/zl add <群号> <QQ号> [昵称]，或 /zl add <QQ号> [昵称]（当前群）",
            )
            return

        group_id, qq, nickname = self._split_target_args(rest, event)
        if not group_id:
            yield self._reply(event, "⚠️ 不在群里，请补上群号：/zl add <群号> <QQ号>")
            return
        error = self._validate_ids(group_id, qq)
        if error:
            yield self._reply(event, error)
            return

        spec = TargetSpec(group_id=group_id, qq_id=qq, nickname=nickname)
        existed = any(t.key == spec.key for t in self.state.targets)
        self.state.add_target(spec)
        if not existed:
            self.state.set_active(spec.key)
        await self._after_target_change()

        verb = "已更新" if existed else "已添加"
        yield self._reply(
            event,
            f"{'♻️' if existed else '➕'} 目标{verb}：{spec.label()}\n"
            f"当前共 {len(self.state.targets)} 个目标，"
            f"当前选中 {(self.state.active_target() or spec).label()}。",
        )

    async def _handle_del(self, event: AstrMessageEvent, rest: str):
        """``/zl del <QQ号> [purge]``：删除目标；加 purge 连语料与档案一起删。"""
        parts = rest.split()
        if not parts:
            yield self._reply(event, "用法：/zl del <QQ号> [purge]")
            return

        qq = parts[0]
        purge = len(parts) > 1 and parts[1].lower() in (
            "purge",
            "彻底",
            "清除",
            "全部",
            "清空",
        )
        if not targets_mod.is_valid_id(qq):
            yield self._reply(event, "⚠️ QQ 号必须是纯数字（5~20 位）。")
            return

        doomed = [t for t in self.state.collect_targets() if t.qq_id == qq]
        if not doomed:
            yield self._reply(event, f"🤔 没找到 QQ 号为 {qq} 的目标，用 /zl list 看看。")
            return

        removed = self.state.remove_target(qq)
        for spec in doomed:
            self.state.overview.pop(spec.key, None)
            if purge:
                await self.storage.clear_messages(spec.group_id, spec.qq_id)
                await self.storage.delete_persona(spec.group_id, spec.qq_id)
        await self._after_target_change()

        tail = "（语料与档案已一并删除）" if purge else "（语料与档案仍保留在库里）"
        yield self._reply(
            event,
            f"🗑️ 已删除 {removed} 个目标{tail}。\n"
            f"剩余 {len(self.state.targets)} 个目标。",
        )

    async def _handle_use(self, event: AstrMessageEvent, rest: str):
        """``/zl use <QQ号>``：切换当前选中目标。"""
        key = rest.strip()
        if not key:
            yield self._reply(event, "用法：/zl use <QQ号>")
            return
        if not self.state.set_active(key):
            yield self._reply(event, f"🤔 没找到目标「{key}」，用 /zl list 看看。")
            return
        await self._refresh_counts()
        await self._persist_state()
        active = self.state.active_target()
        yield self._reply(
            event,
            f"🎯 已切换到：{active.label() if active else key}\n"
            f"该目标现有语料 {self.state.total_messages} 条。",
        )

    async def _handle_persona(self, event: AstrMessageEvent):
        """``/zl persona``：生成可粘贴进 AstrBot 的人格模板。"""
        active = self.state.active_target()
        if active is None:
            yield self._reply(event, "⚠️ 还没有设定目标，先 /zl add <群号> <QQ号>。")
            return

        snapshot = await self._load_snapshot()
        persona_id = self._persona_id(active)
        template = self._build_persona_text(active, snapshot)
        chunks = progress.chunk_text(template)

        yield self._reply(
            event,
            prompts.build_persona_summary(snapshot, active.display_name, active.qq_id),
        )
        yield self._reply(
            event,
            progress.render_persona_header(
                active.display_name, active.qq_id, active.group_id, persona_id, len(chunks)
            ),
        )
        for chunk in chunks:
            yield event.plain_result(chunk)

    async def _handle_push(self, event: AstrMessageEvent):
        """``/zl push``：把人格模板直接写进 AstrBot 人格设定。"""
        active = self.state.active_target()
        if active is None:
            yield self._reply(event, "⚠️ 还没有设定目标，先 /zl add <群号> <QQ号>。")
            return

        snapshot = await self._load_snapshot()
        if not snapshot:
            yield self._reply(
                event, "⚠️ 这个目标还没有档案，先攒语料并跑一轮 /zl distill。"
            )
            return

        persona_id = self._persona_id(active)
        text = self._build_persona_text(active, snapshot)
        ok, message = await persona_bridge.push_persona(self.context, persona_id, text)

        if ok:
            yield self._reply(
                event,
                f"🪪 {message}\n"
                "去 AstrBot WebUI 的「人格设定」里就能看到它，"
                "在会话配置里选中即可生效。",
            )
        else:
            yield self._reply(
                event,
                f"😵 {message}\n"
                "（可以改用 /zl persona 拿到模板，手动粘进「人格设定」）",
            )

    # ------------------------------------------------------------------ #
    # 每日定时总结
    # ------------------------------------------------------------------ #

    async def _handle_digest(self, event: AstrMessageEvent, rest: str):
        """``/zl digest [all]``：立刻跑一次「每日总结」（用当天的全部对话）。"""
        specs = self.state.collect_targets()
        if not specs:
            yield self._reply(event, "⚠️ 还没有设定目标，先 /zl add <群号> <QQ号>。")
            return
        if self.distiller.is_running():
            yield self._reply(event, "⏳ 已有蒸馏正在进行，等它跑完再试。")
            return

        scope_all = rest.strip().lower() in ("all", "全部", "所有", "全")
        scope = "全部 %d 个目标" % len(specs) if scope_all else self._reply_scope()
        umo = event.unified_msg_origin

        # 蒸馏可能耗时（要调 LLM），扔后台跑，先立刻回执，跑完再回报结果
        asyncio.create_task(self._daily_digest_then_report(umo, scope_all))
        yield self._reply(
            event,
            f"🔬 已开始跑每日总结（{scope}，使用今天的全部对话）。\n"
            "完成后我会回一条结果。",
        )

    def _reply_scope(self) -> str:
        """当前目标的简短描述，用于回执文案。"""
        active = self.state.active_target()
        return active.label() if active else "当前目标"

    async def _daily_digest_then_report(self, umo: str, scope_all: bool) -> None:
        """后台执行每日总结，并把结果回发到发起会话。"""
        now = int(time.time())
        day_start = schedule.day_start_ts(now)
        try:
            if scope_all:
                ok, detail = await self.distiller.run_daily_digest(
                    self._umo_for, day_start, now, min_messages=0
                )
            else:
                spec = self.state.active_target()
                if spec is None:
                    ok, detail = True, "没有配置目标"
                else:
                    ok, detail = await self.distiller.digest_day(
                        umo, spec, day_start, now, min_messages=0
                    )
            await self._refresh_counts()
            await self._send_text(
                umo,
                f"{'✅' if ok else '⚠️'} 每日总结（{schedule.date_str(now)}）：{detail}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 手动每日总结失败: %s", PLUGIN_NAME, exc, exc_info=True)
            await self._send_text(umo, "😵 每日总结出错，详见后台日志。")

    def _daily_cfg(self) -> dict[str, Any]:
        """读取并规整「每日定时总结」配置。"""
        raw = self.config.get("daily_digest")
        if not isinstance(raw, dict):
            raw = {}
        return {
            "enabled": bool(raw.get("enabled", False)),
            "time": str(raw.get("time", "") or ""),
            "display_time": format_hhmm(raw.get("time", "")),
            "min_messages": self._safe_int(raw.get("min_messages", 5), 5),
            "notify": bool(raw.get("notify", False)),
        }

    def _umo_for(self, spec: TargetSpec) -> str:
        """某目标该往哪个会话播报（优先用采集时真实见过的 umo）。"""
        return self.state.umo_for(spec.group_id)

    async def _run_daily_digest(self, now: float, day_start: int) -> bool:
        """调度器到点时的回调：对全部目标跑一次当日总结。

        Returns:
            True 表示处理完毕（不必重试）；False 表示遇到可重试的故障。
        """
        daily = self._daily_cfg()
        if not self.state.collect_targets():
            logger.info("[%s] 每日总结跳过：没有配置任何目标。", PLUGIN_NAME)
            return True

        items = await self.distiller.run_daily_digest_detailed(
            self._umo_for, day_start, int(now), min_messages=daily["min_messages"]
        )
        await self._refresh_counts()

        summary = "；".join(
            f"{item.spec.display_name}({item.spec.qq_id})：{item.detail}"
            for item in items
        )
        logger.info(
            "[%s] 每日总结（%s）结果：%s", PLUGIN_NAME, schedule.date_str(now), summary
        )

        if daily["notify"]:
            await self._broadcast_digest(items)

        return all(item.ok for item in items) if items else True

    async def _broadcast_digest(self, items: list[Any]) -> None:
        """把每日总结结果播报到各目标所在的群（可选功能）。

        **逐目标发**：每个群只收到自己群里那些目标的结果，不把别的群的信息
        串过去。
        """
        for item in items:
            spec = item.spec
            head = "✅ 今日蒸馏总结完成" if item.ok else "⚠️ 今日蒸馏总结有异常"
            await self._send_text(
                self._umo_for(spec), f"{head}\n{spec.display_name}({spec.qq_id})：{item.detail}"
            )

    async def _send_text(self, umo: str, text: str) -> None:
        """主动往某个会话发一条纯文本（失败只记日志，绝不外抛）。"""
        if not umo:
            return
        if MessageChain is None:
            # 当前环境没导出 MessageChain（罕见），放弃播报而不是让插件崩
            logger.debug("[%s] 缺少 MessageChain，跳过主动播报。", PLUGIN_NAME)
            return
        try:
            await self.context.send_message(umo, MessageChain().message(text))
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 主动发送消息失败: %s", PLUGIN_NAME, exc)

    async def _persist_digest_date(self, date: str) -> None:
        """持久化「每日总结上次完成日期」，避免重启后重复触发。"""
        try:
            await self.storage.set_state("rt_digest_last_date", str(date or ""))
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 写入每日总结日期失败: %s", PLUGIN_NAME, exc)

    # ------------------------------------------------------------------ #
    # 其他子命令
    # ------------------------------------------------------------------ #

    async def _handle_export(self, event: AstrMessageEvent):
        """``/zl export``：导出 Markdown 档案 + 人格模板到数据目录。"""
        active = self.state.active_target()
        if active is None:
            yield self._reply(event, "⚠️ 还没有设定目标，先 /zl add。")
            return
        try:
            snapshot = await self._load_snapshot() or empty_snapshot()
            markdown = prompts.render_export_markdown(
                snapshot, active.display_name, active.qq_id, PLUGIN_DISPLAY
            )
            persona_text = self._build_persona_text(active, snapshot)

            out_dir = resolve_plugin_data_dir()
            md_path = out_dir / f"persona_{active.qq_id}.md"
            await _write_text(md_path, markdown)
            persona_path = await persona_bridge.export_persona_file(
                out_dir, active, persona_text
            )

            persona_line = f"\n🪪 人格模板：{persona_path}" if persona_path else ""
            yield self._reply(
                event,
                f"📄 已导出：{md_path}{persona_line}\n\n{markdown[:600]}\n\n……（预览已截断）",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 导出失败: %s", PLUGIN_NAME, exc, exc_info=True)
            yield self._reply(event, "😵 导出失败，请查看后台日志。")

    async def _handle_correct(self, event: AstrMessageEvent, rest: str):
        """``/zl correct <内容>`` / ``/zl 纠正 <内容>``。"""
        active = self.state.active_target()
        if active is None:
            yield self._reply(event, "⚠️ 还没有设定目标，先 /zl add。")
            return
        content = rest.strip()
        if not content:
            yield self._reply(event, "用法：/zl correct <要纠正的内容>")
            return
        ok = await self.storage.add_correction(active.group_id, active.qq_id, content)
        if ok:
            yield self._reply(event, f"✅ 已记下纠正（优先级最高）：\n- {content}")
        else:
            yield self._reply(event, "😵 写入纠正失败，请查看后台日志。")

    async def _handle_reset(self, event: AstrMessageEvent, rest: str):
        """``/zl reset confirm``：二次确认后清空当前目标的语料。"""
        active = self.state.active_target()
        if active is None:
            yield self._reply(event, "⚠️ 还没有设定目标，先 /zl add。")
            return
        if rest.strip().lower() not in ("confirm", "确认", "yes", "y"):
            yield self._reply(
                event,
                f"⚠️ 该操作会清空 {active.label()} 的全部语料（档案保留）。\n"
                "确认无误请再次发送：/zl reset confirm",
            )
            return
        removed = await self.storage.clear_messages(active.group_id, active.qq_id)
        await self._refresh_counts()
        yield self._reply(event, f"🧹 已清空 {removed} 条语料。（人格档案与纠正层仍保留）")

    # ------------------------------------------------------------------ #
    # 渲染
    # ------------------------------------------------------------------ #

    async def _build_panel(self) -> str:
        """组装并渲染进度面板。"""
        specs = self.state.collect_targets()
        if not specs:
            return progress.render_no_target()

        for spec in specs:
            await self._ensure_counts(spec)

        active = self.state.active_target() or specs[0]
        snapshot = await self._load_snapshot()
        meta = snapshot.get("meta", {}) if isinstance(snapshot, dict) else {}
        min_ts, max_ts = await self.storage.get_time_span(active.group_id, active.qq_id)
        counts = self.state.counts_for(active)

        rows: list[progress.TargetRow] = []
        if len(specs) > 1:
            rows = await self._build_rows(specs, active.key)

        daily = self._daily_cfg()

        data = progress.PanelData(
            plugin_name=PLUGIN_DISPLAY,
            has_target=True,
            nickname=active.display_name,
            qq=active.qq_id,
            group_id=active.group_id,
            distilling=self.distiller.is_running(),
            listen=bool(self.state.enabled and self.state.listen_enabled),
            total=counts["total"],
            saturation=self._safe_int(self.config.get("saturation_messages", 1500), 1500),
            round_no=int(meta.get("distill_round", 0) or 0),
            last_distill_ts=int(meta.get("last_distill_at", 0) or 0),
            queue=counts["undistilled"],
            min_ts=min_ts,
            max_ts=max_ts,
            completeness=compute_completeness(snapshot),
            formed_layers=count_formed_layers(snapshot),
            total_layers=len(prompts.LAYER_KEYS),
            targets=rows,
            daily_enabled=daily["enabled"],
            daily_time=daily["display_time"],
            daily_last=self.scheduler.last_result(),
        )
        return progress.render_panel(data)

    async def _build_rows(
        self, specs: list[TargetSpec], active_key: str
    ) -> list[progress.TargetRow]:
        """为目标清单构建渲染行（含各目标的轮次与完整度）。"""
        current = self.distiller.current_target()
        rows: list[progress.TargetRow] = []
        for spec in specs:
            await self._ensure_counts(spec)
            snapshot = await self.storage.get_persona(spec.group_id, spec.qq_id)
            meta = snapshot.get("meta", {}) if isinstance(snapshot, dict) else {}
            counts = self.state.counts_for(spec)
            rows.append(
                progress.TargetRow(
                    key=spec.key,
                    nickname=spec.display_name,
                    qq=spec.qq_id,
                    group_id=spec.group_id,
                    total=counts["total"],
                    undistilled=counts["undistilled"],
                    round_no=int(meta.get("distill_round", 0) or 0),
                    completeness=compute_completeness(snapshot),
                    active=spec.key == active_key,
                    distilling=bool(current and current.key == spec.key),
                    has_persona=bool(snapshot),
                )
            )
        return rows

    async def _render_target_list(self) -> str:
        """渲染 ``/zl list``。"""
        specs = self.state.collect_targets()
        if not specs:
            return progress.render_target_list([])
        for spec in specs:
            await self._ensure_counts(spec)
        rows = await self._build_rows(specs, self.state.active_key)
        return progress.render_target_list(rows, self.state.active_key)

    def _render_target(self) -> str:
        """渲染 ``/zl now`` 的目标信息。"""
        specs = self.state.collect_targets()
        if not specs:
            return "❗ 当前还没有设定目标。用 /zl add <群号> <QQ号> 添加。"
        active = self.state.active_target()
        listen = "开启" if (self.state.enabled and self.state.listen_enabled) else "关闭"
        lines = [
            f"🎯 当前目标：{active.label() if active else '未知'}",
            f"📡 采集状态：{listen}",
            f"📋 目标总数：{len(specs)} 个（/zl list 看全部）",
        ]
        return "\n".join(lines)

    async def _load_snapshot(self, spec: Optional[TargetSpec] = None) -> Optional[dict[str, Any]]:
        """读取 Persona 快照，并注入最新的人工纠正层。"""
        target = spec or self.state.active_target()
        if target is None:
            return None
        snapshot = await self.storage.get_persona(target.group_id, target.qq_id)
        if snapshot is None:
            return None
        corrections = await self.storage.get_corrections(target.group_id, target.qq_id)
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

    # ------------------------------------------------------------------ #
    # 人格模板
    # ------------------------------------------------------------------ #

    def _persona_cfg(self) -> dict[str, Any]:
        """读取并规整「人格模板」这一段配置。"""
        raw = self.config.get("persona_template")
        if not isinstance(raw, dict):
            raw = {}
        return {
            "enabled": bool(raw.get("enabled", True)),
            "prefix": str(raw.get("persona_id_prefix", "") or ""),
            "auto_write": bool(raw.get("auto_write", False)),
            "auto_write_threshold": self._safe_int(
                raw.get("auto_write_threshold", 80), 80
            ),
            "include_evidence": bool(raw.get("include_evidence", False)),
            "extra_rules": str(raw.get("extra_rules", "") or ""),
        }

    def _persona_id(self, spec: TargetSpec) -> str:
        """算出该目标对应的人格 ID。"""
        return persona_bridge.make_persona_id(
            self._persona_cfg()["prefix"], spec.nickname, spec.qq_id, spec.group_id
        )

    def _build_persona_text(
        self, spec: TargetSpec, snapshot: Optional[dict[str, Any]]
    ) -> str:
        """生成可粘贴 / 可写入 AstrBot 的人格模板文本。"""
        cfg = self._persona_cfg()
        return prompts.build_astrbot_persona(
            snapshot,
            spec.display_name,
            spec.qq_id,
            include_evidence=cfg["include_evidence"],
            extra_rules=cfg["extra_rules"],
            plugin_name=PLUGIN_DISPLAY,
        )

    async def _on_distilled(
        self, spec: TargetSpec, snapshot: dict[str, Any]
    ) -> None:
        """蒸馏完成回调：达到完整度阈值就自动写入 AstrBot 人格。"""
        cfg = self._persona_cfg()
        if not (cfg["enabled"] and cfg["auto_write"]):
            return
        completeness = compute_completeness(snapshot)
        if completeness < cfg["auto_write_threshold"]:
            return

        persona_id = self._persona_id(spec)
        text = self._build_persona_text(spec, snapshot)
        ok, message = await persona_bridge.push_persona(self.context, persona_id, text)
        if ok:
            logger.info(
                "[%s] 档案完整度 %d%% 达标，%s", PLUGIN_NAME, completeness, message
            )
        else:
            logger.warning("[%s] 自动写入人格失败：%s", PLUGIN_NAME, message)

    # ------------------------------------------------------------------ #
    # 状态与工具
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_ids(group_id: str, qq: str) -> str:
        """校验群号与 QQ 号，合法返回空串，否则返回给用户看的错误文案。"""
        if not targets_mod.is_valid_id(group_id):
            return "⚠️ 群号必须是纯数字（5~20 位）。"
        if not targets_mod.is_valid_id(qq):
            return "⚠️ QQ 号必须是纯数字（5~20 位）。"
        return ""

    async def _after_target_change(self) -> None:
        """目标集合变动后的统一收尾：刷新计数、持久化、同步镜像字段。"""
        self.state.sync_active_fields()
        await self._refresh_counts()
        await self._persist_state()

    async def _ensure_counts(self, spec: TargetSpec) -> None:
        """确保某个目标的计数缓存存在（缺失才查库）。"""
        if spec.key not in self.state.overview:
            await self._refresh_counts([spec])

    async def _refresh_counts(self, specs: Optional[list[TargetSpec]] = None) -> None:
        """用数据库统计刷新目标计数缓存。"""
        for spec in specs if specs is not None else self.state.collect_targets():
            try:
                total = await self.storage.count_messages(spec.group_id, spec.qq_id)
                undistilled = await self.storage.count_undistilled(
                    spec.group_id, spec.qq_id
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] 刷新计数失败: %s", PLUGIN_NAME, exc)
                continue
            self.state.set_counts(spec.key, total, undistilled)

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
        """加载目标与开关：配置打底，数据库中的运行期改动覆盖之。"""
        self.state.enabled = bool(self.config.get("enabled", True))
        self.state.listen_enabled = bool(self.config.get("listen_enabled", True))

        cfg_targets, warnings = targets_mod.collect_config_targets(self.config)
        for warning in warnings:
            logger.warning("[%s] 目标清单：%s", PLUGIN_NAME, warning)

        stored = targets_mod.targets_from_json(await self.storage.get_state("rt_targets"))
        if stored:
            # 运行期通过 /zl add、/zl del 改过的清单优先
            self.state.targets = stored
        else:
            self.state.targets = list(cfg_targets)

        # v0.1.0 遗留：库里只有一个单目标的 state
        if not self.state.targets:
            legacy = targets_mod.make_target(
                await self.storage.get_state("rt_target_group_id"),
                await self.storage.get_state("rt_target_qq_id"),
                await self.storage.get_state("rt_target_nickname") or "",
            )
            if legacy is not None:
                self.state.targets = [legacy]

        stored_active = await self.storage.get_state("rt_active_key")
        if not (stored_active and self.state.set_active(stored_active)):
            if self.state.targets:
                self.state.set_active(self.state.targets[0].key)
        self.state.sync_active_fields()

        for store_key, attr in (
            ("rt_enabled", "enabled"),
            ("rt_listen_enabled", "listen_enabled"),
        ):
            value = await self.storage.get_state(store_key)
            if value is not None:
                setattr(self.state, attr, value == "1")

    async def _persist_state(self) -> None:
        """把目标集合与开关写入持久化 state 表。"""
        await self.storage.set_state("rt_targets", targets_mod.targets_to_json(self.state.targets))
        await self.storage.set_state("rt_active_key", self.state.active_key)
        await self.storage.set_state("rt_enabled", "1" if self.state.enabled else "0")
        await self.storage.set_state(
            "rt_listen_enabled", "1" if self.state.listen_enabled else "0"
        )
        # 兼容旧键：万一需要回退到 v0.1.0，单目标信息还在
        await self.storage.set_state("rt_target_group_id", self.state.group_id)
        await self.storage.set_state("rt_target_qq_id", self.state.qq_id)
        await self.storage.set_state("rt_target_nickname", self.state.nickname)
