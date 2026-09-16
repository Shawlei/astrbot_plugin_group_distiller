"""每日定时总结（每日自动蒸馏）的调度器。

需求形态：用户打开一个开关、选定一个时刻（比如 23:00），到了点插件就把
目标**当天聊过的全部内容**拿去蒸一遍。

实现要点：

- **不依赖 AstrBot 的 cron 模块**：自己跑一个轻量 asyncio 循环，每 20 秒看一眼
  时钟。这样不用赌框架版本、也不需要额外权限。
- **一天只跑一次**：触发日期持久化在数据库 ``state`` 表里，重启不会重复触发。
- **当天补跑**：如果 23:00 时 bot 恰好没运行，23:40 启动起来也照样补一次
  （判定条件是"现在 >= 设定时刻 且 今天还没跑过"）。
- **失败可重试但有上限**：LLM 临时抽风不至于整天不再尝试，但也绝不会每 20 秒
  重试到天亮（每天最多 3 次）。
- **任何异常都不外抛**：后台任务崩掉就等于这个功能静默失效，必须兜住。

本模块零 AstrBot 强依赖，可被单元测试直接导入。
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Callable, Optional

try:  # 允许在未安装 AstrBot 的环境（如单元测试）下导入
    from astrbot.api import logger
except ImportError:  # pragma: no cover - 仅在无 AstrBot 环境触发
    import logging

    logger = logging.getLogger("astrbot_plugin_group_distiller")

PLUGIN_NAME = "astrbot_plugin_group_distiller"

# 默认触发时刻（晚上 11 点，群聊基本聊完了）
DEFAULT_TIME = "23:00"

# 时钟检查间隔：20 秒的精度足够，也不会让事件循环太累
CHECK_INTERVAL = 20.0

# 同一天最多重试几次（避免 LLM 长时间不可用时疯狂重试）
MAX_ATTEMPTS_PER_DAY = 3

# 把各种口语化写法归一到 "HH:MM" 形态的分隔符
_TIME_SEPARATORS = ("：", "点", "时", ".", "。", "h", "H")


def parse_hhmm(value: Any) -> Optional[tuple[int, int]]:
    """解析用户填的时间，返回 ``(时, 分)``；无法解析返回 None。

    容忍常见的口语化写法：``23:00`` / ``23：00`` / ``2300`` / ``23`` /
    ``23点`` / ``23点30`` / ``23时5分`` / ``23.30``。

    Args:
        value: 配置里填的值。

    Returns:
        ``(hour, minute)``，范围校验通过才返回；否则 None。
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return None

    for sep in _TIME_SEPARATORS:
        text = text.replace(sep, ":")
    text = text.replace("分", "").strip()
    text = re.sub(r"\s+", "", text)
    text = text.strip(":")

    if not text or not text.replace(":", "").isdigit():
        return None

    if ":" in text:
        head, _, tail = text.partition(":")
        if not head.isdigit() or ":" in tail:
            # "23:00:30" 这种多段写法一律拒绝，别猜用户到底想要几点
            return None
        # "23:5" 与 "23:05" 都按 5 分处理
        minute = int(tail) if tail.isdigit() else 0
        hour = int(head)
    elif len(text) <= 2:
        hour, minute = int(text), 0
    elif len(text) in (3, 4):
        # 2300 / 930 → 后两位是分钟
        hour, minute = int(text[:-2]), int(text[-2:])
    else:
        return None

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def format_hhmm(value: Any) -> str:
    """把时间配置规范化成 ``HH:MM`` 展示；解析不了就原样回显。"""
    parsed = parse_hhmm(value)
    if parsed is None:
        return str(value or "").strip() or DEFAULT_TIME
    return f"{parsed[0]:02d}:{parsed[1]:02d}"


def day_start_ts(now: Optional[float] = None) -> int:
    """返回"今天 00:00:00"的 Unix 时间戳（本地时区）。"""
    stamp = float(now if now is not None else time.time())
    local = time.localtime(stamp)
    return int(
        time.mktime((local.tm_year, local.tm_mon, local.tm_mday, 0, 0, 0, 0, 0, -1))
    )


def date_str(now: Optional[float] = None) -> str:
    """返回本地日期字符串 ``YYYY-MM-DD``。"""
    stamp = float(now if now is not None else time.time())
    return time.strftime("%Y-%m-%d", time.localtime(stamp))


def minutes_of_day(now: Optional[float] = None) -> int:
    """当前时刻距离今天 00:00 的分钟数。"""
    stamp = float(now if now is not None else time.time())
    local = time.localtime(stamp)
    return local.tm_hour * 60 + local.tm_min


class DailyDigestScheduler:
    """每日定时总结调度器。

    只负责"到点了没有、今天跑过没有"，真正要做什么由 ``on_due`` 回调决定。
    """

    def __init__(
        self,
        settings_provider: Callable[[], dict[str, Any]],
        on_due: Callable[[float, int], Any],
        plugin_name: str = PLUGIN_NAME,
        interval: float = CHECK_INTERVAL,
    ) -> None:
        """构造调度器（不做 IO、不启动任务）。

        Args:
            settings_provider: 返回当前设置字典的回调，键为
                ``enabled`` / ``time`` / ``min_messages`` / ``notify``。
                每 tick 都重新取一次，这样改了配置（重载插件后）能立刻生效。
            on_due: 到点时的异步回调，签名 ``await on_due(now, day_start) -> bool``。
                返回 True 表示本次处理完毕（不再重试），False 表示值得重试。
            plugin_name: 日志前缀。
            interval: 时钟检查间隔（秒）。
        """
        self._get_settings = settings_provider
        self._on_due = on_due
        self.plugin_name = plugin_name
        self._interval = max(1.0, float(interval))
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False
        # 已成功（或已放弃）处理的日期，形如 2026-09-16
        self._last_date = ""
        self._attempts_date = ""
        self._attempts = 0
        self._last_result = ""
        self._state_hook: Optional[Callable[[str], Any]] = None

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """启动后台时钟循环（幂等）。"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """停止后台循环（幂等）。"""
        self._running = False
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] 停止每日总结调度器时出错: %s", self.plugin_name, exc)

    def is_running(self) -> bool:
        """后台循环是否在跑。"""
        if self._running:
            return True
        return self._task is not None and not self._task.done()

    async def _loop(self) -> None:
        """每 ``interval`` 秒检查一次时钟。"""
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                await self.tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001 - 后台任务必须吞掉所有异常
                logger.error("[%s] 每日总结调度异常: %s", self.plugin_name, exc)

    # ------------------------------------------------------------------ #
    # 状态（由 main 负责持久化）
    # ------------------------------------------------------------------ #

    def restore_last_date(self, value: Any) -> None:
        """从持久化状态恢复"上次完成日期"。"""
        self._last_date = str(value or "").strip()

    def last_date(self) -> str:
        """上次完成（或放弃）的日期。"""
        return self._last_date

    def last_result(self) -> str:
        """上次触发的结果说明（用于面板展示）。"""
        return self._last_result

    def set_state_hook(self, callback: Optional[Callable[[str], Any]]) -> None:
        """注册"完成日期发生变化"的钩子，供 main 落盘。

        回调签名 ``callback(date_str)``，同步/异步均可；异常会被吞掉。
        """
        self._state_hook = callback

    async def _emit_state(self) -> None:
        """把最新的完成日期通知给外部（用于持久化）。"""
        hook = getattr(self, "_state_hook", None)
        if hook is None:
            return
        try:
            result = hook(self._last_date)
            if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
                await result
        except Exception as exc:  # noqa: BLE001 - 落盘失败不影响调度
            logger.error("[%s] 持久化每日总结状态失败: %s", self.plugin_name, exc)

    # ------------------------------------------------------------------ #
    # 判定与触发
    # ------------------------------------------------------------------ #

    def seconds_until_next(self, now: Optional[float] = None) -> int:
        """距离下一次触发还有多少秒；未启用/时间非法时返回 -1。"""
        settings = self._get_settings() or {}
        if not settings.get("enabled"):
            return -1
        parsed = parse_hhmm(settings.get("time"))
        if parsed is None:
            return -1
        current = float(now if now is not None else time.time())
        local = time.localtime(current)
        target = int(
            time.mktime(
                (
                    local.tm_year,
                    local.tm_mon,
                    local.tm_mday,
                    parsed[0],
                    parsed[1],
                    0,
                    0,
                    0,
                    -1,
                )
            )
        )
        if target <= current:
            target += 86400  # 今天的点已经过了，算明天
        return max(0, int(target - current))

    async def tick(self, now: Optional[float] = None) -> bool:
        """检查一次是否该触发。

        Args:
            now: 当前时间戳；默认取系统时间（测试可注入）。

        Returns:
            是否真的触发了回调。
        """
        try:
            settings = self._get_settings() or {}
            if not settings.get("enabled"):
                return False

            parsed = parse_hhmm(settings.get("time"))
            if parsed is None:
                logger.warning(
                    "[%s] 每日总结时间「%s」解析不了，本次跳过。",
                    self.plugin_name,
                    settings.get("time"),
                )
                return False

            current = float(now if now is not None else time.time())
            today = date_str(current)
            if self._last_date == today:
                return False

            hour, minute = parsed
            if minutes_of_day(current) < hour * 60 + minute:
                return False  # 还没到点

            if self._attempts_date != today:
                self._attempts_date = today
                self._attempts = 0
            self._attempts += 1

            start = day_start_ts(current)
            logger.info(
                "[%s] 每日总结触发（%s %02d:%02d，第 %d 次尝试）。",
                self.plugin_name,
                today,
                hour,
                minute,
                self._attempts,
            )

            ok = False
            try:
                result = self._on_due(current, start)
                if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
                    result = await result
                ok = bool(result)
            except Exception as exc:  # noqa: BLE001 - 回调异常也要兜住
                logger.error("[%s] 每日总结执行失败: %s", self.plugin_name, exc, exc_info=True)

            if ok:
                self._last_date = today
                self._last_result = f"{today} 已完成"
                await self._emit_state()
            elif self._attempts >= MAX_ATTEMPTS_PER_DAY:
                self._last_date = today
                self._last_result = f"{today} 失败，已放弃（今日重试 {self._attempts} 次）"
                logger.warning(
                    "[%s] 每日总结今日已重试 %d 次仍未成功，放弃本轮。",
                    self.plugin_name,
                    self._attempts,
                )
                await self._emit_state()
            else:
                self._last_result = f"{today} 失败，稍后重试"
            return True
        except Exception as exc:  # noqa: BLE001 - 调度判定本身也不能炸
            logger.error("[%s] 每日总结判定异常: %s", self.plugin_name, exc)
            return False
