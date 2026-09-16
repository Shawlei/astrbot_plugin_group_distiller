"""每日定时总结（每日自动蒸馏）相关测试（v0.3.0）。

覆盖本轮新增能力：

- 时间解析：``23:00`` / ``23点30`` / ``2300`` / ``23`` 等写法，以及非法输入；
- 时钟判定：未启用不触发、未到点不触发、到点触发、同一天只触发一次；
- 当天补跑：设 23:00 但 23:40 才启动，也会立刻补一次；
- 持久化：上次完成日期落盘 / 恢复后不重复触发；
- 失败重试：可重试故障每天最多试 3 次，之后放弃；
- **核心语义**：每日总结用的是"当天全部对话"，包含已被盘中蒸馏标记过的消息，
  但绝不带上昨天的语料；
- 手动 ``/zl digest`` 与完成后的群内播报。

运行方式（任选其一）：

    python tests/test_daily_digest.py
    python -m pytest tests/test_daily_digest.py

所有临时文件、数据库都落在 ``tests/_tmp_dd/`` 内，运行结束自动清理。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TMP_ROOT = Path(__file__).resolve().parent / "_tmp_dd"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

PLUGIN_DIR = ROOT

from tests.test_qa_verify import (  # noqa: E402
    FakeEvent,
    _collect,
    _make_plugin,
    _text_of,
    install_fake_astrbot,
)

GID_A = "987654321"
GID_B = "555555555"
QQ_A = "123456789"
QQ_B = "222222222"


def _db(name: str) -> Path:
    """给每个用例一个独立的项目内临时库路径。"""
    path = TMP_ROOT / name
    if path.exists():
        path.unlink()
    return path


def ts_at(hour: int, minute: int, day_offset: int = 0) -> int:
    """构造"本地时间某天某时某分"的 Unix 时间戳。

    Args:
        hour: 小时。
        minute: 分钟。
        day_offset: 0 表示今天，-1 表示昨天。
    """
    local = time.localtime()
    return int(
        time.mktime(
            (
                local.tm_year,
                local.tm_mon,
                local.tm_mday + day_offset,
                hour,
                minute,
                0,
                0,
                0,
                -1,
            )
        )
    )


class RecordingProvider:
    """记录每次收到的提示词，便于断言"到底喂了什么语料"。"""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    async def text_chat(self, prompt: str = "", system_prompt: str = "", **_kw):
        self.prompts.append(prompt)
        return SimpleNamespace(completion_text=self.payload)


def _payload_for(text: str = "在校生") -> str:
    """造一份能通过解析的最小分析 JSON。"""
    return json.dumps(
        {"layer2_identity": [{"text": text, "confidence": 0.8}], "uncertainty": []},
        ensure_ascii=False,
    )


# =========================================================================== #
# 1. 时间解析与日历工具
# =========================================================================== #


def test_parse_hhmm_variants() -> None:
    """各种口语化时间写法都要认。"""
    from core.schedule import parse_hhmm

    assert parse_hhmm("23:00") == (23, 0)
    assert parse_hhmm("23：00") == (23, 0)      # 全角冒号
    assert parse_hhmm("2300") == (23, 0)
    assert parse_hhmm("23") == (23, 0)
    assert parse_hhmm("23点") == (23, 0)
    assert parse_hhmm("23点30") == (23, 30)
    assert parse_hhmm("23:5") == (23, 5)
    assert parse_hhmm("23:05") == (23, 5)
    assert parse_hhmm("930") == (9, 30)
    assert parse_hhmm("930时") == (9, 30)
    assert parse_hhmm("0:00") == (0, 0)
    assert parse_hhmm("6点5分") == (6, 5)
    assert parse_hhmm("  23:00  ") == (23, 0)

    for bad in ("", None, "abc", "24:00", "23:60", "-1:00", "25", "12345", "23:00:30"):
        assert parse_hhmm(bad) is None, f"{bad!r} 不该被解析成功"


def test_format_hhmm() -> None:
    """展示用格式化：合法值补零，非法值回落到默认时刻。"""
    from core.schedule import DEFAULT_TIME, format_hhmm

    assert format_hhmm("23:0") == "23:00"
    assert format_hhmm("9点5分") == "09:05"
    assert format_hhmm("2300") == "23:00"
    assert format_hhmm("") == DEFAULT_TIME
    assert format_hhmm("乱填") == "乱填"  # 解析不了就原样回显，方便用户发现自己填错了


def test_day_start_and_date_str() -> None:
    """当天零点时间戳必须是本地 00:00，且早于当前时间。"""
    from core.schedule import date_str, day_start_ts, minutes_of_day

    now = ts_at(15, 30)
    start = day_start_ts(now)
    assert time.localtime(start).tm_hour == 0
    assert time.localtime(start).tm_min == 0
    assert time.localtime(start).tm_mday == time.localtime(now).tm_mday
    assert start < now
    assert start == ts_at(0, 0)

    assert date_str(now) == time.strftime("%Y-%m-%d", time.localtime(now))
    assert minutes_of_day(now) == 15 * 60 + 30
    assert minutes_of_day(ts_at(0, 0)) == 0


# =========================================================================== #
# 2. 调度器判定
# =========================================================================== #


def _scheduler(settings: dict[str, Any], callback, name: str = "sched"):
    """构造一个调度器，返回 (scheduler, settings 引用, 调用记录)。

    ``callback`` 同步/异步都行，接收"本次是第几次触发"，返回 bool 表示
    是否处理完毕。
    """
    import inspect

    from core.schedule import DailyDigestScheduler

    box = {"settings": settings}
    calls: list[tuple[float, int]] = []

    async def on_due(now: float, day_start: int) -> bool:
        calls.append((now, day_start))
        result = callback(len(calls))
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    sched = DailyDigestScheduler(lambda: box["settings"], on_due, name)
    return sched, box, calls


def test_scheduler_disabled_never_fires() -> None:
    """开关关着，无论几点都不触发。"""

    async def scenario():
        sched, _box, calls = _scheduler(
            {"enabled": False, "time": "00:00"}, lambda _n: True
        )
        assert await sched.tick(ts_at(12, 0)) is False
        assert await sched.tick(ts_at(23, 59)) is False
        assert calls == []
        assert sched.last_date() == ""
        assert sched.seconds_until_next(ts_at(12, 0)) == -1

    asyncio.run(scenario())


def test_scheduler_before_time_no_fire() -> None:
    """没到点不触发。"""

    async def scenario():
        sched, _box, calls = _scheduler(
            {"enabled": True, "time": "23:00"}, lambda _n: True
        )
        assert await sched.tick(ts_at(0, 30)) is False
        assert await sched.tick(ts_at(22, 59)) is False
        assert calls == []

    asyncio.run(scenario())


def test_scheduler_fires_exactly_once_per_day() -> None:
    """到点触发，之后同一天不再触发。"""

    async def scenario():
        sched, _box, calls = _scheduler(
            {"enabled": True, "time": "23:00"}, lambda _n: True
        )
        assert await sched.tick(ts_at(23, 0)) is True
        assert len(calls) == 1
        assert sched.last_date() == time.strftime("%Y-%m-%d")

        # 同一天再怎么 tick 也不动了
        for minute in (1, 30, 59):
            assert await sched.tick(ts_at(23, minute)) is False
        assert len(calls) == 1

        # 第二天同样的时刻 → 再次触发
        assert await sched.tick(ts_at(23, 0, day_offset=1)) is True
        assert len(calls) == 2

    asyncio.run(scenario())


def test_scheduler_catches_up_after_downtime() -> None:
    """设 23:00、但 23:40 才启动，也要立刻补跑一次。"""

    async def scenario():
        sched, _box, calls = _scheduler(
            {"enabled": True, "time": "23:00"}, lambda _n: True
        )
        # 模拟"启动后第一次检查时钟"时已经过了设定时刻
        assert await sched.tick(ts_at(23, 40)) is True
        assert len(calls) == 1
        # 传进去的 day_start 必须是当天零点
        _now, day_start = calls[0]
        assert day_start == ts_at(0, 0)

    asyncio.run(scenario())


def test_scheduler_restore_prevents_refire() -> None:
    """持久化的完成日期被恢复后，同一天不再触发，并且状态变化会回传落盘。"""

    async def scenario():
        from core.schedule import date_str

        sched, _box, calls = _scheduler(
            {"enabled": True, "time": "23:00"}, lambda _n: True
        )
        saved: list[str] = []

        async def hook(value: str) -> None:
            saved.append(value)

        sched.set_state_hook(hook)
        sched.restore_last_date(date_str(ts_at(23, 10)))
        assert sched.last_date() == date_str(ts_at(23, 10))

        assert await sched.tick(ts_at(23, 30)) is False
        assert calls == []
        assert saved == [], "不该在没触发时也落盘"

        # 换到第二天 → 触发并把新日期回传
        assert await sched.tick(ts_at(23, 30, day_offset=1)) is True
        assert saved == [date_str(ts_at(23, 30, day_offset=1))]

    asyncio.run(scenario())


def test_scheduler_retry_then_give_up() -> None:
    """可重试的失败：当天最多试 3 次，之后放弃并记录日期。"""
    from core.schedule import MAX_ATTEMPTS_PER_DAY

    attempts = {"n": 0}

    async def failing(_now: float, _day_start: int) -> bool:
        attempts["n"] += 1
        return False

    async def scenario():
        from core.schedule import DailyDigestScheduler

        sched = DailyDigestScheduler(
            lambda: {"enabled": True, "time": "08:00"}, failing
        )
        try:
            for i in range(MAX_ATTEMPTS_PER_DAY):
                assert await sched.tick(ts_at(9, 0)) is True
                if i < MAX_ATTEMPTS_PER_DAY - 1:
                    assert sched.last_date() == "", "还没到放弃次数，不该记完成"

            assert attempts["n"] == MAX_ATTEMPTS_PER_DAY
            assert sched.last_date() == time.strftime("%Y-%m-%d"), "用尽重试后应记完成"
            assert "放弃" in sched.last_result()

            # 放弃之后当天不再触发
            assert await sched.tick(ts_at(10, 0)) is False
            assert attempts["n"] == MAX_ATTEMPTS_PER_DAY
        finally:
            await sched.stop()

    asyncio.run(scenario())


def test_scheduler_callback_exception_swallowed() -> None:
    """回调抛异常不能把调度器带崩，且算作"值得重试"。"""

    async def boom(_now: float, _day_start: int) -> bool:
        raise RuntimeError("模拟回调炸了")

    async def scenario():
        from core.schedule import DailyDigestScheduler

        sched = DailyDigestScheduler(lambda: {"enabled": True, "time": "08:00"}, boom)
        try:
            assert await sched.tick(ts_at(9, 0)) is True
            assert sched.last_date() == "", "异常应视为可重试"
            assert "重试" in sched.last_result()
        finally:
            await sched.stop()

    asyncio.run(scenario())


def test_scheduler_invalid_time_does_not_fire() -> None:
    """时间填成乱码 → 不触发、不崩溃。"""

    async def scenario():
        sched, _box, calls = _scheduler(
            {"enabled": True, "time": "半夜"}, lambda _n: True
        )
        assert await sched.tick(ts_at(23, 0)) is False
        assert calls == []

        sched2, _box2, calls2 = _scheduler(
            {"enabled": True, "time": ""}, lambda _n: True
        )
        assert await sched2.tick(ts_at(23, 0)) is False
        assert calls2 == []

    asyncio.run(scenario())


def test_scheduler_seconds_until_next() -> None:
    """倒计时：已过点算明天，未到点算今天。"""
    from core.schedule import DailyDigestScheduler

    sched = DailyDigestScheduler(lambda: {"enabled": True, "time": "23:00"}, None)  # type: ignore[arg-type]
    # 22:00 → 距 23:00 还有 1 小时
    assert sched.seconds_until_next(ts_at(22, 0)) == 3600
    # 23:30 → 今天的点已过，应算到明天 23:00，即 23.5 小时
    assert sched.seconds_until_next(ts_at(23, 30)) == int(23.5 * 3600)


# =========================================================================== #
# 3. 每日总结的语义（当天全部对话）
# =========================================================================== #


def test_digest_day_uses_whole_day_only() -> None:
    """核心语义：只喂"当天"的语料，且包含当天已被标记为已蒸馏的消息。"""
    from core.distiller import Distiller
    from core.storage import MessageRecord, Storage
    from core.targets import TargetSpec

    async def scenario():
        st = Storage(_db("digest_day.db"))
        await st.init()

        today_morning = ts_at(9, 0)
        today_noon = ts_at(12, 0)
        yesterday = ts_at(20, 0, day_offset=-1)

        # 昨天：不该进本次总结
        await st.insert_message(
            MessageRecord("y1", GID_A, QQ_A, "老王", "昨天说的话", timestamp=yesterday)
        )
        # 今天：一条正常，一条**已经被蒸馏过**
        await st.insert_message(
            MessageRecord("t1", GID_A, QQ_A, "老王", "今天上午说的", timestamp=today_morning)
        )
        await st.insert_message(
            MessageRecord("t2", GID_A, QQ_A, "老王", "今天中午说的已蒸过", timestamp=today_noon)
        )
        # 把"今天中午"那条标记为已蒸馏，验证总结仍会重新吃它
        rows = await st.fetch_undistilled(GID_A, QQ_A, 100)
        await st.mark_distilled(
            [r["id"] for r in rows if "今天中午" in str(r["content"])]
        )
        assert await st.count_undistilled(GID_A, QQ_A) == 2, "应只剩 2 条未蒸馏"

        provider = RecordingProvider(_payload_for())
        state = _state_with(GID_A, QQ_A, "老王")
        d = Distiller(st, {"distill_batch_messages": 300}, state, _context_with(provider))

        now = ts_at(23, 0)
        ok, detail = await d.digest_day(
            "umo", TargetSpec(GID_A, QQ_A, "老王"), ts_at(0, 0), now
        )
        assert ok is True, detail
        assert len(provider.prompts) == 1, "应该只调一次 LLM"

        prompt = provider.prompts[0]
        assert "今天上午说的" in prompt, "当天的语料没喂进去"
        assert "今天中午说的已蒸过" in prompt, "已被盘中蒸馏过的当天消息也应重新过一遍"
        assert "昨天说的话" not in prompt, "昨天的语料不该进当日总结"

        snap = await st.get_persona(GID_A, QQ_A)
        assert snap is not None and snap["meta"]["distill_round"] == 1
        await st.close()

    asyncio.run(scenario())


def test_digest_day_skips_when_below_threshold() -> None:
    """当天语料少于阈值 → 跳过，且不调 LLM。"""
    from core.distiller import Distiller
    from core.storage import MessageRecord, Storage
    from core.targets import TargetSpec

    async def scenario():
        st = Storage(_db("digest_min.db"))
        await st.init()
        await st.insert_message(
            MessageRecord("t1", GID_A, QQ_A, "老王", "只说了两句", timestamp=ts_at(9, 0))
        )
        provider = RecordingProvider(_payload_for())
        d = Distiller(st, {}, _state_with(GID_A, QQ_A, "老王"), _context_with(provider))

        ok, detail = await d.digest_day(
            "umo", TargetSpec(GID_A, QQ_A, "老王"), ts_at(0, 0), ts_at(23, 0), min_messages=5
        )
        assert ok is True, "跳过不算失败，不该让调度器重试"
        assert "低于阈值" in detail, detail
        assert provider.prompts == [], "低于阈值不该调 LLM"
        assert await st.get_persona(GID_A, QQ_A) is None
        await st.close()

    asyncio.run(scenario())


def test_digest_day_no_messages() -> None:
    """当天没说话 → 安静跳过。"""
    from core.distiller import Distiller
    from core.storage import Storage
    from core.targets import TargetSpec

    async def scenario():
        st = Storage(_db("digest_empty.db"))
        await st.init()
        provider = RecordingProvider(_payload_for())
        d = Distiller(st, {}, _state_with(GID_A, QQ_A, "老王"), _context_with(provider))

        ok, detail = await d.digest_day(
            "umo", TargetSpec(GID_A, QQ_A, "老王"), ts_at(0, 0), ts_at(23, 0)
        )
        assert ok is True and "没有语料" in detail, detail
        assert provider.prompts == []
        await st.close()

    asyncio.run(scenario())


def test_digest_day_llm_failure_is_retryable() -> None:
    """LLM 不可用 → 判定为可重试，且语料不被标记为已蒸馏。"""
    from core.distiller import Distiller
    from core.storage import MessageRecord, Storage
    from core.targets import TargetSpec

    async def scenario():
        st = Storage(_db("digest_fail.db"))
        await st.init()
        await st.insert_message(
            MessageRecord("t1", GID_A, QQ_A, "老王", "今天说的话", timestamp=ts_at(10, 0))
        )

        class NoProvider:
            pass

        class Ctx:
            async def get_using_provider_async(self, **_kw):
                return None

        d = Distiller(st, {}, _state_with(GID_A, QQ_A, "老王"), Ctx())
        ok, detail = await d.digest_day(
            "umo", TargetSpec(GID_A, QQ_A, "老王"), ts_at(0, 0), ts_at(23, 0)
        )
        assert ok is False, "拿不到 Provider 应判定为可重试"
        assert "Provider" in detail, detail
        assert await st.get_persona(GID_A, QQ_A) is None
        # 语料没被标记，下一轮还能重来
        assert await st.count_undistilled(GID_A, QQ_A) == 1
        await st.close()

    asyncio.run(scenario())


def test_run_daily_digest_all_targets() -> None:
    """run_daily_digest 会逐个目标处理，并给每个目标用它自己的会话来源。"""
    from core.distiller import Distiller
    from core.storage import MessageRecord, Storage
    from core.targets import TargetSpec

    def _state_with(*specs):
        from core.collector import RuntimeState

        return RuntimeState(enabled=True, listen_enabled=True, targets=list(specs))

    async def scenario():
        st = Storage(_db("digest_all.db"))
        await st.init()
        for i, (gid, qq, name) in enumerate(
            ((GID_A, QQ_A, "老王"), (GID_B, QQ_B, "小李"))
        ):
            await st.insert_message(
                MessageRecord(
                    f"m{i}", gid, qq, name, f"{name}今天说的话", timestamp=ts_at(10, i)
                )
            )

        provider = RecordingProvider(_payload_for())
        state = _state_with(
            TargetSpec(GID_A, QQ_A, "老王"), TargetSpec(GID_B, QQ_B, "小李")
        )
        d = Distiller(st, {}, state, _context_with(provider))

        seen: list[str] = []

        def umo_for(spec) -> str:
            seen.append(spec.key)
            return f"umo:{spec.group_id}"

        ok, detail = await d.run_daily_digest(umo_for, ts_at(0, 0), ts_at(23, 0))
        assert ok is True, detail
        assert seen == [f"{GID_A}:{QQ_A}", f"{GID_B}:{QQ_B}"], seen
        # 两个目标各自成档
        assert await st.get_persona(GID_A, QQ_A) is not None
        assert await st.get_persona(GID_B, QQ_B) is not None
        assert len(provider.prompts) == 2
        await st.close()

    asyncio.run(scenario())


def test_run_daily_digest_without_targets() -> None:
    """没有目标 → 处理完毕（不需要重试），也不该报错。"""
    from core.collector import RuntimeState
    from core.distiller import Distiller
    from core.storage import Storage

    async def scenario():
        st = Storage(_db("digest_notarget.db"))
        await st.init()
        d = Distiller(st, {}, RuntimeState(), _context_with(None))
        ok, detail = await d.run_daily_digest("umo", ts_at(0, 0), ts_at(23, 0))
        assert ok is True and "没有配置" in detail, detail
        await st.close()

    asyncio.run(scenario())


def test_run_daily_digest_partial_failure_retries() -> None:
    """一个目标失败 → 整体判定需要重试，但其它目标照常完成。"""
    from core.collector import RuntimeState
    from core.distiller import Distiller
    from core.storage import MessageRecord, Storage
    from core.targets import TargetSpec

    async def scenario():
        st = Storage(_db("digest_partial.db"))
        await st.init()
        await st.insert_message(
            MessageRecord("a", GID_A, QQ_A, "老王", "老王的话", timestamp=ts_at(9, 0))
        )
        # 给 B 目标塞一条"坏"语料：把 payload 弄成无法解析的 JSON
        await st.insert_message(
            MessageRecord("b", GID_B, QQ_B, "小李", "小李的话", timestamp=ts_at(9, 1))
        )

        provider = RecordingProvider(_payload_for())
        d = Distiller(
            st,
            {},
            RuntimeState(
                enabled=True,
                targets=[TargetSpec(GID_A, QQ_A, "老王"), TargetSpec(GID_B, QQ_B, "小李")],
            ),
            _context_with(provider),
        )

        # 让第二个目标触发解析失败
        real_chat = d._chat

        async def flaky_chat(prov, system_prompt, user_prompt):
            if "小李" in user_prompt:
                return "这不是 JSON"
            return await real_chat(prov, system_prompt, user_prompt)

        d._chat = flaky_chat  # type: ignore[assignment]

        ok, detail = await d.run_daily_digest(
            lambda spec: f"umo:{spec.group_id}", ts_at(0, 0), ts_at(23, 0)
        )
        assert ok is False, "有一个目标失败就该允许重试"
        assert await st.get_persona(GID_A, QQ_A) is not None, "成功的目标不该被连坐"
        assert "小李" in detail and "老王" in detail, detail
        await st.close()

    asyncio.run(scenario())


def test_storage_range_queries() -> None:
    """区间查询：只取区间内、且上下文归属正确。"""
    from core.storage import MessageRecord, Storage

    async def scenario():
        st = Storage(_db("range.db"))
        await st.init()
        await st.insert_message(
            MessageRecord("a", GID_A, QQ_A, "老王", "区间内", timestamp=100)
        )
        await st.insert_message(
            MessageRecord("b", GID_A, QQ_A, "老王", "区间前", timestamp=50)
        )
        await st.insert_message(
            MessageRecord("c", GID_A, QQ_A, "老王", "区间后", timestamp=300)
        )
        await st.insert_message(
            MessageRecord(
                "ctx_a", GID_A, "555", "路人",
                "给老王的上下文", timestamp=101, is_context=True, context_for_qq=QQ_A,
            )
        )
        await st.insert_message(
            MessageRecord(
                "ctx_b", GID_A, "555", "路人",
                "给小李的上下文", timestamp=102, is_context=True, context_for_qq=QQ_B,
            )
        )

        rows = await st.fetch_range(GID_A, QQ_A, 100, 200, 50)
        contents = [r["content"] for r in rows]
        assert "区间内" in contents
        assert "区间前" not in contents and "区间后" not in contents
        assert "给老王的上下文" in contents
        assert "给小李的上下文" not in contents, "别人目标的上下文不该混进来"

        assert await st.count_range(GID_A, QQ_A, 100, 200) == 1  # 只数目标自己的话
        assert await st.count_range(GID_A, QQ_A, 0, 99) == 1     # 只覆盖到"区间前"那条
        assert await st.count_range(GID_A, QQ_A, 51, 99) == 0
        await st.close()

    asyncio.run(scenario())


# =========================================================================== #
# 4. 与插件主类的集成
# =========================================================================== #


def _state_with(gid: str, qq: str, nickname: str = ""):
    """构造一个只含单个目标的 RuntimeState。"""
    from core.collector import RuntimeState

    return RuntimeState(enabled=True, listen_enabled=True, group_id=gid, qq_id=qq, nickname=nickname)


def _context_with(provider) -> Any:
    """构造一个能吐 provider 的最小 Context。"""

    class Ctx:
        async def get_using_provider_async(self, **_kw):
            return provider

    return Ctx()


def test_daily_cfg_defaults_and_normalization() -> None:
    """配置规整：默认关闭、时间归一、阈值与播报开关。"""

    async def scenario():
        install_fake_astrbot()
        plugin, _ = await _make_plugin(_db("cfg.db"), admin_only=False)

        default = plugin._daily_cfg()
        assert default["enabled"] is False
        assert default["display_time"] == "23:00"
        assert default["min_messages"] == 5
        assert default["notify"] is False

        plugin.config["daily_digest"] = {
            "enabled": True,
            "time": "9点5分",
            "min_messages": 0,
            "notify": True,
        }
        cfg = plugin._daily_cfg()
        assert cfg["enabled"] is True
        assert cfg["display_time"] == "09:05"
        assert cfg["min_messages"] == 0
        assert cfg["notify"] is True

        # 垃圾配置也不能崩
        plugin.config["daily_digest"] = "not-a-dict"
        fallback = plugin._daily_cfg()
        assert fallback["enabled"] is False

        await plugin.storage.close()

    asyncio.run(scenario())


def test_scheduler_lifecycle_with_plugin() -> None:
    """启用了才启动调度器；terminate 时要停掉。"""

    async def scenario():
        install_fake_astrbot()

        plugin, _ = await _make_plugin(_db("life_off.db"), admin_only=False)
        await plugin.initialize()
        assert plugin.scheduler.is_running() is False, "没启用不该起后台任务"
        await plugin.terminate()
        assert plugin.scheduler.is_running() is False

        plugin2, _ = await _make_plugin(
            _db("life_on.db"),
            admin_only=False,
            daily_digest={"enabled": True, "time": "23:30"},
        )
        await plugin2.initialize()
        assert plugin2.scheduler.is_running() is True, "启用了就该起后台任务"
        await plugin2.terminate()
        assert plugin2.scheduler.is_running() is False, "terminate 必须停掉调度器"

    asyncio.run(scenario())


def test_daily_digest_date_persisted_and_restored() -> None:
    """完成日期要落库，重启后能恢复，避免当天重复触发。"""

    async def scenario():
        install_fake_astrbot()
        from core.schedule import date_str
        from core.storage import MessageRecord

        db = _db("persist.db")
        plugin, _ = await _make_plugin(
            db,
            admin_only=False,
            daily_digest={"enabled": True, "time": "08:00", "min_messages": 0},
            targets=[{"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"}],
            provider=RecordingProvider(_payload_for("在校生")),
        )
        await plugin._load_runtime_state()
        await plugin.storage.insert_message(
            MessageRecord("m1", GID_A, QQ_A, "老王", "今天说的", timestamp=int(time.time()))
        )

        today = date_str()
        # 走真实的调度路径：tick 触发 → 回调干活 → 状态钩子落盘
        assert await plugin.scheduler.tick(ts_at(23, 30)) is True
        assert plugin.scheduler.last_date() == today
        assert await plugin.storage.get_state("rt_digest_last_date") == today
        assert await plugin.storage.get_persona(GID_A, QQ_A) is not None
        await plugin.storage.close()

        # 模拟重启：同一个库，新的插件实例
        plugin2, _ = await _make_plugin(
            db,
            admin_only=False,
            daily_digest={"enabled": True, "time": "08:00"},
            targets=[{"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"}],
        )
        restored = await plugin2.storage.get_state("rt_digest_last_date")
        assert restored == today, restored
        plugin2.scheduler.restore_last_date(restored)
        assert plugin2.scheduler.last_date() == today
        assert await plugin2.scheduler.tick(ts_at(23, 40)) is False, "当天不该再触发"
        await plugin2.storage.close()

    asyncio.run(scenario())


def test_zl_digest_command_and_report() -> None:
    """``/zl digest`` 立刻回执，后台跑完再用主动消息回报结果。"""

    async def scenario():
        install_fake_astrbot()
        import main as plugin_main

        plugin, _ = await _make_plugin(
            _db("cmd.db"),
            admin_only=False,
            provider=RecordingProvider(_payload_for("在校生")),
            targets=[{"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"}],
        )
        await plugin._load_runtime_state()

        sent: list[tuple[str, str]] = []

        class FakeChain:
            def __init__(self) -> None:
                self.parts: list[str] = []

            def message(self, text: str):
                self.parts.append(text)
                return self

        async def fake_send(umo: str, chain) -> None:
            sent.append((umo, "".join(getattr(chain, "parts", []))))

        plugin.context.send_message = fake_send  # type: ignore[attr-defined]
        original = plugin_main.MessageChain
        plugin_main.MessageChain = FakeChain  # type: ignore[assignment]

        try:
            from core.storage import MessageRecord

            await plugin.storage.insert_message(
                MessageRecord("m1", GID_A, QQ_A, "老王", "今天聊了挺多", timestamp=int(time.time()))
            )

            out = await _collect(
                plugin.zl(FakeEvent("/zl digest", group_id=GID_A, sender_id="10001"))
            )
            assert "已开始" in _text_of(out[0]), _text_of(out[0])

            # 等后台任务跑完
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            assert await plugin.storage.get_persona(GID_A, QQ_A) is not None
            assert sent, "跑完应当主动回报结果"
            assert "每日总结" in sent[0][1], sent

            # 没有目标时应给出引导
            plugin.state.clear_targets()
            out2 = await _collect(
                plugin.zl(FakeEvent("/zl digest", group_id=GID_A, sender_id="10001"))
            )
            assert "还没有设定目标" in _text_of(out2[0]), _text_of(out2[0])
        finally:
            plugin_main.MessageChain = original  # type: ignore[assignment]
            await plugin.storage.close()

    asyncio.run(scenario())


def test_daily_notify_broadcasts_to_each_group() -> None:
    """notify=True 时，总结结果会播报到每个目标所在的群。"""

    async def scenario():
        install_fake_astrbot()
        import main as plugin_main

        plugin, _ = await _make_plugin(
            _db("notify.db"),
            admin_only=False,
            provider=RecordingProvider(_payload_for("在校生")),
            daily_digest={"enabled": True, "time": "08:00", "notify": True, "min_messages": 0},
            targets=[
                {"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"},
                {"group_id": GID_B, "qq_id": QQ_B, "nickname": "小李"},
            ],
        )
        await plugin._load_runtime_state()

        sent: list[tuple[str, str]] = []

        class FakeChain:
            def __init__(self) -> None:
                self.parts: list[str] = []

            def message(self, text: str):
                self.parts.append(text)
                return self

        async def fake_send(umo: str, chain) -> None:
            sent.append((umo, "".join(getattr(chain, "parts", []))))

        plugin.context.send_message = fake_send  # type: ignore[attr-defined]
        original = plugin_main.MessageChain
        plugin_main.MessageChain = FakeChain  # type: ignore[assignment]

        try:
            from core.storage import MessageRecord

            now = int(time.time())
            await plugin.storage.insert_message(
                MessageRecord("a", GID_A, QQ_A, "老王", "老王今天的话", timestamp=now)
            )
            await plugin.storage.insert_message(
                MessageRecord("b", GID_B, QQ_B, "小李", "小李今天的话", timestamp=now)
            )

            assert await plugin._run_daily_digest(time.time(), int(ts_at(0, 0))) is True
            assert len(sent) == 2, f"每个群应各播报一条，实际 {sent}"
            by_group = {umo: text for umo, text in sent}
            assert any(GID_A in u for u in by_group), by_group
            assert any(GID_B in u for u in by_group), by_group

            text_a = next(t for u, t in by_group.items() if GID_A in u)
            text_b = next(t for u, t in by_group.items() if GID_B in u)
            assert "今日蒸馏总结完成" in text_a, text_a
            # 关键：A 群只该看到 A 群目标的结果，不该串到 B 群的小李
            assert "老王" in text_a and "小李" not in text_a, text_a
            assert "小李" in text_b and "老王" not in text_b, text_b
        finally:
            plugin_main.MessageChain = original  # type: ignore[assignment]
            await plugin.storage.close()

    asyncio.run(scenario())


def test_daily_notify_off_by_default() -> None:
    """默认不播报，免得不小心刷群。"""

    async def scenario():
        install_fake_astrbot()
        import main as plugin_main

        plugin, _ = await _make_plugin(
            _db("notify_off.db"),
            admin_only=False,
            provider=RecordingProvider(_payload_for()),
            daily_digest={"enabled": True, "time": "08:00", "min_messages": 0},
            targets=[{"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"}],
        )
        await plugin._load_runtime_state()

        sent: list[Any] = []

        async def fake_send(umo, chain):
            sent.append(umo)

        plugin.context.send_message = fake_send  # type: ignore[attr-defined]
        original = plugin_main.MessageChain
        plugin_main.MessageChain = object  # type: ignore[assignment]

        try:
            from core.storage import MessageRecord

            await plugin.storage.insert_message(
                MessageRecord("a", GID_A, QQ_A, "老王", "今天的话", timestamp=int(time.time()))
            )
            await plugin._run_daily_digest(time.time(), int(ts_at(0, 0)))
            assert sent == [], "notify 关着却发了消息"
        finally:
            plugin_main.MessageChain = original  # type: ignore[assignment]
            await plugin.storage.close()

    asyncio.run(scenario())


def test_panel_shows_daily_digest_line() -> None:
    """面板：启用后展示每日总结时刻，未启用则不显示。"""

    async def scenario():
        install_fake_astrbot()
        plugin, _ = await _make_plugin(
            _db("panel_daily.db"),
            admin_only=False,
            targets=[{"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"}],
        )
        await plugin._load_runtime_state()

        panel_off = await plugin._build_panel()
        assert "每日总结" not in panel_off, "没启用不该显示这一行"

        plugin.config["daily_digest"] = {"enabled": True, "time": "23:0"}
        panel_on = await plugin._build_panel()
        assert "⏰ 每日总结" in panel_on, panel_on
        assert "23:00" in panel_on, panel_on

        # 帮助里要提到 digest 指令
        from core import progress

        assert "/zl digest" in progress.render_help()

        await plugin.storage.close()

    asyncio.run(scenario())


def test_admin_guard_covers_digest() -> None:
    """管理员开关对 /zl digest 同样生效。"""

    async def scenario():
        install_fake_astrbot()
        plugin, _ = await _make_plugin(
            _db("admin_digest.db"),
            admin_only=True,
            daily_digest={"enabled": True, "time": "23:00"},
            targets=[{"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"}],
        )

        class NonAdmin(FakeEvent):
            def is_admin(self) -> bool:
                return False

        out = await _collect(
            plugin.zl(NonAdmin("/zl digest", group_id=GID_A, sender_id="10001"))
        )
        assert "仅管理员" in _text_of(out[0]), _text_of(out[0])
        await plugin.storage.close()

    asyncio.run(scenario())


def main() -> int:
    """直接运行时的入口，逐个执行测试函数。"""
    tests = [
        test_parse_hhmm_variants,
        test_format_hhmm,
        test_day_start_and_date_str,
        test_scheduler_disabled_never_fires,
        test_scheduler_before_time_no_fire,
        test_scheduler_fires_exactly_once_per_day,
        test_scheduler_catches_up_after_downtime,
        test_scheduler_restore_prevents_refire,
        test_scheduler_retry_then_give_up,
        test_scheduler_callback_exception_swallowed,
        test_scheduler_invalid_time_does_not_fire,
        test_scheduler_seconds_until_next,
        test_digest_day_uses_whole_day_only,
        test_digest_day_skips_when_below_threshold,
        test_digest_day_no_messages,
        test_digest_day_llm_failure_is_retryable,
        test_run_daily_digest_all_targets,
        test_run_daily_digest_without_targets,
        test_run_daily_digest_partial_failure_retries,
        test_storage_range_queries,
        test_daily_cfg_defaults_and_normalization,
        test_scheduler_lifecycle_with_plugin,
        test_daily_digest_date_persisted_and_restored,
        test_zl_digest_command_and_report,
        test_daily_notify_broadcasts_to_each_group,
        test_daily_notify_off_by_default,
        test_panel_shows_daily_digest_line,
        test_admin_guard_covers_digest,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}\n      -> {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}\n      -> {exc!r}")
        else:
            print(f"PASS  {test.__name__}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)
        shutil.rmtree(PLUGIN_DIR / "data_local", ignore_errors=True)
