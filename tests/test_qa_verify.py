"""QA 独立验证测试（不依赖 AstrBot 本体，也不依赖 pytest）。

用途：证明 ``astrbot_plugin_group_distiller`` 真的能工作，而不仅是文件存在。
覆盖作者 ``tests/test_core.py`` 未覆盖的边界：
进度条极限、/zl 指令族分派、采集过滤、去重、增量 merge、落盘路径、
sqlite 跨线程、监听器纯净性、事件循环阻塞、Lock 正确性、权限、导出路径、terminate 清理，
以及一条 mock 事件驱动的端到端链路。

运行方式（任选其一）：

    python tests/test_qa_verify.py
    python -m pytest tests/test_qa_verify.py      # 若环境有 pytest

所有临时文件/数据库都落在 ``tests/_qa_tmp/`` 之内，运行结束自动清理。
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sqlite3
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TMP_ROOT = Path(__file__).resolve().parent / "_qa_tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

PLUGIN_DIR = ROOT


# =========================================================================== #
# 测试基建：伪造 astrbot 依赖 + 伪造事件
# =========================================================================== #


def install_fake_astrbot(data_path_override: Path | None = None) -> None:
    """把伪造的 ``astrbot`` 系列模块注入 sys.modules。

    Args:
        data_path_override: 若给定，则同时伪造
            ``astrbot.core.utils.astrbot_path.get_astrbot_data_path`` 返回该路径，
            用于验证真实 AstrBot 环境下的落盘位置。
    """
    astrbot = types.ModuleType("astrbot")
    astrbot.__path__ = []  # type: ignore[attr-defined]
    api = types.ModuleType("astrbot.api")
    api.__path__ = []  # type: ignore[attr-defined]

    noop = lambda *a, **k: None  # noqa: E731
    api.logger = types.SimpleNamespace(
        info=noop, error=noop, warning=noop, debug=noop, critical=noop
    )

    class AstrBotConfig(dict):
        """AstrBot 配置对象的最小替身。"""

    api.AstrBotConfig = AstrBotConfig

    # --- astrbot.api.event ---
    event_mod = types.ModuleType("astrbot.api.event")

    class EventMessageType:
        ALL = "EventMessageType.ALL"

    class PlatformAdapterType:
        AIOCQHTTP = "PlatformAdapterType.AIOCQHTTP"

    class PermissionType:
        ADMIN = "PermissionType.ADMIN"

    def _passthrough_decorator(*_a, **_k):
        def deco(fn):
            return fn

        return deco

    filter = types.SimpleNamespace(
        command=_passthrough_decorator,
        event_message_type=_passthrough_decorator,
        platform_adapter_type=_passthrough_decorator,
        permission_type=_passthrough_decorator,
        EventMessageType=EventMessageType,
        PlatformAdapterType=PlatformAdapterType,
        PermissionType=PermissionType,
    )

    class AstrMessageEvent:
        """最小事件替身（仅占位，真实测试用 FakeEvent）。"""

    event_mod.filter = filter
    event_mod.AstrMessageEvent = AstrMessageEvent

    # --- astrbot.api.star ---
    star_mod = types.ModuleType("astrbot.api.star")

    class Star:
        def __init__(self, context: Any = None) -> None:
            self.context = context

    class Context:
        pass

    class StarMetadata:
        pass

    def register(*_a, **_k):
        def deco(cls):
            return cls

        return deco

    star_mod.Star = Star
    star_mod.Context = Context
    star_mod.register = register
    star_mod.register_star = register

    # --- astrbot.api.message_components ---
    mc = types.ModuleType("astrbot.api.message_components")

    class At:
        def __init__(self, qq: Any = None, **_k) -> None:
            self.qq = qq

    class Plain:
        def __init__(self, text: str = "") -> None:
            self.text = text

    mc.At = At
    mc.Plain = Plain

    api.event = event_mod
    api.star = star_mod
    api.message_components = mc
    astrbot.api = api

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.event"] = event_mod
    sys.modules["astrbot.api.star"] = star_mod
    sys.modules["astrbot.api.message_components"] = mc

    # --- 伪造 astrbot.core.utils.astrbot_path（默认指向项目内，避免污染真实盘符）---
    if True:
        if data_path_override is None:
            data_path_override = TMP_ROOT / "astrbot_data"
        core = types.ModuleType("astrbot.core")
        core.__path__ = []  # type: ignore[attr-defined]
        utils = types.ModuleType("astrbot.core.utils")
        utils.__path__ = []  # type: ignore[attr-defined]
        path_mod = types.ModuleType("astrbot.core.utils.astrbot_path")
        path_mod.get_astrbot_data_path = lambda: str(data_path_override)  # noqa: E731
        utils.astrbot_path = path_mod
        core.utils = utils
        astrbot.core = core
        sys.modules["astrbot.core"] = core
        sys.modules["astrbot.core.utils"] = utils
        sys.modules["astrbot.core.utils.astrbot_path"] = path_mod


class FakeEvent:
    """伪 AstrMessageEvent，字段与 AstrBot 真机对齐。"""

    def __init__(
        self,
        message_str: str = "",
        *,
        group_id: str | None = None,
        sender_id: str = "10001",
        sender_name: str = "群友",
        admin: bool = False,
        message_id: str | None = None,
        timestamp: int | None = None,
        umo: str = "aiocqhttp:GroupMessage:987654321",
    ) -> None:
        self.message_str = message_str
        self.unified_msg_origin = umo
        self._group_id = group_id
        self._sender_id = sender_id
        self._sender_name = sender_name
        self._admin = admin
        self.message_obj = types.SimpleNamespace(
            message_id=message_id, timestamp=timestamp
        )

    def get_group_id(self):
        return self._group_id

    def get_sender_id(self):
        return self._sender_id

    def get_sender_name(self):
        return self._sender_name

    def is_admin(self):
        return self._admin

    def get_message_outline(self):
        return self.message_str

    def get_message_str(self):
        return self.message_str

    def plain_result(self, text: str):
        return ("plain", text)

    def chain_result(self, chain):
        ats, texts = [], []
        for seg in chain:
            if hasattr(seg, "qq"):
                ats.append(seg.qq)
            if hasattr(seg, "text"):
                texts.append(seg.text)
        return ("chain", ats, "".join(texts))


class FakeContext:
    """伪 AstrBot Context。"""

    def __init__(self, provider: Any = None) -> None:
        self._provider = provider
        self.sent: list[Any] = []

    def get_provider_by_id(self, provider_id: str):
        return None

    async def get_using_provider_async(self, umo: str | None = None):
        return self._provider

    async def send_message(self, umo, chain):
        self.sent.append((umo, chain))
        return True


class FakeProvider:
    """伪 LLM Provider，返回固定 JSON，绝不联网。"""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    async def text_chat(self, prompt: str = "", system_prompt: str = "", **_k):
        self.calls += 1
        return types.SimpleNamespace(completion_text=self.payload)


def _text_of(result: Any) -> str:
    """从 _reply 产出的结果里取出纯文本。"""
    if isinstance(result, tuple):
        if result and result[0] == "plain":
            return result[1]
        if result and result[0] == "chain":
            return result[2]
    return str(result)


def _base_config(**over: Any) -> dict:
    """默认配置：关闭 @、关闭自动蒸馏、不记上下文，便于确定性断言。"""
    cfg = {
        "enabled": True,
        "listen_enabled": True,
        "auto_distill": False,
        "distill_interval_messages": 200,
        "distill_batch_messages": 300,
        "saturation_messages": 1500,
        "min_message_length": 1,
        "ignore_commands": True,
        "record_context": False,
        "llm_provider_id": "",
        "admin_only": False,
        "reply_with_at": False,
        "custom_prompt_extra": "",
        "target_group_id": "",
        "target_qq_id": "",
        "target_nickname": "",
    }
    cfg.update(over)
    return cfg


async def _make_plugin(db_path: Path, provider: Any = None, **cfg_over: Any):
    """构造插件并注入项目内临时数据库。"""
    import main as plugin_main  # noqa: E402

    config = _base_config(**cfg_over)
    ctx = FakeContext(provider)
    plugin = plugin_main.GroupDistillerPlugin(ctx, config)

    # 把 storage 重定向到项目内临时库，避免污染插件目录
    from core.storage import Storage  # noqa: E402

    plugin.storage = Storage(db_path)
    plugin.collector.storage = plugin.storage
    plugin.distiller.storage = plugin.storage
    await plugin.storage.init()
    return plugin, plugin_main


# =========================================================================== #
# A1. 进度条渲染边界
# =========================================================================== #


def test_progress_bar_extremes() -> None:
    from core import progress

    cases = [
        (0, 100),
        (100, 100),
        (50, 100),
        (150, 100),          # 超过饱和
        (1500, 1500),        # 恰好饱和
        (2_000_000, 1500),   # 百万级
        (-5, 100),           # 负数
        (7, 0),              # 饱和基准为 0
        (0, 0),
    ]
    for cur, tot in cases:
        bar = progress.make_bar(cur, tot)
        assert len(bar) == 10, f"进度条非 10 格: cur={cur} tot={tot} => {bar!r}"
        assert set(bar) <= {"█", "░"}, f"出现非法字符: {bar!r}"
        assert bar.count("█") <= 10, f"超过 10 格满格: {bar!r}"

    # 精确断言：0% / 100% / 50% / 溢出封顶 / 负数封底
    assert progress.make_bar(0, 100) == "░" * 10
    assert progress.make_bar(100, 100) == "█" * 10
    assert progress.make_bar(50, 100) == "█" * 5 + "░" * 5
    assert progress.make_bar(150, 100) == "█" * 10
    assert progress.make_bar(2_000_000, 1500) == "█" * 10
    assert progress.make_bar(-5, 100) == "░" * 10

    # 千分位
    assert progress.fmt_int(2_000_000) == "2,000,000"
    assert progress.fmt_int(1234) == "1,234"
    assert progress.fmt_int("bad") == "0"

    # 面板：超饱和百分比封顶 100，绝不出现 101%+
    data = progress.PanelData(
        has_target=True, nickname="张三", qq="123456789", group_id="987654321",
        total=2_000_000, saturation=1500,
    )
    panel = progress.render_panel(data)
    assert "2,000,000" in panel
    assert "100%" in panel
    assert "101%" not in panel and "133%" not in panel

    # 空数据面板不抛异常
    empty = progress.render_panel(
        progress.PanelData(has_target=True, qq="1", group_id="2", total=0, saturation=1500)
    )
    assert "[░" in empty and "0%" in empty


# =========================================================================== #
# A2. /zl 子命令解析（含兜底 / 非法输入 / 无斜杠 / 全角斜杠）
# =========================================================================== #


def test_parse_command_unit() -> None:
    install_fake_astrbot()
    import main as plugin_main

    pp = plugin_main._parse_command
    assert pp("/zl") == ("panel", "")
    assert pp("zl") == ("panel", "")
    assert pp("") == ("panel", "")
    assert pp("/zl help") == ("help", "")
    assert pp("zl help") == ("help", "")
    assert pp("／zl help") == ("help", "")
    assert pp("/zl set 987654321 123456789") == ("set", "987654321 123456789")
    assert pp("/zl set 123456789") == ("set", "123456789")
    assert pp("/zl 纠正 他不会这么说") == ("纠正", "他不会这么说")
    assert pp("／zl 蒸馏") == ("蒸馏", "")
    # 大写子命令应被小写化
    assert pp("/zl HELP") == ("help", "")


def test_zl_command_dispatch() -> None:
    """逐个跑子命令，断言不抛异常且返回合理文本。"""

    async def scenario():
        install_fake_astrbot()
        db = TMP_ROOT / "cmd.db"
        plugin, plugin_main = await _make_plugin(db, admin_only=False)
        # 先设定目标，使依赖 has_target 的子命令可走通
        await _collect(plugin.zl(FakeEvent("/zl set 987654321 123456789", group_id="987654321")))

        results: dict[str, str] = {}

        async def run(label: str, fe: FakeEvent):
            out = []
            try:
                async for r in plugin.zl(fe):
                    out.append(_text_of(r))
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(f"指令 {label!r} 抛异常: {exc!r}") from exc
            assert out, f"指令 {label!r} 无任何返回"
            results[label] = "\n".join(out)

        ev = lambda s, **k: FakeEvent(s, group_id="987654321", sender_id="10001", **k)  # noqa: E731

        await run("/zl", ev("/zl"))
        await run("/zl help", ev("/zl help"))
        await run("zl help", ev("zl help"))
        await run("／zl help", ev("／zl help"))
        await run("/zl set <群> <QQ>", ev("/zl set 987654321 123456789"))
        await run("/zl set abc 123(非法)", ev("/zl set abc 123"))
        await run("/zl set 1 2 3(带昵称)", ev("/zl set 987654321 123456789 老王"))
        await run("/zl set <QQ>(单参数)", ev("/zl set 222222222"))
        await run("/zl now", ev("/zl now"))
        await run("/zl 蒸馏", ev("/zl 蒸馏"))
        await run("/zl correct", ev("/zl correct 他不会这么说"))
        await run("/zl 纠正", ev("/zl 纠正 他不会这么说"))
        await run("/zl reset(无 confirm)", ev("/zl reset"))
        await run("/zl reset confirm", ev("/zl reset confirm"))
        await run("/zl export", ev("/zl export"))
        await run("/zl on", ev("/zl on"))
        await run("/zl off", ev("/zl off"))
        await run("/zl profile", ev("/zl profile"))
        await run("/zl 不存在的子命令", ev("/zl 不存在的子命令"))
        # 单参数 set 但不在群里 → 应友好提示
        await run("/zl set 单参数(不在群)", FakeEvent("/zl set 222222222", group_id=None))

        # 关键断言
        assert "群号必须是纯数字" in results["/zl set abc 123(非法)"]
        assert "用法" in results["/zl set <群> <QQ>"] or "目标已设定" in results["/zl set <群> <QQ>"]
        assert "目标已设定" in results["/zl set <群> <QQ>"]
        assert "confirm" in results["/zl reset(无 confirm)"]
        assert "已清空" in results["/zl reset confirm"]
        assert "未知子命令" in results["/zl 不存在的子命令"]
        assert "不在群里" in results["/zl set 单参数(不在群)"]
        assert "优先级最高" in results["/zl correct"]
        assert "优先级最高" in results["/zl 纠正"]
        assert "帮助" in results["/zl help"]
        # /zl 蒸馏 走手动触发分支
        assert "蒸馏" in results["/zl 蒸馏"]

        # 让触发的后台蒸馏任务收尾，避免悬挂任务
        await _wait_task(plugin.distiller)

        await plugin.storage.close()

    asyncio.run(scenario())


async def _collect(agen) -> list:
    out = []
    async for r in agen:
        out.append(r)
    return out


# =========================================================================== #
# A3. 采集过滤逻辑
# =========================================================================== #


def test_collector_filtering() -> None:
    from core.collector import Collector, RuntimeState
    from core.storage import Storage

    async def scenario():
        db = TMP_ROOT / "collector.db"
        st = Storage(db)
        await st.init()
        state = RuntimeState(
            enabled=True, listen_enabled=True, group_id="999", qq_id="123"
        )
        cfg = {"record_context": False, "min_message_length": 1, "ignore_commands": True}
        col = Collector(st, cfg, state)

        # 1) 非目标群 → 丢弃
        await col.handle_event(FakeEvent("目标在别的群说话", group_id="888", sender_id="123"))
        # 2) 非目标 QQ（无上下文模式）→ 丢弃
        await col.handle_event(FakeEvent("路人的话", group_id="999", sender_id="555"))
        # 3) 空消息（目标）→ 丢弃
        await col.handle_event(FakeEvent("", group_id="999", sender_id="123"))
        # 4) 超短消息（目标，min=1 时单字保留；此处先测单字保留）
        await col.handle_event(FakeEvent("好", group_id="999", sender_id="123"))
        # 5) 指令消息（目标，/ 开头）→ 丢弃
        await col.handle_event(FakeEvent("/zl help", group_id="999", sender_id="123"))
        # 6) 正常目标消息 → 保留
        await col.handle_event(FakeEvent("今天天气真不错啊", group_id="999", sender_id="123"))

        assert len(col._buffer) == 2, f"缓冲区条数异常: {len(col._buffer)}"
        contents = [r.content for r in col._buffer]
        assert "好" in contents and "今天天气真不错啊" in contents
        assert all(r.is_context is False for r in col._buffer)
        assert "路人的话" not in contents and "/zl help" not in contents

        # min_message_length=4 → 单字被过滤
        col2 = Collector(st, {**cfg, "min_message_length": 4}, RuntimeState(
            enabled=True, listen_enabled=True, group_id="999", qq_id="123"))
        await col2.handle_event(FakeEvent("好", group_id="999", sender_id="123"))
        await col2.handle_event(FakeEvent("这句话够长了吧", group_id="999", sender_id="123"))
        assert [r.content for r in col2._buffer] == ["这句话够长了吧"]

        # ignore_commands=False → 指令被保留
        col3 = Collector(st, {**cfg, "ignore_commands": False}, RuntimeState(
            enabled=True, listen_enabled=True, group_id="999", qq_id="123"))
        await col3.handle_event(FakeEvent("/zl help", group_id="999", sender_id="123"))
        assert [r.content for r in col3._buffer] == ["/zl help"]

        # 采集开关关闭 → 全部丢弃
        col4 = Collector(st, cfg, RuntimeState(
            enabled=True, listen_enabled=False, group_id="999", qq_id="123"))
        await col4.handle_event(FakeEvent("不该被采集", group_id="999", sender_id="123"))
        assert col4._buffer == []

        await st.close()

    asyncio.run(scenario())


def test_context_recording() -> None:
    """开启 record_context 时，目标消息前后各 1 条他人消息应被记为上下文。"""
    from core.collector import Collector, RuntimeState
    from core.storage import Storage

    async def scenario():
        db = TMP_ROOT / "ctx.db"
        st = Storage(db)
        await st.init()
        state = RuntimeState(enabled=True, listen_enabled=True, group_id="999", qq_id="123")
        cfg = {"record_context": True, "min_message_length": 1, "ignore_commands": True}
        col = Collector(st, cfg, state)

        await col.handle_event(FakeEvent("路人甲上文", group_id="999", sender_id="555", message_id="c1"))
        await col.handle_event(FakeEvent("目标发言", group_id="999", sender_id="123", message_id="t1"))
        await col.handle_event(FakeEvent("路人乙下文", group_id="999", sender_id="556", message_id="c2"))
        await col.flush()

        ctxs = await st.fetch_undistilled("999", "123", 100)
        contents = {r["content"]: r["is_context"] for r in ctxs}
        assert contents.get("目标发言") == 0
        assert contents.get("路人甲上文") == 1  # 上文
        assert contents.get("路人乙下文") == 1  # 下文
        assert await st.count_messages("999", "123") == 1  # 目标语料只算 1 条
        await st.close()

    asyncio.run(scenario())


# =========================================================================== #
# A4. 去重
# =========================================================================== #


def test_dedup_by_message_id() -> None:
    from core.collector import Collector, RuntimeState
    from core.storage import Storage

    async def scenario():
        db = TMP_ROOT / "dedup.db"
        st = Storage(db)
        await st.init()
        state = RuntimeState(enabled=True, listen_enabled=True, group_id="999", qq_id="123")
        col = Collector(st, {"record_context": False, "min_message_length": 1, "ignore_commands": True}, state)

        for _ in range(2):
            await col.handle_event(
                FakeEvent("同一条消息", group_id="999", sender_id="123", message_id="dup-1")
            )
        await col.flush()
        assert await st.count_messages("999", "123") == 1, "相同 message_id 未去重"
        await st.close()

    asyncio.run(scenario())


# =========================================================================== #
# A5. 增量 merge
# =========================================================================== #


def test_merge_incremental_and_corrections() -> None:
    from core.distiller import empty_snapshot, merge_snapshot

    base = empty_snapshot()
    round1 = merge_snapshot(
        base,
        {"layer3_style": [{"text": "喜欢用～结尾", "confidence": 0.9, "quotes": ["好呀～"]}]},
        corrections=["他是程序员"],
    )
    assert round1["meta"]["distill_round"] == 1
    assert round1["layers"]["layer3_style"][0]["evidence"] == 1
    assert round1["corrections"] == ["他是程序员"]

    # ① 新证据只补充不推翻：新增一条 + 重复一条
    round2 = merge_snapshot(
        round1,
        {
            "layer3_style": [
                {"text": "喜欢用～结尾", "confidence": 0.6},   # 重复，证据+1
                {"text": "爱用叠词", "confidence": 0.7},       # 新增
            ]
        },
        corrections=["他是程序员"],
    )
    layer = {i["text"]: i for i in round2["layers"]["layer3_style"]}
    assert len(round2["layers"]["layer3_style"]) == 2, "重复条目未被合并"
    assert layer["喜欢用～结尾"]["evidence"] == 2, "证据未累加"
    assert layer["喜欢用～结尾"]["confidence"] == 0.9, "高置信被低置信覆盖"
    assert "爱用叠词" in layer
    assert round2["meta"]["distill_round"] == 2

    # ② 冲突被标注而非覆盖
    conflict = merge_snapshot(
        round2,
        {"layer3_style": [{"text": "从不用句号", "confidence": 0.8, "conflict": True}]},
        corrections=["他是程序员"],
    )
    conflicted = [i for i in conflict["layers"]["layer3_style"] if i["text"] == "从不用句号"]
    assert conflicted and conflicted[0]["conflict"] is True
    # 原有条目仍在（没被推翻）
    assert any(i["text"] == "喜欢用～结尾" for i in conflict["layers"]["layer3_style"])

    # ③ 纠正层优先级最高、不被 LLM 推断覆盖
    assert conflict["corrections"] == ["他是程序员"]
    # 即便本轮 LLM 分析里塞一个 corrections 字段，也不该被采纳
    sneaky = merge_snapshot(
        conflict,
        {"layer3_style": [], "corrections": ["LLM 想覆盖纠正"]},
        corrections=["他是程序员"],
    )
    assert sneaky["corrections"] == ["他是程序员"], "纠正层被 LLM 输出污染"


# =========================================================================== #
# A6. 数据落盘位置
# =========================================================================== #


def test_data_dir_under_plugin_data() -> None:
    """真实 AstrBot 环境下 DB 必须落在 <data>/plugin_data/<plugin>/。"""
    fake_root = TMP_ROOT / "astrbot_root"
    fake_root.mkdir(parents=True, exist_ok=True)
    install_fake_astrbot(data_path_override=fake_root)

    import importlib

    import core.storage as storage_mod

    importlib.reload(storage_mod)
    got = storage_mod.resolve_plugin_data_dir()
    expected = fake_root / "plugin_data" / "astrbot_plugin_group_distiller"
    assert got == expected, f"落盘目录错误: {got} != {expected}"
    assert got.exists()

    # 默认 DB 文件名与目录拼接正确
    async def scenario():
        st = storage_mod.Storage()
        await st.init()
        p = Path(st._db_path)
        assert p.parent == expected, f"DB 不在 plugin_data 下: {p}"
        assert p.name == "distiller.db"
        await st.close()

    asyncio.run(scenario())


def test_data_dir_fallback_is_plugin_local() -> None:
    """无 AstrBot 环境下会回退到插件目录内 data_local —— 这是已知兜底行为。"""
    # 移除伪造的 astrbot 模块，触发 ImportError 兜底
    for name in list(sys.modules):
        if name == "astrbot" or name.startswith("astrbot."):
            sys.modules.pop(name, None)

    import importlib

    import core.storage as storage_mod

    importlib.reload(storage_mod)
    got = storage_mod.resolve_plugin_data_dir()
    assert got.exists()
    # 记录实际回退位置，供报告使用
    print(f"    [info] 无 AstrBot 时回退目录 = {got}")
    # 清理回退产生的目录，避免污染仓库/盘符
    if PLUGIN_DIR in got.parents:
        shutil.rmtree(got, ignore_errors=True)
    shutil.rmtree(PLUGIN_DIR / "data_local", ignore_errors=True)
    # 恢复伪造环境，供后续用例使用
    install_fake_astrbot()


# =========================================================================== #
# B1. sqlite 跨线程
# =========================================================================== #


def test_sqlite_check_same_thread_control_experiment() -> None:
    """对照实验：默认连接跨线程使用必然报 ProgrammingError。"""
    db = TMP_ROOT / "control.db"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(str(db))  # 默认 check_same_thread=True
    try:
        def _use(c):
            c.execute("SELECT 1")

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_use, conn)
            raised = None
            try:
                fut.result(timeout=10)
            except Exception as exc:  # noqa: BLE001
                raised = exc
        assert raised is not None, "对照实验未复现跨线程错误，测试无效"
        assert "thread" in str(raised).lower(), f"异常类型不符: {raised!r}"
    finally:
        conn.close()


def test_sqlite_storage_cross_thread_ok() -> None:
    """Storage 用 check_same_thread=False，跨线程执行不得抛 ProgrammingError。"""
    from core.storage import MessageRecord, Storage

    class RecordingStorage(Storage):
        def _init_sync(self):
            self._init_thread = threading.get_ident()
            super()._init_sync()

    async def scenario():
        db = TMP_ROOT / "xthread.db"
        if db.exists():
            db.unlink()
        st = RecordingStorage(db)
        await st.init()
        main_thread = threading.get_ident()
        # 连接在 to_thread 的工作线程里创建，与主线程必然不同
        assert st._init_thread != main_thread, "连接创建线程与主线程相同，无法验证跨线程"
        # 直接在主线程调用同步写方法 → 跨线程使用该连接
        ok = st._insert_message_sync(
            MessageRecord(message_id="x1", group_id="999", speaker_qq="123", content="跨线程")
        )
        assert ok is True, "跨线程写库失败"
        assert st._count_messages_sync("999", "123") == 1

        # 并发压力：200 次 to_thread 写库
        async def worker(i):
            return await st.insert_message(
                MessageRecord(message_id=f"m{i}", group_id="999", speaker_qq="123", content=f"c{i}")
            )
        results = await asyncio.gather(*[worker(i) for i in range(200)])
        assert all(results)
        assert await st.count_messages("999", "123") == 201
        await st.close()

    asyncio.run(scenario())


# =========================================================================== #
# B2/B3. 监听器纯净性 + 事件循环阻塞
# =========================================================================== #


def test_listener_is_pure_coroutine() -> None:
    install_fake_astrbot()
    import ast
    import inspect
    import textwrap

    import main as plugin_main

    fn = plugin_main.GroupDistillerPlugin.on_any_message
    assert not inspect.isasyncgenfunction(fn), "监听器是 async generator，会 yield 出消息"
    assert inspect.iscoroutinefunction(fn), "监听器不是普通协程"

    # 用 AST 判断是否真的存在 yield 语句（避免被 docstring 里的“yield”字样误伤）
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    assert not any(
        isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(tree)
    ), "监听器含 yield 语句，会插话/复读"

    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                called.add(node.func.id)
    forbidden = {"stop_event", "should_call_llm", "set_result", "send"}
    assert not (forbidden & called), f"监听器调用了打断主链路的接口: {forbidden & called}"

    async def scenario():
        db = TMP_ROOT / "listener.db"
        plugin, _ = await _make_plugin(db)
        plugin.state.group_id = "999"
        plugin.state.qq_id = "123"
        ret = await plugin.on_any_message(FakeEvent("随便一句", group_id="999", sender_id="123"))
        assert ret is None, "监听器有非 None 返回值"
        await plugin.storage.close()

    asyncio.run(scenario())


def test_no_blocking_io_in_hot_path() -> None:
    """静态扫描：监听热路径不得出现同步 DB / time.sleep / 同步 HTTP。"""
    for fname in ("main.py", "core/collector.py"):
        src = (PLUGIN_DIR / fname).read_text(encoding="utf-8")
        assert "time.sleep(" not in src, f"{fname} 含 time.sleep"
        assert "requests." not in src, f"{fname} 含同步 HTTP"

    import inspect

    import main as plugin_main
    from core import collector as collector_mod

    here = inspect.getsource(collector_mod.Collector.handle_event)
    assert "time.sleep" not in here
    assert "asyncio.create_task" in here, "缓冲区满时未异步落盘"
    # handle_event 本身不应直接 await 同步 DB（写库在后台任务里）
    assert "insert_message" not in here

    listener = inspect.getsource(plugin_main.GroupDistillerPlugin.on_any_message)
    assert "time.sleep" not in listener
    assert ".commit(" not in listener


def test_flush_is_nonblocking_via_task() -> None:
    """缓冲区虽满，handle_event 也不得同步阻塞：落库应挂在后台任务上。"""
    from core.collector import Collector, RuntimeState
    from core.storage import Storage

    async def scenario():
        db = TMP_ROOT / "nonblock.db"
        st = Storage(db)
        await st.init()
        col = Collector(st, {"record_context": False, "min_message_length": 1, "ignore_commands": True},
                        RuntimeState(enabled=True, listen_enabled=True, group_id="999", qq_id="123"))
        col._running = True
        col._max_buffer = 3
        for i in range(3):
            await col.handle_event(FakeEvent(f"消息{i}", group_id="999", sender_id="123", message_id=f"n{i}"))
        # 给后台 flush 任务一点时间
        await asyncio.sleep(0.05)
        assert await st.count_messages("999", "123") == 3
        await st.close()

    asyncio.run(scenario())


# =========================================================================== #
# B4. asyncio.Lock 正确性
# =========================================================================== #


def test_distiller_lock_released_on_exception() -> None:
    from core.collector import RuntimeState
    from core.distiller import Distiller
    from core.storage import Storage

    async def scenario():
        db = TMP_ROOT / "lock.db"
        st = Storage(db)
        await st.init()
        state = RuntimeState(enabled=True, listen_enabled=True, group_id="999", qq_id="123")
        d = Distiller(st, _base_config(), state, context=FakeContext())

        assert isinstance(d._lock, asyncio.Lock)
        assert not d._lock.locked()

        async def boom(_umo):
            raise RuntimeError("模拟蒸馏内部异常")

        d._distill_once = boom  # type: ignore[assignment]
        await d._run("umo")  # 不应抛出
        assert d._running is False, "异常后 _running 未复位"
        assert not d._lock.locked(), "异常后锁未释放 → 会死锁"

        # 锁释放后应能再跑一轮
        ran = {"ok": False}

        async def fine(_umo):
            ran["ok"] = True

        d._distill_once = fine  # type: ignore[assignment]
        await d._run("umo")
        assert ran["ok"] and not d._lock.locked() and d._running is False

        # 触发路径确实走 _run（含 async with）
        d._distill_once = fine  # type: ignore[assignment]
        res = await d.trigger("umo", manual=True)
        assert res.ok is True
        await _wait_task(d)
        await st.close()

    asyncio.run(scenario())


async def _wait_task(d) -> list:
    task = getattr(d, "_task", None)
    if task is not None:
        try:
            await task
        except asyncio.CancelledError:
            pass
    return []


def test_distiller_lock_created_inside_loop() -> None:
    """Lock 必须在事件循环内创建/使用，不得绑定到别的 loop 上。"""
    from core.collector import RuntimeState
    from core.distiller import Distiller
    from core.storage import Storage
    from core.storage import Storage as _S  # noqa: F401

    async def scenario():
        d = Distiller(Storage(TMP_ROOT / "lock2.db"), _base_config(),
                      RuntimeState(), context=FakeContext())
        async with d._lock:
            assert d._lock.locked()
        assert not d._lock.locked()

    asyncio.run(scenario())


# =========================================================================== #
# B5. /zl 权限控制
# =========================================================================== #


def test_admin_only_runtime_check() -> None:
    install_fake_astrbot()
    import inspect

    import main as plugin_main

    src = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    assert "permission_type" not in src, "依赖了静态权限装饰器，拿不到配置值"
    zl_src = inspect.getsource(plugin_main.GroupDistillerPlugin.zl)
    assert "admin_only" in zl_src and "_is_admin" in zl_src, "未在函数体内做运行时权限判断"

    async def scenario():
        db = TMP_ROOT / "admin.db"
        plugin, _ = await _make_plugin(db, admin_only=True)

        # 非管理员 → 被拦
        out = await _collect(plugin.zl(FakeEvent("/zl help", group_id="999", sender_id="1", admin=False)))
        assert "🔒" in _text_of(out[0])

        # 管理员 → 放行
        out = await _collect(plugin.zl(FakeEvent("/zl help", group_id="999", sender_id="1", admin=True)))
        assert "🔒" not in _text_of(out[0]) and "帮助" in _text_of(out[0])

        # admin_only=False → 非管理员也放行
        plugin2, _ = await _make_plugin(TMP_ROOT / "admin2.db", admin_only=False)
        out = await _collect(plugin2.zl(FakeEvent("/zl help", group_id="999", sender_id="1", admin=False)))
        assert "🔒" not in _text_of(out[0]) and "帮助" in _text_of(out[0])

        # is_admin 抛异常时安全降级为 False（不崩溃）
        class BoomEvent(FakeEvent):
            def is_admin(self):
                raise RuntimeError("is_admin 崩了")

        out = await _collect(plugin.zl(BoomEvent("/zl help", group_id="999", sender_id="1")))
        assert "🔒" in _text_of(out[0])

        await plugin.storage.close()
        await plugin2.storage.close()

    asyncio.run(scenario())


# =========================================================================== #
# B6. 导出路径
# =========================================================================== #


def test_export_writes_to_data_dir_not_plugin_dir() -> None:
    install_fake_astrbot()
    import main as plugin_main

    out_dir = TMP_ROOT / "export_root" / "plugin_data" / "astrbot_plugin_group_distiller"
    out_dir.mkdir(parents=True, exist_ok=True)
    orig = plugin_main.resolve_plugin_data_dir
    plugin_main.resolve_plugin_data_dir = lambda: out_dir  # type: ignore[assignment]

    async def scenario():
        db = TMP_ROOT / "export.db"
        plugin, _ = await _make_plugin(db, admin_only=False)
        await _collect(plugin.zl(FakeEvent("/zl set 987654321 123456789", group_id="987654321")))
        await _collect(plugin.zl(FakeEvent("/zl export", group_id="987654321")))

        target = out_dir / "persona_123456789.md"
        assert target.exists(), "导出文件未落到数据目录"
        text = target.read_text(encoding="utf-8")
        assert "人格档案" in text and "123456789" in text
        # 插件目录内不得出现导出文件
        assert not (PLUGIN_DIR / "persona_123456789.md").exists(), "导出写进了插件目录"
        await plugin.storage.close()

    try:
        asyncio.run(scenario())
    finally:
        plugin_main.resolve_plugin_data_dir = orig  # type: ignore[assignment]


# =========================================================================== #
# B7. terminate 清理
# =========================================================================== #


def test_terminate_flushes_and_closes() -> None:
    async def scenario():
        db = TMP_ROOT / "terminate.db"
        plugin, _ = await _make_plugin(db)
        plugin.state.group_id = "999"
        plugin.state.qq_id = "123"
        plugin.state.enabled = True
        plugin.state.listen_enabled = True
        await plugin.collector.start()

        # 只入 buffer，不主动 flush
        for i in range(5):
            await plugin.collector.handle_event(
                FakeEvent(f"语料{i}", group_id="999", sender_id="123", message_id=f"t{i}")
            )
        assert len(plugin.collector._buffer) == 5
        assert plugin.storage._conn is not None

        # 挂一个假的长跑蒸馏任务，验证 terminate 会取消它
        async def forever():
            await asyncio.sleep(3600)

        plugin.distiller._task = asyncio.create_task(forever())

        await plugin.terminate()

        task = plugin.distiller._task
        assert task is not None
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.cancelled(), "terminate 未取消后台蒸馏任务"
        assert plugin.storage._conn is None, "terminate 未关闭 DB 连接"

        # 重新打开同一库，确认 buffer 已落盘
        from core.storage import Storage

        st2 = Storage(db)
        await st2.init()
        assert await st2.count_messages("999", "123") == 5, "terminate 前缓冲未 flush，语料丢失"
        await st2.close()

    asyncio.run(scenario())


# =========================================================================== #
# C. 静态校验
# =========================================================================== #


def test_conf_schema_static() -> None:
    schema_path = PLUGIN_DIR / "_conf_schema.json"
    raw = schema_path.read_text(encoding="utf-8")
    schema = json.loads(raw)  # 非法 JSON 会抛异常
    assert isinstance(schema, dict) and schema
    for key, spec in schema.items():
        assert isinstance(spec, dict), f"{key} 配置项不是对象"
        for field in ("type", "description", "default"):
            assert field in spec, f"配置项 {key} 缺字段 {field}"

    # 代码里读取的配置键
    read_keys: set[str] = set()
    for py in PLUGIN_DIR.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        read_keys |= set(re.findall(r"config\.get\(\s*[\"']([A-Za-z0-9_]+)[\"']", text))
    schema_keys = set(schema.keys())

    print(f"    [info] schema 声明 {len(schema_keys)} 键: {sorted(schema_keys)}")
    print(f"    [info] 代码读取 {len(read_keys)} 键: {sorted(read_keys)}")

    # ① schema 声明的都被读到（除非文档性字段）
    unused = schema_keys - read_keys
    # ② 代码读的键都在 schema 里
    undeclared = read_keys - schema_keys

    assert not undeclared, f"代码读取了 schema 未声明的键（拼写错误会静默失效）: {sorted(undeclared)}"
    # unused 允许存在但不报告为失败；此处打印
    if unused:
        print(f"    [warn] schema 声明但代码未读取: {sorted(unused)}")


def test_metadata_yaml_static() -> None:
    text = (PLUGIN_DIR / "metadata.yaml").read_text(encoding="utf-8")
    # 无 pyyaml，做结构化文本校验
    pairs: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith((" ", "\t", "#")):
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            pairs[k.strip()] = v.strip()
    assert pairs.get("name") == "astrbot_plugin_group_distiller", f"name 错误: {pairs.get('name')}"
    for k in ("display_name", "desc", "version", "author", "repo"):
        assert k in pairs, f"metadata.yaml 缺字段 {k}"
    assert pairs["repo"].startswith("http"), "repo 不是 URL"
    assert re.match(r"v?\d+\.\d+\.\d+", pairs["version"]), f"version 非法: {pairs['version']}"


def test_requirements_nonempty() -> None:
    text = (PLUGIN_DIR / "requirements.txt").read_text(encoding="utf-8").strip()
    assert text, "requirements.txt 为空"


def test_all_py_compile() -> None:
    import py_compile

    for py in PLUGIN_DIR.rglob("*.py"):
        if "_qa_tmp" in str(py):
            continue
        py_compile.compile(str(py), doraise=True)


def test_core_modules_importable_without_astrbot() -> None:
    for name in list(sys.modules):
        if name == "astrbot" or name.startswith("astrbot."):
            sys.modules.pop(name, None)
    # 这些模块必须能在无 AstrBot 环境下导入
    import importlib

    for mod in ("core", "core.storage", "core.collector", "core.distiller",
                "core.prompts", "core.progress", "tests.test_core"):
        importlib.import_module(mod)


# =========================================================================== #
# D. 端到端
# =========================================================================== #


def test_end_to_end_with_mock_event() -> None:
    install_fake_astrbot()
    payload = json.dumps(
        {
            "layer1_rules": [{"text": "绝不发语音", "confidence": 0.9, "quotes": ["我不发语音"]}],
            "layer2_identity": [],
            "layer3_style": [{"text": "喜欢用～", "confidence": 0.8, "quotes": ["好呀～"]}],
            "layer4_behavior": [],
            "layer5_interests": [],
            "uncertainty": ["身份证据不足"],
        },
        ensure_ascii=False,
    )
    provider = FakeProvider(payload)

    async def scenario():
        db = TMP_ROOT / "e2e.db"
        if db.exists():
            db.unlink()
        plugin, plugin_main = await _make_plugin(
            db, provider=provider, saturation_messages=1500, admin_only=False
        )
        plugin.state.group_id = "987654321"
        plugin.state.qq_id = "123456789"
        plugin.state.nickname = "老王"
        plugin.state.enabled = True
        plugin.state.listen_enabled = True

        # 30 条模拟群消息 → 走真实监听器
        for i in range(30):
            ev = FakeEvent(
                f"第{i}条群聊消息，聊点有的没的",
                group_id="987654321",
                sender_id="123456789",
                sender_name="老王",
                message_id=f"e2e-{i}",
                timestamp=1_700_000_000 + i,
            )
            await plugin.on_any_message(ev)
        await plugin.collector.flush()

        total = await plugin.storage.count_messages("987654321", "123456789")
        assert total == 30, f"入库条数错误: {total}"

        # 进度面板断言
        panel = await plugin._build_panel()
        assert "123456789" in panel, "面板缺目标 QQ"
        assert "987654321" in panel, "面板缺群号"
        assert "30 条" in panel, f"面板缺语料条数: {panel}"
        assert "2%" in panel, f"面板进度百分比错误: {panel}"  # round(30/1500*100)=2

        # 手动蒸馏一轮（LLM 已打桩，不联网）
        await plugin.distiller._distill_once("aiocqhttp:GroupMessage:987654321")
        assert provider.calls == 1, "LLM Provider 未被调用，蒸馏链路断了"

        snap = await plugin.storage.get_persona("987654321", "123456789")
        assert isinstance(snap, dict), "蒸馏后未生成 Persona"
        assert snap["meta"]["distill_round"] == 1
        assert snap["layers"]["layer1_rules"][0]["text"] == "绝不发语音"
        assert await plugin.storage.count_undistilled("987654321", "123456789") == 0, "语料未标记已蒸馏"

        # 档案渲染包含蒸馏结论
        from core import progress

        prof = progress.render_profile(snap, "老王", "123456789")
        assert "绝不发语音" in prof

        await plugin.storage.close()

    asyncio.run(scenario())


# =========================================================================== #
# 补充：并发单飞 / 自动触发 / 模糊输入 / 合并健壮性
# =========================================================================== #


def test_distiller_single_flight() -> None:
    """同一时间只允许一轮蒸馏：第二次 trigger 必须被拒。"""
    from core.collector import RuntimeState
    from core.distiller import Distiller
    from core.storage import Storage

    async def scenario():
        st = Storage(TMP_ROOT / "sf.db")
        await st.init()
        state = RuntimeState(enabled=True, listen_enabled=True, group_id="999", qq_id="123")
        d = Distiller(st, _base_config(), state, context=FakeContext())

        async def slow(_umo):
            await asyncio.sleep(0.2)

        d._distill_once = slow  # type: ignore[assignment]
        r1 = await d.trigger("umo", manual=True)
        assert r1.ok is True, "首轮触发被拒"
        r2 = await d.trigger("umo", manual=True)
        assert r2.ok is False, "并发第二轮未被拒 → 会烧双份 token"
        assert "已有一轮" in r2.message
        await _wait_task(d)
        assert not d.is_running()
        await st.close()

    asyncio.run(scenario())


def test_maybe_auto_threshold() -> None:
    """未达阈值不触发，达到阈值才自动触发一轮。"""
    payload = json.dumps({"layer1_rules": [{"text": "低调", "confidence": 0.7}]})

    async def scenario():
        db = TMP_ROOT / "auto.db"
        if db.exists():
            db.unlink()
        provider = FakeProvider(payload)
        plugin, _ = await _make_plugin(
            db, provider=provider, auto_distill=True, distill_interval_messages=5
        )
        plugin.state.group_id = "999"
        plugin.state.qq_id = "123"

        async def feed(n, base):
            for i in range(n):
                await plugin.collector.handle_event(
                    FakeEvent(f"语料{base + i}", group_id="999", sender_id="123",
                             message_id=f"a{base + i}")
                )
            await plugin.collector.flush()

        await feed(4, 0)
        await plugin.distiller.maybe_auto("umo")
        assert provider.calls == 0, "未达阈值却触发了蒸馏"

        await feed(1, 4)  # 累计 5 条，达到阈值
        await plugin.distiller.maybe_auto("umo")
        await _wait_task(plugin.distiller)
        assert provider.calls == 1, "达到阈值却未触发自动蒸馏"
        await plugin.storage.close()

    asyncio.run(scenario())


def test_zl_fuzz_no_crash() -> None:
    """畸形 / 极端 message_str 不得让 /zl 抛异常。"""
    cases = [
        None, "", "   ", "/zl", "zl", "／zl", "///", "/zl    ",
        "/zl set", "/zl set   ", "中文没有前缀", "🎉", "／zl 蒸馏",
        "/zl correct", "/zl reset", "/zl set 0 0", "/zl SET 1 2 3 4 5",
        "/zl " + "x" * 5000, "/zl 帮助", "zl help", "\u3000/zl help",
    ]

    async def scenario():
        install_fake_astrbot()
        db = TMP_ROOT / "fuzz.db"
        plugin, _ = await _make_plugin(db, admin_only=False)
        await _collect(plugin.zl(FakeEvent("/zl set 987654321 123456789", group_id="987654321")))

        for raw in cases:
            out = []
            try:
                async for r in plugin.zl(FakeEvent(raw, group_id="987654321")):
                    out.append(_text_of(r))
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(f"输入 {raw!r} 抛异常: {exc!r}") from exc
            assert out, f"输入 {raw!r} 无返回"
        await _wait_task(plugin.distiller)
        await plugin.storage.close()

    asyncio.run(scenario())


def test_merge_and_parse_robustness() -> None:
    from core.distiller import empty_snapshot, merge_snapshot, parse_analysis

    # 非 dict 分析不崩溃，轮次照常自增
    s = merge_snapshot(empty_snapshot(), "not a dict")  # type: ignore[arg-type]
    assert s["meta"]["distill_round"] == 1

    # 层里塞垃圾：None/数字/空串/空文本被丢弃，非法置信回退 0.5
    s2 = merge_snapshot(
        empty_snapshot(),
        {
            "layer1_rules": [None, 123, "", {"text": ""}, {"text": "ok", "confidence": "bad"}],
        },
    )
    items = s2["layers"]["layer1_rules"]
    assert [i["text"] for i in items] == ["ok"]
    assert items[0]["confidence"] == 0.5

    # uncertainty 传成字符串时会被逐字符拆散（低危健壮性问题，记录但不阻断）
    s3 = merge_snapshot(
        empty_snapshot(), {"layer1_rules": [], "uncertainty": "abc"}
    )
    assert isinstance(s3["uncertainty"], list)  # 不崩溃

    # JSON 抽取：嵌套 / 代码块 / 空对象
    assert parse_analysis('前缀 {"a": {"b": [1, 2]}} 后缀') == {"a": {"b": [1, 2]}}
    assert parse_analysis("```json\n{}\n```") == {}
    assert parse_analysis("{") is None
    assert parse_analysis("} {") is None


# =========================================================================== #
# 运行入口
# =========================================================================== #


ALL_TESTS = [
    test_progress_bar_extremes,
    test_parse_command_unit,
    test_zl_command_dispatch,
    test_collector_filtering,
    test_context_recording,
    test_dedup_by_message_id,
    test_merge_incremental_and_corrections,
    test_data_dir_under_plugin_data,
    test_data_dir_fallback_is_plugin_local,
    test_sqlite_check_same_thread_control_experiment,
    test_sqlite_storage_cross_thread_ok,
    test_listener_is_pure_coroutine,
    test_no_blocking_io_in_hot_path,
    test_flush_is_nonblocking_via_task,
    test_distiller_lock_released_on_exception,
    test_distiller_lock_created_inside_loop,
    test_admin_only_runtime_check,
    test_export_writes_to_data_dir_not_plugin_dir,
    test_terminate_flushes_and_closes,
    test_conf_schema_static,
    test_metadata_yaml_static,
    test_requirements_nonempty,
    test_all_py_compile,
    test_core_modules_importable_without_astrbot,
    test_end_to_end_with_mock_event,
    test_distiller_single_flight,
    test_maybe_auto_threshold,
    test_zl_fuzz_no_crash,
    test_merge_and_parse_robustness,
]


def main() -> int:
    failures = 0
    for test in ALL_TESTS:
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
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)
