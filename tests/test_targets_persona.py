"""多目标 / 人格模板 相关的新功能测试（v0.2.0）。

覆盖本轮新增能力：

- 目标解析：``template_list`` 清单 / 纯文本清单 / 旧版扁平字段三路合并去重；
- 多目标采集：同群多人、多群多人，且上下文归属不串台；
- ``/zl add | del | use | list | set`` 的目标管理行为与持久化；
- 数据库升级：v0.1.0 老库自动补 ``context_for_qq`` 列；
- 人格模板：文本生成、人格 ID 规整、字数分块；
- 人格写入：创建 / 更新 / 接口缺失 / 抛异常 四条路径；
- 蒸馏达标后自动写入 AstrBot 人格。

运行方式（任选其一）：

    python tests/test_targets_persona.py
    python -m pytest tests/test_targets_persona.py

所有临时文件、数据库都落在 ``tests/_tmp_mt/`` 内，运行结束自动清理。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TMP_ROOT = Path(__file__).resolve().parent / "_tmp_mt"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

PLUGIN_DIR = ROOT

# 复用 QA 那套伪造 AstrBot 环境与事件替身，避免重复造轮子
from tests.test_qa_verify import (  # noqa: E402
    FakeContext,
    FakeEvent,
    FakeProvider,
    _base_config,
    _collect,
    _make_plugin,
    _text_of,
    install_fake_astrbot,
)

# 目标 ID 用真实长度（9 位），顺便验证长度校验不会误伤
GID_A = "987654321"
GID_B = "555555555"
QQ_A = "123456789"
QQ_B = "222222222"
QQ_C = "333333333"


def _db(name: str) -> Path:
    """给每个用例一个独立的项目内临时库路径。"""
    path = TMP_ROOT / name
    if path.exists():
        path.unlink()
    return path


# =========================================================================== #
# 1. 目标解析与合并
# =========================================================================== #


def test_parse_targets_text_unit() -> None:
    """文本清单：分隔符、注释、昵称带空格、非法行告警。"""
    from core.targets import parse_targets_text

    text = "\n".join(
        [
            "# 这是注释，应被忽略",
            "// 这也是注释",
            "",
            f"{GID_A} {QQ_A} 老王",
            f"{GID_A},{QQ_B},小李",
            f"{GID_A}\u3001{QQ_C}\u3001小 张 张",
            f"{GID_B}|999999999|阿呆",
            f"{GID_A}:{QQ_A}",          # 冒号不是分隔符 → 整段当群号 → 非法
            "只有一段",
            f"123 {QQ_A}",              # 群号太短 → 非法
        ]
    )
    specs, warnings = parse_targets_text(text)

    keys = [s.key for s in specs]
    assert f"{GID_A}:{QQ_A}" in keys
    assert f"{GID_A}:{QQ_B}" in keys
    assert f"{GID_A}:{QQ_C}" in keys
    assert f"{GID_B}:999999999" in keys
    assert len(specs) == 4, f"解析出的目标数不对: {keys}"

    # 昵称带空格也要完整保留
    nick = {s.key: s.nickname for s in specs}
    assert nick[f"{GID_A}:{QQ_C}"] == "小 张 张", f"昵称被截断: {nick}"

    # 三行非法输入应各自产出一条告警，并精确指出行号
    assert len(warnings) == 3, f"告警条数不对: {warnings}"
    for lineno in ("第 8 行", "第 9 行", "第 10 行"):
        assert any(lineno in w for w in warnings), (lineno, warnings)
    # 群号太短那一行要说明是「不合法」，而不是笼统的格式错误
    assert "不合法" in warnings[-1], warnings[-1]


def test_parse_targets_text_dedupe_and_empty() -> None:
    """重复目标去重；空输入返回空列表。"""
    from core.targets import parse_targets_text

    specs, warnings = parse_targets_text("")
    assert specs == [] and warnings == []

    specs2, _ = parse_targets_text(
        f"{GID_A} {QQ_A}\n{GID_A} {QQ_A} 另一个昵称\n"
    )
    assert len(specs2) == 1, "重复目标未去重"


def test_collect_config_targets_three_sources() -> None:
    """可视化清单 + 文本清单 + 旧版扁平字段，三路合并去重。"""
    from core.targets import collect_config_targets

    cfg = {
        # ① template_list 形态（带 AstrBot 自己的 __template_key）
        "targets": [
            {"__template_key": "target", "group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"},
            {"__template_key": "target", "group_id": GID_A, "qq_id": QQ_B, "nickname": "小李"},
        ],
        # ② 文本清单：其中一条与①重复
        "targets_text": f"{GID_B} {QQ_C} 阿呆\n{GID_A} {QQ_A} 老王重复",
        # ③ 旧版扁平字段：又是一条新的
        "target_group_id": GID_B,
        "target_qq_id": "444444444",
        "target_nickname": "旧字段",
    }
    specs, warnings = collect_config_targets(cfg)
    keys = [s.key for s in specs]

    assert warnings == [], f"不该有告警: {warnings}"
    assert len(specs) == 4, f"合并去重结果不对: {keys}"
    assert keys == [
        f"{GID_A}:{QQ_A}",
        f"{GID_A}:{QQ_B}",
        f"{GID_B}:{QQ_C}",
        f"{GID_B}:444444444",
    ], keys


def test_collect_config_targets_tolerates_garbage() -> None:
    """配置里塞垃圾（None / 非列表 / 缺字段）不得崩溃。"""
    from core.targets import collect_config_targets

    for bad in (
        None,
        {},
        {"targets": None},
        {"targets": "not-a-list"},
        {"targets": [None, 123, {}, {"group_id": "", "qq_id": ""}]},
        {"targets_text": 12345},
    ):
        specs, _ = collect_config_targets(bad)
        assert specs == [], f"垃圾输入 {bad!r} 竟解析出目标: {specs}"


def test_id_validation_rules() -> None:
    """群号/QQ 号校验：长度、非数字、带前缀噪声。"""
    from core.targets import is_valid_id, make_target

    assert is_valid_id(QQ_A) is True
    assert is_valid_id("1234") is False          # 太短
    assert is_valid_id("1" * 21) is False        # 太长
    assert is_valid_id("abc12345") is False      # 含字母
    assert is_valid_id("") is False
    assert is_valid_id(None) is False
    # 复制粘贴常见的噪声应被剥掉
    assert is_valid_id(f"群号:{GID_A}") is True
    assert is_valid_id(f"@{QQ_A}") is True

    assert make_target("", QQ_A) is None
    assert make_target(GID_A, "abc") is None
    assert make_target(GID_A, QQ_A, "x" * 100).nickname == "x" * 24


def test_targets_state_roundtrip() -> None:
    """目标列表 JSON 序列化 / 反序列化；坏数据安全退化。"""
    from core.targets import TargetSpec, targets_from_json, targets_to_json

    specs = [
        TargetSpec(GID_A, QQ_A, "老王"),
        TargetSpec(GID_B, QQ_C, "阿呆"),
    ]
    raw = targets_to_json(specs)
    assert json.loads(raw)[0]["qq_id"] == QQ_A
    back = targets_from_json(raw)
    assert [s.key for s in back] == [s.key for s in specs]
    assert back[1].nickname == "阿呆"

    for bad in (None, "", "{", "{}", "[1,2]", '"str"'):
        assert targets_from_json(bad) == []


# =========================================================================== #
# 2. 多目标采集
# =========================================================================== #


def test_multi_target_two_in_one_group_no_crosstalk() -> None:
    """同群两个目标：上下文必须各自归属，绝不串台。"""
    from core.collector import Collector, RuntimeState
    from core.storage import Storage
    from core.targets import TargetSpec

    async def scenario():
        st = Storage(_db("ctx_iso.db"))
        await st.init()
        state = RuntimeState(
            enabled=True,
            listen_enabled=True,
            targets=[
                TargetSpec(GID_A, QQ_A, "老王"),
                TargetSpec(GID_A, QQ_B, "小李"),
            ],
        )
        col = Collector(
            st,
            {"record_context": True, "min_message_length": 1, "ignore_commands": True},
            state,
        )

        await col.handle_event(FakeEvent("路人甲", group_id=GID_A, sender_id="555", message_id="c1"))
        await col.handle_event(FakeEvent("老王说话", group_id=GID_A, sender_id=QQ_A, message_id="t1"))
        await col.handle_event(FakeEvent("路人乙", group_id=GID_A, sender_id="556", message_id="c2"))
        await col.handle_event(FakeEvent("小李说话", group_id=GID_A, sender_id=QQ_B, message_id="t2"))
        await col.handle_event(FakeEvent("路人丙", group_id=GID_A, sender_id="557", message_id="c3"))
        await col.flush()

        rows_a = await st.fetch_undistilled(GID_A, QQ_A, 100)
        rows_b = await st.fetch_undistilled(GID_A, QQ_B, 100)
        text_a = {r["content"] for r in rows_a}
        text_b = {r["content"] for r in rows_b}

        # 老王的语料里绝不能出现小李的发言
        assert "老王说话" in text_a
        assert "小李说话" not in text_a, f"串台了！老王的语料里混进了小李: {text_a}"
        assert "小李说话" in text_b
        assert "老王说话" not in text_b, f"串台了！小李的语料里混进了老王: {text_b}"

        # 上下文归属正确
        ctx_a = {r["content"] for r in rows_a if r["is_context"]}
        ctx_b = {r["content"] for r in rows_b if r["is_context"]}
        assert ctx_a == {"路人甲", "路人乙"}, ctx_a
        assert ctx_b == {"路人乙", "路人丙"}, ctx_b

        # 目标语料计数各自为 1，上下文不计入
        assert await st.count_messages(GID_A, QQ_A) == 1
        assert await st.count_messages(GID_A, QQ_B) == 1
        await st.close()

    asyncio.run(scenario())


def test_multi_target_two_groups_each_two() -> None:
    """两个群各两人：四个目标互不干扰，非目标消息一律丢弃。"""
    from core.collector import Collector, RuntimeState
    from core.storage import Storage
    from core.targets import TargetSpec

    specs = [
        TargetSpec(GID_A, QQ_A, "A群老王"),
        TargetSpec(GID_A, QQ_B, "A群小李"),
        TargetSpec(GID_B, QQ_A, "B群老王"),   # 同一个 QQ 在另一个群
        TargetSpec(GID_B, QQ_C, "B群阿呆"),
    ]

    async def scenario():
        st = Storage(_db("two_groups.db"))
        await st.init()
        state = RuntimeState(enabled=True, listen_enabled=True, targets=specs)
        col = Collector(
            st,
            {"record_context": False, "min_message_length": 1, "ignore_commands": True},
            state,
        )

        feed = [
            (GID_A, QQ_A, "A群老王发言"),
            (GID_A, QQ_B, "A群小李发言"),
            (GID_B, QQ_A, "B群老王发言"),
            (GID_B, QQ_C, "B群阿呆发言"),
            (GID_A, "999999999", "路人甲插话"),      # 不是目标 → 丢弃
            (GID_B, "888888888", "路人乙插话"),      # 不是目标 → 丢弃
            (GID_A, QQ_C, "C在A群说话但A群没挂他"),  # 该群无此目标 → 丢弃
        ]
        for i, (gid, qq, text) in enumerate(feed):
            await col.handle_event(
                FakeEvent(text, group_id=gid, sender_id=qq, message_id=f"m{i}")
            )
        await col.flush()

        assert await st.count_messages(GID_A, QQ_A) == 1
        assert await st.count_messages(GID_A, QQ_B) == 1
        assert await st.count_messages(GID_B, QQ_A) == 1
        assert await st.count_messages(GID_B, QQ_C) == 1

        # 每人的库内容都只有自己那一句
        for gid, qq, expected in (
            (GID_A, QQ_A, "A群老王发言"),
            (GID_A, QQ_B, "A群小李发言"),
            (GID_B, QQ_A, "B群老王发言"),
            (GID_B, QQ_C, "B群阿呆发言"),
        ):
            rows = await st.fetch_undistilled(gid, qq, 100)
            assert [r["content"] for r in rows] == [expected], (gid, qq, rows)

        # 内存计数也各归各
        assert state.counts_for(specs[0])["total"] == 1
        assert state.counts_for(specs[3])["total"] == 1
        await st.close()

    asyncio.run(scenario())


def test_runtime_state_legacy_single_target_still_works() -> None:
    """老写法（只传 group_id + qq_id）仍应被视为一个目标。"""
    from core.collector import RuntimeState
    from core.targets import TargetSpec

    state = RuntimeState(group_id=GID_A, qq_id=QQ_A, nickname="老王")
    assert state.has_target() is True
    specs = state.collect_targets()
    assert len(specs) == 1 and specs[0].key == f"{GID_A}:{QQ_A}"

    # 新增目标后，targets 成为权威来源
    state.add_target(TargetSpec(GID_B, QQ_B, "小李"))
    assert len(state.collect_targets()) == 2
    assert state.find_target(QQ_B).group_id == GID_B

    assert state.set_active(QQ_B) is True
    assert state.active_target().qq_id == QQ_B
    assert state.qq_id == QQ_B  # 镜像字段已同步

    assert state.remove_target(QQ_B) == 1
    assert len(state.collect_targets()) == 1
    assert state.active_target().qq_id == QQ_A  # 自动回落到剩下的目标


# =========================================================================== #
# 3. 数据库升级
# =========================================================================== #


def test_storage_migration_adds_context_column() -> None:
    """v0.1.0 的老库（无 context_for_qq）应被自动补列且数据不丢。"""
    from core.storage import Storage

    db = _db("legacy.db")
    # 手工造一个老版本表结构
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE messages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id   TEXT UNIQUE,
            group_id     TEXT NOT NULL,
            speaker_qq   TEXT NOT NULL,
            speaker_name TEXT NOT NULL DEFAULT '',
            content      TEXT NOT NULL,
            raw_type     TEXT NOT NULL DEFAULT 'text',
            timestamp    INTEGER NOT NULL DEFAULT 0,
            created_at   INTEGER NOT NULL DEFAULT 0,
            is_context   INTEGER NOT NULL DEFAULT 0,
            distilled    INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute(
        "INSERT INTO messages (message_id, group_id, speaker_qq, content, is_context)"
        " VALUES ('old1', ?, ?, '老数据', 0)",
        (GID_A, QQ_A),
    )
    conn.commit()
    conn.close()

    async def scenario():
        st = Storage(db)
        await st.init()  # 触发迁移
        assert st._conn is not None
        cols = {
            row["name"]
            for row in st._conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        assert "context_for_qq" in cols, f"未补列: {cols}"
        # 老数据仍在，且能被查出来
        assert await st.count_messages(GID_A, QQ_A) == 1
        rows = await st.fetch_undistilled(GID_A, QQ_A, 10)
        assert [r["content"] for r in rows] == ["老数据"]
        await st.close()

        # 幂等：再开一次不应报错
        st2 = Storage(db)
        await st2.init()
        await st2.close()

    asyncio.run(scenario())


# =========================================================================== #
# 4. 人格模板文本
# =========================================================================== #


def _rich_snapshot():
    """造一份五层齐全的 Persona 快照。"""
    from core.distiller import empty_snapshot, merge_snapshot

    analysis = {
        "layer1_rules": [{"text": "绝不发语音", "confidence": 0.9, "quotes": ["我不发语音的"]}],
        "layer2_identity": [{"text": "在校生，年龄二十出头", "confidence": 0.7}],
        "layer3_style": [{"text": "爱用～结尾", "confidence": 0.8, "quotes": ["好呀～"]}],
        "layer4_behavior": [{"text": "半夜最活跃", "confidence": 0.6}],
        "layer5_interests": [{"text": "沉迷抽卡游戏", "confidence": 0.75}],
        "uncertainty": ["工作状态证据不足"],
    }
    return merge_snapshot(empty_snapshot(), analysis, ["他其实偶尔也用语音"])


def test_build_astrbot_persona_content() -> None:
    """人格模板：结构齐全、含 5 层结论、含纠正层、默认不带原话。"""
    from core import prompts

    snap = _rich_snapshot()
    text = prompts.build_astrbot_persona(snap, "老王", QQ_A)

    for must in (
        "老王",
        QQ_A,
        "绝不发语音",
        "在校生",
        "爱用～结尾",
        "半夜最活跃",
        "沉迷抽卡游戏",
        "他其实偶尔也用语音",
        "工作状态证据不足",
        "绝不要说自己是 AI",
    ):
        assert must in text, f"人格模板缺少关键内容: {must}"

    # 角色设定类模板不应该把证据原话塞进去
    assert "我不发语音的" not in text, "默认不该带原话样例"
    assert "好呀～" not in text

    # 层标题按顺序出现
    order = [
        text.index(prompts.PERSONA_SECTION_TITLES[key]) for key in prompts.LAYER_KEYS
    ]
    assert order == sorted(order), f"各层顺序错乱: {order}"


def test_build_astrbot_persona_options() -> None:
    """带原话 / 追加规矩 / 空快照三种情况。"""
    from core import prompts

    snap = _rich_snapshot()

    with_quotes = prompts.build_astrbot_persona(snap, "老王", QQ_A, include_evidence=True)
    assert "我不发语音的" in with_quotes, "开了原话却没带上"

    with_rules = prompts.build_astrbot_persona(
        snap, "老王", QQ_A, extra_rules="绝对不许发自拍"
    )
    assert "绝对不许发自拍" in with_rules
    assert "主人额外交代的规矩" in with_rules

    empty = prompts.build_astrbot_persona(None, "", "")
    assert "绝不要说自己是 AI" in empty  # 骨架仍在
    assert prompts.PERSONA_EMPTY_HINT in empty  # 空层有明确占位
    assert "{nickname}" not in empty and "{" not in empty.split("---")[0]


def test_build_persona_summary() -> None:
    """人格模板体检报告：没档案 / 缺层 / 齐全三种口径。"""
    from core import prompts

    assert "还没有任何档案" in prompts.build_persona_summary(None, "老王", QQ_A)

    partial = _rich_snapshot()
    partial["layers"]["layer5_interests"] = []
    summary = prompts.build_persona_summary(partial, "老王", QQ_A)
    assert "4/5" in summary
    assert "缺少" in summary and "兴趣偏好" in summary

    full = prompts.build_persona_summary(_rich_snapshot(), "老王", QQ_A)
    assert "5/5" in full
    assert "可以直接搬进 AstrBot" in full


def test_chunk_text_boundaries() -> None:
    """分块：空输入、短文本、超长单行、多行。"""
    from core import progress

    assert progress.chunk_text("") == []
    assert progress.chunk_text("   ") == []
    assert progress.chunk_text("abc", limit=10) == ["abc"]

    chunks = progress.chunk_text("x" * 25, limit=10)
    assert "".join(chunks) == "x" * 25
    assert all(len(c) <= 10 for c in chunks), chunks

    text = "\n".join(f"第{i}行内容" for i in range(50))
    chunks2 = progress.chunk_text(text, limit=60)
    assert all(len(c) <= 60 for c in chunks2), [len(c) for c in chunks2]
    assert "\n".join(chunks2).replace("\n", "") == text.replace("\n", "")


# =========================================================================== #
# 5. 人格写入 AstrBot
# =========================================================================== #


class FakePersonaMgr:
    """伪 AstrBot PersonaManager。"""

    def __init__(self) -> None:
        self.personas: dict[str, str] = {}

    async def get_persona(self, persona_id: str):
        if persona_id not in self.personas:
            raise ValueError(f"Persona with ID {persona_id} does not exist.")
        return persona_id

    async def create_persona(self, persona_id: str, system_prompt: str, **_k):
        if persona_id in self.personas:
            raise ValueError("already exists")
        self.personas[persona_id] = system_prompt
        return persona_id

    async def update_persona(self, persona_id: str, system_prompt: str | None = None, **_k):
        if persona_id not in self.personas:
            raise ValueError("missing")
        if system_prompt is not None:
            self.personas[persona_id] = system_prompt
        return persona_id


def test_make_persona_id_sanitize() -> None:
    """人格 ID：清理非法字符、限长、空值兜底。"""
    from core import persona_bridge

    pid = persona_bridge.make_persona_id("群友蒸馏", "老 王", QQ_A)
    assert pid == f"群友蒸馏-老王-{QQ_A}", pid

    # 全都是非法字符时也不能产出空 ID
    weird = persona_bridge.make_persona_id("//", "///", QQ_A)
    assert weird and weird.strip("-") == weird, weird
    assert QQ_A in weird, weird

    long_id = persona_bridge.make_persona_id("前缀", "名" * 100, QQ_A)
    assert len(long_id) <= 60, len(long_id)

    # 空前缀回落到默认前缀
    assert persona_bridge.make_persona_id(
        "", "老王", QQ_A
    ).startswith(persona_bridge.DEFAULT_PERSONA_PREFIX)


def test_push_persona_all_paths() -> None:
    """写入人格：新建 / 更新 / 无管理器 / 管理器抛异常。"""

    async def scenario():
        from core import persona_bridge

        # ① 正常路径：先新建，再更新
        ctx = FakeContext()
        mgr = FakePersonaMgr()
        ctx.persona_manager = mgr

        ok, msg = await persona_bridge.push_persona(ctx, "pid-1", "第一版人格")
        assert ok is True and "已新建" in msg, msg
        assert mgr.personas["pid-1"] == "第一版人格"

        ok2, msg2 = await persona_bridge.push_persona(ctx, "pid-1", "第二版人格")
        assert ok2 is True and "已更新" in msg2, msg2
        assert mgr.personas["pid-1"] == "第二版人格"

        # ② 没有 persona_manager（老版本 AstrBot）→ 明确让人手动复制
        plain_ctx = FakeContext()
        ok3, msg3 = await persona_bridge.push_persona(plain_ctx, "pid-2", "内容")
        assert ok3 is False
        assert "persona_manager" in msg3 and "/zl persona" in msg3, msg3

        # ③ 管理器内部抛异常 → 不得把异常抛出来
        class BoomMgr(FakePersonaMgr):
            async def get_persona(self, persona_id: str):
                raise RuntimeError("数据库炸了")

            async def create_persona(self, persona_id: str, system_prompt: str, **_k):
                raise RuntimeError("数据库炸了")

        boom_ctx = FakeContext()
        boom_ctx.persona_manager = BoomMgr()
        ok4, msg4 = await persona_bridge.push_persona(boom_ctx, "pid-3", "内容")
        assert ok4 is False and "失败" in msg4, msg4

        # ④ 空模板直接拒绝
        ok5, msg5 = await persona_bridge.push_persona(ctx, "pid-4", "   ")
        assert ok5 is False and "空" in msg5, msg5

        # ⑤ 同步接口也能兼容（不 awaitable 的实现）
        class SyncMgr:
            def __init__(self):
                self.store: dict[str, str] = {}

            def get_persona(self, persona_id):
                if persona_id not in self.store:
                    raise ValueError("missing")
                return persona_id

            def create_persona(self, persona_id, system_prompt, **_k):
                self.store[persona_id] = system_prompt
                return persona_id

        sync_ctx = FakeContext()
        sync_ctx.persona_manager = SyncMgr()
        ok6, _ = await persona_bridge.push_persona(sync_ctx, "pid-5", "同步版")
        assert ok6 is True and sync_ctx.persona_manager.store["pid-5"] == "同步版"

    asyncio.run(scenario())


# =========================================================================== #
# 6. /zl 目标管理指令
# =========================================================================== #


def test_zl_target_management_commands() -> None:
    """/zl add | list | use | del | set 的完整行为。"""

    async def scenario():
        install_fake_astrbot()
        plugin, _ = await _make_plugin(_db("mgmt.db"), admin_only=False)
        ev = lambda s: FakeEvent(s, group_id=GID_A, sender_id="10001")  # noqa: E731

        # ① add 三个目标（两个同群、一个在另一个群）
        await _collect(plugin.zl(ev(f"/zl add {GID_A} {QQ_A} 老王")))
        await _collect(plugin.zl(ev(f"/zl add {GID_A} {QQ_B} 小李")))
        out = await _collect(plugin.zl(ev(f"/zl add {GID_B} {QQ_C} 小 张 张")))
        assert "已添加" in _text_of(out[0]), _text_of(out[0])
        assert len(plugin.state.targets) == 3, plugin.state.targets
        assert plugin.state.targets[2].nickname == "小 张 张"
        assert plugin.state.active_target().qq_id == QQ_C  # 新加的成为当前

        # ② 重复添加同一个目标 → 只更新，不重复计入
        out = await _collect(plugin.zl(ev(f"/zl add {GID_A} {QQ_A}")))
        assert "已更新" in _text_of(out[0]), _text_of(out[0])
        assert len(plugin.state.targets) == 3

        # ③ list 应列出全部三个
        out = await _collect(plugin.zl(ev("/zl list")))
        listing = _text_of(out[0])
        assert "共 3 个" in listing, listing
        for qq in (QQ_A, QQ_B, QQ_C):
            assert qq in listing, f"清单缺 {qq}"

        # ④ use 切换
        out = await _collect(plugin.zl(ev(f"/zl use {QQ_A}")))
        assert "已切换到" in _text_of(out[0]), _text_of(out[0])
        assert plugin.state.active_target().qq_id == QQ_A
        assert plugin.state.qq_id == QQ_A

        # ⑤ 非法输入被友好拒绝
        out = await _collect(plugin.zl(ev("/zl add abc 123")))
        assert "群号必须是纯数字" in _text_of(out[0])
        out = await _collect(plugin.zl(ev("/zl add " + GID_A + " abc")))
        assert "QQ 号必须是纯数字" in _text_of(out[0])
        out = await _collect(plugin.zl(ev("/zl add")))
        assert "用法" in _text_of(out[0])

        # ⑥ 持久化：目标清单已写进 state 表
        raw = await plugin.storage.get_state("rt_targets")
        saved = json.loads(raw)
        assert len(saved) == 3, saved
        assert await plugin.storage.get_state("rt_active_key") == f"{GID_A}:{QQ_A}"

        # ⑦ del（不 purge）→ 只摘掉目标
        out = await _collect(plugin.zl(ev(f"/zl del {QQ_C}")))
        assert "已删除 1 个目标" in _text_of(out[0]), _text_of(out[0])
        assert len(plugin.state.targets) == 2
        assert QQ_C not in await plugin.storage.get_state("rt_targets")

        # ⑧ del 不存在的目标 → 给出提示而不是静默
        out = await _collect(plugin.zl(ev("/zl del 999999999")))
        assert "没找到" in _text_of(out[0]), _text_of(out[0])

        # ⑨ set 会替换整份清单
        await _collect(plugin.zl(ev(f"/zl set {GID_B} {QQ_C} 阿呆")))
        assert len(plugin.state.targets) == 1
        assert plugin.state.active_target().key == f"{GID_B}:{QQ_C}"

        # ⑩ 目标被删光后，面板回到"未设定目标"引导
        await _collect(plugin.zl(ev(f"/zl del {QQ_C} purge")))
        assert plugin.state.targets == []
        out = await _collect(plugin.zl(ev("/zl")))
        assert "还没有设定蒸馏目标" in _text_of(out[0]), _text_of(out[0])

        await plugin.storage.close()

    asyncio.run(scenario())


def test_zl_del_purge_wipes_corpus_and_persona() -> None:
    """purge 应把语料、Persona、纠正层一并删除。"""

    async def scenario():
        install_fake_astrbot()
        plugin, _ = await _make_plugin(_db("purge.db"), admin_only=False)
        ev = lambda s: FakeEvent(s, group_id=GID_A, sender_id="10001")  # noqa: E731

        await _collect(plugin.zl(ev(f"/zl add {GID_A} {QQ_A} 老王")))
        plugin.state.enabled = True
        plugin.state.listen_enabled = True
        for i in range(3):
            await plugin.collector.handle_event(
                FakeEvent(f"语料{i}", group_id=GID_A, sender_id=QQ_A, message_id=f"p{i}")
            )
        await plugin.collector.flush()
        await plugin.storage.add_correction(GID_A, QQ_A, "他其实很少用句号")
        from core.distiller import empty_snapshot

        await plugin.storage.save_persona(GID_A, QQ_A, empty_snapshot())

        assert await plugin.storage.count_messages(GID_A, QQ_A) == 3
        out = await _collect(plugin.zl(ev(f"/zl del {QQ_A} purge")))
        assert "一并删除" in _text_of(out[0]), _text_of(out[0])

        assert await plugin.storage.count_messages(GID_A, QQ_A) == 0
        assert await plugin.storage.get_persona(GID_A, QQ_A) is None
        assert await plugin.storage.get_corrections(GID_A, QQ_A) == []

        await plugin.storage.close()

    asyncio.run(scenario())


def test_targets_loaded_from_config_on_startup() -> None:
    """启动时应从配置的三路来源读目标，并让运行期存储覆盖配置。"""

    async def scenario():
        install_fake_astrbot()
        cfg_targets = [
            {"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"},
            {"group_id": GID_A, "qq_id": QQ_B, "nickname": "小李"},
        ]
        plugin, _ = await _make_plugin(
            _db("cfgload.db"),
            admin_only=False,
            targets=cfg_targets,
            targets_text=f"{GID_B} {QQ_C} 阿呆",
        )
        await plugin._load_runtime_state()
        assert len(plugin.state.targets) == 3, plugin.state.targets
        assert plugin.state.active_target().key == f"{GID_A}:{QQ_A}"

        # 运行期改过清单后再加载，应以库里的为准
        await plugin._persist_state()
        plugin.state.clear_targets()
        await plugin._load_runtime_state()
        assert len(plugin.state.targets) == 3, "运行期清单未被优先采用"

        # 只留一个目标后重启 → 库里的清单只该有一个
        await _collect(
            plugin.zl(FakeEvent(f"/zl set {GID_B} {QQ_C}", group_id=GID_B, sender_id="1"))
        )
        await plugin._load_runtime_state()
        assert len(plugin.state.targets) == 1
        assert plugin.state.active_target().key == f"{GID_B}:{QQ_C}"

        await plugin.storage.close()

    asyncio.run(scenario())


def test_legacy_single_target_state_migrated() -> None:
    """v0.1.0 留在 state 表里的单目标字段应被迁移成目标清单。"""

    async def scenario():
        install_fake_astrbot()
        plugin, _ = await _make_plugin(_db("legacy_state.db"), admin_only=False)
        # 模拟旧版本写入的 key
        await plugin.storage.set_state("rt_target_group_id", GID_A)
        await plugin.storage.set_state("rt_target_qq_id", QQ_A)
        await plugin.storage.set_state("rt_target_nickname", "老王")

        await plugin._load_runtime_state()
        assert len(plugin.state.targets) == 1, plugin.state.targets
        spec = plugin.state.targets[0]
        assert spec.group_id == GID_A and spec.qq_id == QQ_A and spec.nickname == "老王"

        await plugin.storage.close()

    asyncio.run(scenario())


# =========================================================================== #
# 7. /zl persona 与 /zl push
# =========================================================================== #


def test_zl_persona_and_push_commands() -> None:
    """``/zl persona`` 输出模板；``/zl push`` 写进 AstrBot 人格。"""

    async def scenario():
        install_fake_astrbot()
        plugin, _ = await _make_plugin(_db("persona.db"), admin_only=False)
        mgr = FakePersonaMgr()
        plugin.context.persona_manager = mgr
        ev = lambda s: FakeEvent(s, group_id=GID_A, sender_id="10001")  # noqa: E731

        await _collect(plugin.zl(ev(f"/zl add {GID_A} {QQ_A} 老王")))
        # 还没有档案 → push 应该拒绝并给出引导
        out = await _collect(plugin.zl(ev("/zl push")))
        assert "还没有档案" in _text_of(out[0]), _text_of(out[0])
        assert mgr.personas == {}

        # 造一份档案
        await plugin.storage.save_persona(GID_A, QQ_A, _rich_snapshot())

        # persona：头部 + 体检 + 模板正文
        outs = await _collect(plugin.zl(ev("/zl persona")))
        joined = "\n".join(_text_of(r) for r in outs)
        assert "人格模板" in joined
        assert "绝不发语音" in joined, "模板正文没发出来"
        assert "绝不要说自己是 AI" in joined
        assert f"群友蒸馏-老王-{QQ_A}" in joined, "没给出建议人格 ID"

        # push：写进人格设定
        out = await _collect(plugin.zl(ev("/zl push")))
        msg = _text_of(out[0])
        assert "已新建" in msg, msg
        assert mgr.personas, "人格没有被创建"
        pid, content = next(iter(mgr.personas.items()))
        assert pid == f"群友蒸馏-老王-{QQ_A}", pid
        assert "绝不发语音" in content

        # 再 push 一次 → 走更新分支，不重复创建
        out = await _collect(plugin.zl(ev("/zl push")))
        assert "已更新" in _text_of(out[0]), _text_of(out[0])
        assert len(mgr.personas) == 1

        await plugin.storage.close()

    asyncio.run(scenario())


def test_zl_persona_without_target() -> None:
    """没有目标时 persona / push / list 都应给出引导而不是崩溃。"""

    async def scenario():
        install_fake_astrbot()
        plugin, _ = await _make_plugin(_db("persona_no_target.db"), admin_only=False)
        ev = lambda s: FakeEvent(s, group_id=GID_A, sender_id="10001")  # noqa: E731

        for cmd in ("/zl persona", "/zl push", "/zl list", "/zl profile", "/zl export"):
            out = await _collect(plugin.zl(ev(cmd)))
            assert out, f"{cmd} 无返回"
            text = "\n".join(_text_of(r) for r in out)
            assert text.strip(), f"{cmd} 返回空"
            assert "Traceback" not in text

        await plugin.storage.close()

    asyncio.run(scenario())


# =========================================================================== #
# 8. 蒸馏达标后自动写入人格
# =========================================================================== #


def test_auto_write_persona_after_distill() -> None:
    """完整度达阈值 → 自动写入；未达阈值 → 不写；关闭开关 → 不写。"""

    payload = json.dumps(
        {
            "layer1_rules": [{"text": "绝不发语音", "confidence": 0.9}],
            "layer2_identity": [{"text": "在校生", "confidence": 0.8}],
            "layer3_style": [{"text": "爱用～", "confidence": 0.8}],
            "layer4_behavior": [{"text": "半夜活跃", "confidence": 0.7}],
            "layer5_interests": [{"text": "沉迷抽卡", "confidence": 0.7}],
            "uncertainty": [],
        },
        ensure_ascii=False,
    )

    async def scenario():
        install_fake_astrbot()

        async def build(db_name: str, threshold: int, auto: bool):
            plugin, _ = await _make_plugin(
                _db(db_name),
                provider=FakeProvider(payload),
                admin_only=False,
                persona_template={
                    "enabled": True,
                    "auto_write": auto,
                    "auto_write_threshold": threshold,
                },
            )
            mgr = FakePersonaMgr()
            plugin.context.persona_manager = mgr
            plugin.distiller.set_on_distilled(plugin._on_distilled)
            return plugin, mgr

        # ① 阈值 80（5 层齐全 → 100%）→ 应写入
        plugin, mgr = await build("auto_write_on.db", 80, True)
        plugin.state.group_id = GID_A
        plugin.state.qq_id = QQ_A
        plugin.state.nickname = "老王"
        for i in range(5):
            await plugin.collector.handle_event(
                FakeEvent(f"语料{i}", group_id=GID_A, sender_id=QQ_A, message_id=f"aw{i}")
            )
        await plugin.collector.flush()
        await plugin.distiller._distill_once("umo")
        assert mgr.personas, "达标却没有自动写入人格"
        assert "绝不发语音" in next(iter(mgr.personas.values()))
        await plugin.storage.close()

        # ② 阈值 101（不可能达到）→ 不写入
        plugin2, mgr2 = await build("auto_write_off.db", 101, True)
        plugin2.state.group_id = GID_A
        plugin2.state.qq_id = QQ_A
        for i in range(5):
            await plugin2.collector.handle_event(
                FakeEvent(f"语料{i}", group_id=GID_A, sender_id=QQ_A, message_id=f"bw{i}")
            )
        await plugin2.collector.flush()
        await plugin2.distiller._distill_once("umo")
        assert mgr2.personas == {}, "未达阈值却写了人格"
        # 但档案本身必须正常生成
        assert await plugin2.storage.get_persona(GID_A, QQ_A) is not None
        await plugin2.storage.close()

        # ③ auto_write=False → 不写入
        plugin3, mgr3 = await build("auto_write_disabled.db", 0, False)
        plugin3.state.group_id = GID_A
        plugin3.state.qq_id = QQ_A
        for i in range(5):
            await plugin3.collector.handle_event(
                FakeEvent(f"语料{i}", group_id=GID_A, sender_id=QQ_A, message_id=f"cw{i}")
            )
        await plugin3.collector.flush()
        await plugin3.distiller._distill_once("umo")
        assert mgr3.personas == {}, "开关关着却写了人格"
        await plugin3.storage.close()

    asyncio.run(scenario())


def test_auto_write_failure_does_not_break_distill() -> None:
    """人格写入失败不应影响蒸馏本身（档案照常落库）。"""
    payload = json.dumps(
        {"layer1_rules": [{"text": "低调", "confidence": 0.9}]}, ensure_ascii=False
    )

    async def scenario():
        install_fake_astrbot()
        plugin, _ = await _make_plugin(
            _db("auto_write_boom.db"),
            provider=FakeProvider(payload),
            admin_only=False,
            persona_template={"enabled": True, "auto_write": True, "auto_write_threshold": 0},
        )

        class BoomMgr(FakePersonaMgr):
            async def get_persona(self, persona_id: str):
                raise RuntimeError("炸了")

        plugin.context.persona_manager = BoomMgr()
        plugin.distiller.set_on_distilled(plugin._on_distilled)
        plugin.state.group_id = GID_A
        plugin.state.qq_id = QQ_A
        for i in range(3):
            await plugin.collector.handle_event(
                FakeEvent(f"语料{i}", group_id=GID_A, sender_id=QQ_A, message_id=f"db{i}")
            )
        await plugin.collector.flush()
        await plugin.distiller._distill_once("umo")  # 不应抛出

        snap = await plugin.storage.get_persona(GID_A, QQ_A)
        assert isinstance(snap, dict), "写入人格失败竟导致档案丢失"
        assert snap["layers"]["layer1_rules"][0]["text"] == "低调"
        await plugin.storage.close()

    asyncio.run(scenario())


# =========================================================================== #
# 9. 面板在多目标下的渲染
# =========================================================================== #


def test_panel_shows_target_list_only_when_multiple() -> None:
    """单目标面板保持简洁，多目标才追加清单。"""

    async def scenario():
        install_fake_astrbot()
        plugin, _ = await _make_plugin(_db("panel.db"), admin_only=False)
        ev = lambda s: FakeEvent(s, group_id=GID_A, sender_id="10001")  # noqa: E731

        await _collect(plugin.zl(ev(f"/zl add {GID_A} {QQ_A} 老王")))
        panel = await plugin._build_panel()
        assert "🎯 目标清单（" not in panel, "单目标不该显示清单"
        assert "老王" in panel and QQ_A in panel and GID_A in panel

        await _collect(plugin.zl(ev(f"/zl add {GID_B} {QQ_C} 阿呆")))
        panel2 = await plugin._build_panel()
        assert "🎯 目标清单（2 个" in panel2, panel2
        assert "▶" in panel2 and "老王" in panel2 and "阿呆" in panel2
        # 当前目标（刚添加的那个）应出现在清单之前的详情区
        assert "阿呆" in panel2.split("🎯 目标清单")[0]

        # 目标清单里带上各自的群号，便于区分同名目标
        assert GID_B in panel2

        await plugin.storage.close()

    asyncio.run(scenario())


def main() -> int:
    """直接运行时的入口，逐个执行测试函数。"""
    tests = [
        test_parse_targets_text_unit,
        test_parse_targets_text_dedupe_and_empty,
        test_collect_config_targets_three_sources,
        test_collect_config_targets_tolerates_garbage,
        test_id_validation_rules,
        test_targets_state_roundtrip,
        test_multi_target_two_in_one_group_no_crosstalk,
        test_multi_target_two_groups_each_two,
        test_runtime_state_legacy_single_target_still_works,
        test_storage_migration_adds_context_column,
        test_build_astrbot_persona_content,
        test_build_astrbot_persona_options,
        test_build_persona_summary,
        test_chunk_text_boundaries,
        test_make_persona_id_sanitize,
        test_push_persona_all_paths,
        test_zl_target_management_commands,
        test_zl_del_purge_wipes_corpus_and_persona,
        test_targets_loaded_from_config_on_startup,
        test_legacy_single_target_state_migrated,
        test_zl_persona_and_push_commands,
        test_zl_persona_without_target,
        test_auto_write_persona_after_distill,
        test_auto_write_failure_does_not_break_distill,
        test_panel_shows_target_list_only_when_multiple,
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
        # 兜底清掉可能被兜底逻辑写出的插件目录内 data_local
        shutil.rmtree(PLUGIN_DIR / "data_local", ignore_errors=True)
