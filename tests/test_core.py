"""核心逻辑单元测试（不依赖 AstrBot 本体）。

运行方式（任选其一）：

    python tests/test_core.py
    python -m pytest tests/test_core.py

覆盖：存储读写与去重、增量 merge、完整度计算、进度条、语料抽样、JSON 解析。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 项目硬性目录纪律：任何文件都不得落到 E:\Zhengliu 之外（含系统临时目录）。
# 因此测试用临时目录建在插件根内 tests/_tmp_core/，用例结束后立即删除。
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 即 E:\Zhengliu
TMP_CORE = Path(__file__).resolve().parent / "_tmp_core"


def _assert_under_project(*paths: Path) -> None:
    """断言给定路径都位于项目根目录之内。

    Windows 路径大小写不敏感，故统一转小写后再比较，并追加分隔符避免
    前缀误判（例如 ``E:\\Zhengliu2`` 不会被误认为在 ``E:\\Zhengliu`` 内）。
    """
    root = str(PROJECT_ROOT.resolve()).lower()
    for p in paths:
        resolved = str(Path(p).resolve()).lower()
        assert resolved == root or resolved.startswith(
            root + os.sep
        ), f"路径越界（不在项目 {PROJECT_ROOT} 内）: {p}"

from core import progress, prompts  # noqa: E402
from core.distiller import (  # noqa: E402
    compute_completeness,
    count_formed_layers,
    empty_snapshot,
    merge_snapshot,
    parse_analysis,
)
from core.storage import MessageRecord, Storage  # noqa: E402


def _rec(
    mid: str,
    content: str,
    *,
    ts: int = 0,
    qq: str = "123",
    name: str = "张三",
    ctx: bool = False,
    gid: str = "999",
) -> MessageRecord:
    """构造一条测试用消息记录。"""
    return MessageRecord(
        message_id=mid,
        group_id=gid,
        speaker_qq=qq,
        speaker_name=name,
        content=content,
        raw_type="text",
        timestamp=ts,
        created_at=ts,
        is_context=ctx,
    )


def test_storage_basic() -> None:
    """存储层：写入、去重、统计、标记、纠正、Persona、清空。

    临时库建在项目内 ``tests/_tmp_core/``，绝不写系统临时目录。
    """
    # 准备项目内临时目录，并清理上一次可能残留的内容
    TMP_CORE.mkdir(parents=True, exist_ok=True)
    for old in TMP_CORE.iterdir():
        if old.is_dir():
            shutil.rmtree(old, ignore_errors=True)
        else:
            old.unlink(missing_ok=True)

    async def _scenario() -> None:
        db = TMP_CORE / "t.db"
        # 硬性断言：临时库与临时目录都必须落在项目目录树之内
        _assert_under_project(db, TMP_CORE)

        st = Storage(db)
        await st.init()
        _assert_under_project(Path(st._db_path))

        assert await st.insert_message(_rec("m1", "hello")) is True
        assert await st.insert_message(_rec("m1", "hello")) is False  # 去重
        assert await st.insert_message(_rec("m2", "world", ts=100)) is True
        # 上下文消息不计入目标语料
        assert await st.insert_message(_rec("c1", "上文", ts=50, ctx=True)) is True

        assert await st.count_messages("999", "123") == 2
        assert await st.count_undistilled("999", "123") == 2

        rows = await st.fetch_undistilled("999", "123", 10)
        assert len(rows) == 3  # 2 条目标 + 1 条上下文
        await st.mark_distilled([r["id"] for r in rows])
        assert await st.count_undistilled("999", "123") == 0

        lo, hi = await st.get_time_span("999", "123")
        assert lo == 0 and hi == 100

        assert await st.add_correction("999", "123", "他其实很少用句号") is True
        assert await st.get_corrections("999", "123") == ["他其实很少用句号"]

        assert await st.save_persona("999", "123", empty_snapshot()) is True
        snap = await st.get_persona("999", "123")
        assert isinstance(snap, dict)
        assert snap["meta"]["distill_round"] == 0

        assert await st.set_state("rt_target_qq_id", "123") is True
        assert await st.get_state("rt_target_qq_id") == "123"

        removed = await st.clear_messages("999", "123")
        assert removed >= 2
        assert await st.count_messages("999", "123") == 0

        await st.close()

    try:
        asyncio.run(_scenario())
    finally:
        shutil.rmtree(TMP_CORE, ignore_errors=True)


def test_merge_snapshot_incremental() -> None:
    """增量 merge：证据累加、轮次自增、层数统计、完整度。"""
    base = empty_snapshot()
    analysis = {
        "layer3_style": [
            {"text": "喜欢用～结尾", "confidence": 0.8, "quotes": ["好呀～"]}
        ],
        "uncertainty": ["identity 证据不足"],
    }

    merged = merge_snapshot(base, analysis, ["他是程序员"])
    assert merged["meta"]["distill_round"] == 1
    assert merged["layers"]["layer3_style"][0]["text"] == "喜欢用～结尾"
    assert merged["layers"]["layer3_style"][0]["evidence"] == 1
    assert merged["corrections"] == ["他是程序员"]
    assert merged["uncertainty"] == ["identity 证据不足"]

    # 相同结论再次出现 → 证据累加而不是新增条目
    merged2 = merge_snapshot(merged, analysis, ["他是程序员"])
    assert merged2["meta"]["distill_round"] == 2
    assert len(merged2["layers"]["layer3_style"]) == 1
    assert merged2["layers"]["layer3_style"][0]["evidence"] == 2

    assert count_formed_layers(merged2) == 1
    assert compute_completeness(merged2) == 20


def test_merge_conflict_flag() -> None:
    """冲突标注应被保留。"""
    snap = merge_snapshot(
        empty_snapshot(),
        {"layer3_style": [{"text": "从不用句号", "confidence": 0.6, "conflict": True}]},
    )
    assert snap["layers"]["layer3_style"][0]["conflict"] is True


def test_array_fields_not_split() -> None:
    """数组字段被误传为字符串时，应整体作为一条，绝不逐字符拆散。"""
    snap = merge_snapshot(empty_snapshot(), {"layer1_rules": [], "uncertainty": "abc"})
    assert snap["uncertainty"] == ["abc"]

    snap2 = merge_snapshot(
        empty_snapshot(),
        {"layer3_style": [{"text": "爱用叠词", "confidence": 0.7, "quotes": "好呀～"}]},
    )
    assert snap2["layers"]["layer3_style"][0]["quotes"] == ["好呀～"]


def test_progress_bar() -> None:
    """进度条边界：0%、100%、50%，以及饱和基准为 0。"""
    assert progress.make_bar(0, 100) == "░" * 10
    assert progress.make_bar(100, 100) == "█" * 10
    assert progress.make_bar(50, 100) == "█" * 5 + "░" * 5
    assert progress.make_bar(10, 0) == "█" * 10  # 饱和基准 0 视为满


def test_panel_and_profile_render() -> None:
    """面板与档案渲染不抛异常且包含关键信息。"""
    data = progress.PanelData(
        has_target=True,
        nickname="张三",
        qq="123456789",
        group_id="987654321",
        total=1284,
        saturation=1500,
        round_no=3,
        last_distill_ts=1700000000,
        queue=84,
        min_ts=1700000000,
        max_ts=1701300000,
        completeness=60,
        formed_layers=3,
    )
    panel = progress.render_panel(data)
    assert "张三" in panel
    assert "1,284" in panel
    assert "60%" in panel

    assert "还没有设定蒸馏目标" in progress.render_no_target()
    assert "/zl help" in progress.render_help()

    snap = merge_snapshot(
        empty_snapshot(),
        {"layer1_rules": [{"text": "绝不发语音", "confidence": 0.9}]},
        ["他偶尔也用语音"],
    )
    profile = progress.render_profile(snap, "张三", "123456789")
    assert "绝不发语音" in profile
    assert "他偶尔也用语音" in profile


def test_transcript_sampling() -> None:
    """超长语料应触发抽样，并在说明中如实声明。"""
    records = [
        {
            "speaker_name": f"n{i}",
            "content": "x" * 100,
            "timestamp": i,
            "is_context": False,
        }
        for i in range(100)
    ]
    text, note = prompts.format_transcript(records, max_chars=500)
    assert "抽样" in note
    assert len(text) <= 700

    short = [{"speaker_name": "a", "content": "hi", "timestamp": 1}]
    text2, note2 = prompts.format_transcript(short, max_chars=500)
    assert "全量" in note2
    assert "hi" in text2


def test_parse_analysis() -> None:
    """JSON 抽取：代码块、前后缀、非法输入。"""
    assert parse_analysis('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_analysis('前缀 {"x": [1, 2]} 后缀') == {"x": [1, 2]}
    assert parse_analysis("no json here") is None
    assert parse_analysis("") is None
    assert parse_analysis("[1, 2, 3]") is None


def test_analyzer_prompt_build() -> None:
    """提示词组装：占位符全部被填充，不含裸花括号。"""
    user = prompts.build_analyzer_user(
        nickname="张三",
        qq="123",
        records=[{"speaker_name": "张三", "content": "你好", "timestamp": 1}],
        existing_snapshot=None,
        extra="",
    )
    assert "张三" in user
    assert "Layer 1" in user
    assert "{nickname}" not in user


def main() -> int:
    """直接运行时的入口，逐个执行测试函数。"""
    tests = [
        test_storage_basic,
        test_merge_snapshot_incremental,
        test_merge_conflict_flag,
        test_array_fields_not_split,
        test_progress_bar,
        test_panel_and_profile_render,
        test_transcript_sampling,
        test_parse_analysis,
        test_analyzer_prompt_build,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {exc!r}")
        else:
            print(f"PASS  {test.__name__}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
