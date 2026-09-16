"""蒸馏细节增强 / 人格同步 / 模板填充 相关测试（v0.4.0）。

覆盖本轮新增能力：

- 蒸馏提示词加严：必须有"禁止空话套话"的反面清单，必须要求场景应答样例；
- 语料字符预算 ``distill_max_chars`` 可配、有下限保护；
- 场景应答样例（``speech_samples``）的规整、去重、证据累加、条数上限；
- 结论条目可选情境字段 ``trigger`` / ``avoid`` 在合并全程不丢；
- 人格模板里必须出现场景样例与情境字段，档案/导出/面板同步展示；
- 每日总结结束后**覆盖同步** AstrBot 人格（不看完整度阈值），可关闭；
- 两个填空框的默认模板：Schema 与代码常量不漂移、注释行运行时被剥掉、
  去掉注释后真正生效。

运行方式（任选其一）：

    python tests/test_detail_persona.py
    python -m pytest tests/test_detail_persona.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TMP_ROOT = Path(__file__).resolve().parent / "_tmp_dp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)
QA_TMP = Path(__file__).resolve().parent / "_qa_tmp"

PLUGIN_DIR = ROOT

from tests.test_qa_verify import (  # noqa: E402
    FakeEvent,
    _collect,
    _make_plugin,
    _text_of,
    install_fake_astrbot,
)

GID_A = "987654321"
QQ_A = "123456789"

SCHEMA_PATH = ROOT / "_conf_schema.json"


def _db(name: str) -> Path:
    """给每个用例一个独立的项目内临时库路径。"""
    path = TMP_ROOT / name
    if path.exists():
        path.unlink()
    return path


def _load_schema() -> dict[str, Any]:
    """读插件配置 Schema。"""
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _payload_for(extra: dict[str, Any] | None = None) -> str:
    """造一份结构完整的分析 JSON（含场景样例与情境字段）。"""
    data: dict[str, Any] = {
        "layer1_rules": [
            {
                "text": "绝不发语音",
                "confidence": 0.9,
                "quotes": ["我不发语音的"],
                "trigger": "别人让他发语音时",
                "avoid": "熟人私聊也不会发",
            }
        ],
        "layer2_identity": [{"text": "在校生，二十出头", "confidence": 0.7}],
        "layer3_style": [{"text": "爱用～结尾", "confidence": 0.8, "quotes": ["好呀～"]}],
        "layer4_behavior": [{"text": "半夜最活跃", "confidence": 0.6}],
        "layer5_interests": [{"text": "沉迷抽卡游戏", "confidence": 0.75}],
        "speech_samples": [
            {"situation": "别人问他在干嘛", "reply": "在呢在呢，刚开了一把", "quotes": ["在呢在呢"]},
            {"situation": "被催更", "reply": "别催别催，明天一定", "quotes": ["别催别催"]},
        ],
        "uncertainty": ["工作状态证据不足"],
    }
    if extra:
        data.update(extra)
    return json.dumps(data, ensure_ascii=False)


class RecordingProvider:
    """记录每次收到的提示词，便于断言"到底喂了什么"。"""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    async def text_chat(self, prompt: str = "", system_prompt: str = "", **_kw):
        self.prompts.append(prompt)
        return SimpleNamespace(completion_text=self.payload)


class FakePersonaMgr:
    """伪 AstrBot PersonaManager。"""

    def __init__(self) -> None:
        self.personas: dict[str, str] = {}

    async def get_persona(self, persona_id: str):
        if persona_id not in self.personas:
            raise ValueError("missing")
        return persona_id

    async def create_persona(self, persona_id: str, system_prompt: str, **_k):
        if persona_id in self.personas:
            raise ValueError("exists")
        self.personas[persona_id] = system_prompt
        return persona_id

    async def update_persona(self, persona_id: str, system_prompt: str | None = None, **_k):
        if persona_id not in self.personas:
            raise ValueError("missing")
        if system_prompt is not None:
            self.personas[persona_id] = system_prompt
        return persona_id


def _state_with(specs) -> Any:
    """构造带目标的 RuntimeState。"""
    from core.collector import RuntimeState

    return RuntimeState(enabled=True, listen_enabled=True, targets=list(specs))


def _context_with(provider) -> Any:
    """构造能吐 provider 的最小 Context。"""

    class Ctx:
        async def get_using_provider_async(self, **_kw):
            return provider

    return Ctx()


# =========================================================================== #
# 1. 蒸馏提示词加严
# =========================================================================== #


def test_analyzer_prompt_demands_detail() -> None:
    """分析师提示词必须含"禁止空话套话"的反面清单与场景样例要求。"""
    from core import prompts

    system = prompts.ANALYZER_SYSTEM
    # 反面清单：这些空话被点名禁止
    for banned in ("话不多", "比较活跃", "性格开朗", "喜欢聊天", "偶尔发言"):
        assert banned in system, f"提示词没点名禁止「{banned}」这类空话"
    # 关键要求
    assert "speech_samples" in system
    assert "trigger" in system and "avoid" in system
    assert "照抄" in system or "原样" in system
    # 用户提示词模板也要带上样例字段说明与收尾提醒
    assert "speech_samples" in prompts.ANALYZER_USER_TEMPLATE
    assert "额外约束" in prompts.ANALYZER_USER_TEMPLATE
    assert "怎么说话" in prompts.ANALYZER_USER_TEMPLATE


def test_analyzer_prompt_build_accepts_new_budget() -> None:
    """语料超预算时必须抽样，并把样例字段要求带进提示词。"""
    from core import prompts

    records = [
        {"speaker_name": "老王", "content": "x" * 200, "timestamp": i}
        for i in range(50)
    ]
    prompt = prompts.build_analyzer_user(
        nickname="老王", qq=QQ_A, records=records, max_chars=2000
    )
    assert "抽样" in prompt, "超预算没提示抽样"
    assert "speech_samples" in prompt
    assert len(prompt) < 6000, f"预算没生效，提示词长度 {len(prompt)}"

    small = prompts.build_analyzer_user(
        nickname="老王", qq=QQ_A, records=records[:2], max_chars=20000
    )
    assert "全量" in small, "没超预算却被判为抽样"


def test_distill_max_chars_config() -> None:
    """字符预算：可配、有下限保护、非法值回落默认。"""
    from core.collector import RuntimeState
    from core.distiller import Distiller
    from core.storage import Storage

    async def scenario():
        st = Storage(_db("chars.db"))
        await st.init()
        for cfg, expected in (
            ({}, 12000),
            ({"distill_max_chars": 30000}, 30000),
            ({"distill_max_chars": 500}, 2000),      # 低于下限 → 抬到 2000
            ({"distill_max_chars": 0}, 2000),
            ({"distill_max_chars": "abc"}, 12000),   # 非法 → 默认
            ({"distill_max_chars": None}, 12000),
        ):
            d = Distiller(st, cfg, RuntimeState(), _context_with(RecordingProvider("{}")))
            assert d._max_chars() == expected, (cfg, d._max_chars())
        await st.close()

    asyncio.run(scenario())


# =========================================================================== #
# 2. 场景应答样例与情境字段
# =========================================================================== #


def test_speech_samples_merge() -> None:
    """样例：新条目追加、重复条目只累加证据、非法输入被丢弃。"""
    from core.distiller import empty_snapshot, merge_snapshot

    first = merge_snapshot(
        empty_snapshot(),
        {
            "speech_samples": [
                {"situation": "问他在干嘛", "reply": "在呢在呢", "quotes": ["在呢在呢"]},
                {"situation": "被催更", "reply": "明天一定"},
                "裸字符串样例",
            ]
        },
    )
    samples = first["speech_samples"]
    assert len(samples) == 3, samples
    assert samples[0]["situation"] == "问他在干嘛"
    assert samples[2]["situation"] == "" and samples[2]["reply"] == "裸字符串样例"

    # 语义相同的样例（只差空白/大小写）应当累加证据而不是新增
    second = merge_snapshot(
        first, {"speech_samples": [{"situation": "问他在干嘛", "reply": " 在呢在呢 "}]}
    )
    assert len(second["speech_samples"]) == 3, second["speech_samples"]
    assert second["speech_samples"][0]["evidence"] == 2

    # 非法输入不炸、不污染
    for bad in (None, "abc", 123, [None, 1, {}, {"reply": ""}]):
        snap = merge_snapshot(empty_snapshot(), {"speech_samples": bad})
        assert isinstance(snap["speech_samples"], list)


def test_speech_samples_capped() -> None:
    """样条数超上限时，保留证据最强的那些。"""
    from core import prompts
    from core.distiller import empty_snapshot, merge_snapshot

    many = [
        {"situation": f"情境{i}", "reply": f"回复{i}"}
        for i in range(prompts.MAX_SPEECH_SAMPLES + 15)
    ]
    snap = merge_snapshot(empty_snapshot(), {"speech_samples": many})
    assert len(snap["speech_samples"]) == prompts.MAX_SPEECH_SAMPLES

    # 让其中一条反复出现（证据变强），再灌一批新的，它必须还在
    snap2 = merge_snapshot(snap, {"speech_samples": [many[0]] * 1})
    snap2 = merge_snapshot(snap2, {"speech_samples": [many[0]] * 1})
    snap3 = merge_snapshot(
        snap2,
        {"speech_samples": [{"situation": f"新增{i}", "reply": f"新回复{i}"} for i in range(20)]},
    )
    keys = [(s.get("situation"), s.get("reply")) for s in snap3["speech_samples"]]
    assert ("情境0", "回复0") in keys, "证据最强的样例被裁掉了"
    assert len(snap3["speech_samples"]) == prompts.MAX_SPEECH_SAMPLES


def test_trigger_avoid_preserved() -> None:
    """trigger / avoid 在规整、合并、复读（二次合并）全程不能丢。"""
    from core.distiller import empty_snapshot, merge_snapshot

    snap = merge_snapshot(
        empty_snapshot(),
        {
            "layer1_rules": [
                {
                    "text": "绝不发语音",
                    "confidence": 0.9,
                    "trigger": "被要求发语音时",
                    "avoid": "熟人也不会",
                }
            ]
        },
    )
    item = snap["layers"]["layer1_rules"][0]
    assert item["trigger"] == "被要求发语音时"
    assert item["avoid"] == "熟人也不会"

    # 再合并一轮（会先 _copy_layers 复读已有条目）后仍要在
    snap2 = merge_snapshot(snap, {"layer3_style": [{"text": "爱用～", "confidence": 0.8}]})
    assert snap2["layers"]["layer1_rules"][0]["trigger"] == "被要求发语音时"

    # 已有条目缺情境字段时，新证据应当补上
    snap3 = merge_snapshot(
        snap2,
        {
            "layer1_rules": [
                {"text": "绝不发语音", "confidence": 0.9, "avoid": "公开场合也绝不发"}
            ]
        },
    )
    assert snap3["layers"]["layer1_rules"][0]["avoid"] == "熟人也不会", "不该被覆盖"
    assert snap3["layers"]["layer1_rules"][0]["evidence"] == 2


# =========================================================================== #
# 3. 人格模板把新素材用起来
# =========================================================================== #


def _rich_snapshot():
    from core.distiller import empty_snapshot, merge_snapshot

    return merge_snapshot(
        empty_snapshot(), json.loads(_payload_for()), ["他其实偶尔也用语音"]
    )


def test_persona_template_includes_samples_and_context() -> None:
    """人格模板要出现场景样例与情境字段，且不默认泄露原话。"""
    from core import prompts

    text = prompts.build_astrbot_persona(_rich_snapshot(), "老王", QQ_A)
    assert "六、他被人这么问时" in text, text[:400]
    assert "别人问他在干嘛" in text and "在呢在呢，刚开了一把" in text
    assert "出现时机：别人让他发语音时" in text
    assert "什么时候不这样：熟人私聊也不会发" in text
    assert "怎么用这份档案" in text
    # 默认不带证据原话
    assert "我不发语音的" not in text

    with_quotes = prompts.build_astrbot_persona(
        _rich_snapshot(), "老王", QQ_A, include_evidence=True
    )
    assert "我不发语音的" in with_quotes


def test_persona_summary_counts_samples() -> None:
    """体检报告要报出场景样例条数。"""
    from core import prompts

    summary = prompts.build_persona_summary(_rich_snapshot(), "老王", QQ_A)
    assert "场景应答样例：2 条" in summary, summary
    assert "5/5" in summary

    thin = prompts.build_persona_summary({"layers": {}}, "老王", QQ_A)
    assert "场景应答样例：0 条" in thin, thin
    assert "建议再攒点语料" in thin


def test_profile_and_export_show_samples() -> None:
    """``/zl profile`` 与导出的 Markdown 都要带上场景样例。"""
    from core import progress, prompts

    snap = _rich_snapshot()
    profile = progress.render_profile(snap, "老王", QQ_A, GID_A)
    assert "场景应答样例" in profile
    assert "别人问他在干嘛" in profile
    assert "应答样例 2 条" in profile
    # 原有结论不能被挤掉
    assert "绝不发语音" in profile
    assert "他其实偶尔也用语音" in profile

    md = prompts.render_export_markdown(snap, "老王", QQ_A)
    assert "## 场景应答样例" in md
    assert "在呢在呢，刚开了一把" in md


# =========================================================================== #
# 4. 每日总结后同步覆盖人格
# =========================================================================== #


def test_daily_digest_syncs_persona_by_default() -> None:
    """每日总结跑完 → 人格被覆盖写入 AstrBot（不看完整度阈值）。"""

    async def scenario():
        install_fake_astrbot()
        plugin, _ = await _make_plugin(
            _db("sync_on.db"),
            admin_only=False,
            provider=RecordingProvider(_payload_for()),
            daily_digest={"enabled": True, "time": "08:00", "min_messages": 0},
            targets=[{"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"}],
        )
        await plugin._load_runtime_state()
        mgr = FakePersonaMgr()
        plugin.context.persona_manager = mgr  # type: ignore[attr-defined]

        from core.storage import MessageRecord

        await plugin.storage.insert_message(
            MessageRecord("m1", GID_A, QQ_A, "老王", "今天聊了不少", timestamp=int(time.time()))
        )

        assert await plugin._run_daily_digest(time.time(), 0) is True
        assert mgr.personas, "每日总结结束却没有同步人格"
        pid, content = next(iter(mgr.personas.items()))
        assert pid == f"群友蒸馏-老王-{QQ_A}", pid
        assert "绝不发语音" in content
        # 场景样例也要进人格
        assert "在呢在呢，刚开了一把" in content, "人格里没有场景样例"
        await plugin.storage.close()

    asyncio.run(scenario())


def test_daily_digest_sync_can_be_disabled() -> None:
    """sync_after_digest=False 时不写人格；enabled=False 时同理。"""

    async def scenario():
        install_fake_astrbot()

        async def run(name: str, persona_cfg: dict[str, Any]):
            plugin, _ = await _make_plugin(
                _db(name),
                admin_only=False,
                provider=RecordingProvider(_payload_for()),
                daily_digest={"enabled": True, "time": "08:00", "min_messages": 0},
                persona_template=persona_cfg,
                targets=[{"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"}],
            )
            await plugin._load_runtime_state()
            mgr = FakePersonaMgr()
            plugin.context.persona_manager = mgr  # type: ignore[attr-defined]

            from core.storage import MessageRecord

            await plugin.storage.insert_message(
                MessageRecord("m1", GID_A, QQ_A, "老王", "今天的话", timestamp=int(time.time()))
            )
            assert await plugin._run_daily_digest(time.time(), 0) is True
            got = len(mgr.personas)
            await plugin.storage.close()
            return got

        assert await run("sync_off.db", {"sync_after_digest": False}) == 0
        assert await run("sync_disabled.db", {"enabled": False}) == 0

    asyncio.run(scenario())


def test_daily_digest_sync_ignores_completeness_threshold() -> None:
    """明确区分：auto_write 看阈值，sync_after_digest 不看。

    只蒸出一层结论（完整度 20%）时：
      - auto_write + 阈值 80 → 不写；
      - sync_after_digest（每日总结路径）→ 照写。
    """

    async def scenario():
        install_fake_astrbot()
        thin = json.dumps(
            {"layer1_rules": [{"text": "绝不发语音", "confidence": 0.9}]},
            ensure_ascii=False,
        )
        plugin, _ = await _make_plugin(
            _db("threshold.db"),
            admin_only=False,
            provider=RecordingProvider(thin),
            daily_digest={"enabled": True, "time": "08:00", "min_messages": 0},
            persona_template={"auto_write": True, "auto_write_threshold": 80},
            targets=[{"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"}],
        )
        await plugin._load_runtime_state()
        mgr = FakePersonaMgr()
        plugin.context.persona_manager = mgr  # type: ignore[attr-defined]

        from core.storage import MessageRecord

        await plugin.storage.insert_message(
            MessageRecord("m1", GID_A, QQ_A, "老王", "只说了一句", timestamp=int(time.time()))
        )

        assert await plugin._run_daily_digest(time.time(), 0) is True
        assert len(mgr.personas) == 1, "每日总结路径应当无视阈值照样同步"
        await plugin.storage.close()

    asyncio.run(scenario())


def test_daily_digest_sync_skips_targets_without_persona() -> None:
    """某目标当天没语料（没档案）时不该硬造一个人格出来。"""

    async def scenario():
        install_fake_astrbot()
        plugin, _ = await _make_plugin(
            _db("sync_skip.db"),
            admin_only=False,
            provider=RecordingProvider(_payload_for()),
            daily_digest={"enabled": True, "time": "08:00", "min_messages": 0},
            targets=[{"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"}],
        )
        await plugin._load_runtime_state()
        mgr = FakePersonaMgr()
        plugin.context.persona_manager = mgr  # type: ignore[attr-defined]

        assert await plugin._run_daily_digest(time.time(), 0) is True
        assert mgr.personas == {}, "没语料也硬写了人格"
        await plugin.storage.close()

    asyncio.run(scenario())


# =========================================================================== #
# 5. 两个填空框的默认模板
# =========================================================================== #


def test_schema_defaults_match_code_templates() -> None:
    """默认模板在 Schema 与代码常量之间不能漂移（防两边各改一半）。"""
    from core import prompts

    schema = _load_schema()
    assert schema["custom_prompt_extra"]["default"] == prompts.DEFAULT_CUSTOM_PROMPT_EXTRA
    assert (
        schema["persona_template"]["items"]["extra_rules"]["default"]
        == prompts.DEFAULT_PERSONA_EXTRA_RULES
    )
    # 两个框的默认值都必须非空（用户反馈"输入框是空的"）
    assert schema["custom_prompt_extra"]["default"].strip()
    assert schema["persona_template"]["items"]["extra_rules"]["default"].strip()


def test_default_templates_are_inert_until_enabled() -> None:
    """默认模板整体是注释 → 运行时被剥成空串，不会擅自加约束。"""
    from core import prompts

    assert prompts.strip_template_comments(prompts.DEFAULT_CUSTOM_PROMPT_EXTRA) == ""
    assert prompts.strip_template_comments(prompts.DEFAULT_PERSONA_EXTRA_RULES) == ""


def test_strip_template_comments_rules() -> None:
    """注释剥离规则：只吃行首 # / //，正文里的 # 不受影响。"""
    from core import prompts

    text = "\n".join(
        [
            "# 整行注释",
            "// 另一种注释",
            "   # 缩进后的注释也算",
            "有效内容一",
            "",
            "有效内容二 # 行中间的话题标签要保留",
            "   ",
        ]
    )
    out = prompts.strip_template_comments(text)
    assert out.splitlines() == ["有效内容一", "", "有效内容二 # 行中间的话题标签要保留"]
    assert prompts.strip_template_comments("") == ""
    assert prompts.strip_template_comments(None) == ""
    assert prompts.strip_template_comments("# 只有注释") == ""
    assert prompts.strip_template_comments("   ") == ""


def test_uncommented_template_takes_effect() -> None:
    """去掉行首 # 之后，「人格追加规矩」真正进入人格模板。"""

    async def scenario():
        install_fake_astrbot()
        from core import prompts
        from core.storage import MessageRecord

        assert prompts.DEFAULT_PERSONA_EXTRA_RULES.startswith("#")
        assert "每次回复不超过两句话" in prompts.strip_template_comments(
            "# 注释\n- 每次回复不超过两句话"
        )

        plugin, _ = await _make_plugin(
            _db("uncomment.db"),
            admin_only=False,
            provider=RecordingProvider(_payload_for()),
            persona_template={"extra_rules": "# 注释\n- 每次回复不超过两句话"},
            targets=[{"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"}],
        )
        await plugin._load_runtime_state()
        await plugin.storage.insert_message(
            MessageRecord("m1", GID_A, QQ_A, "老王", "今天的话", timestamp=int(time.time()))
        )
        await plugin.distiller._distill_once("umo")

        snap = await plugin.storage.get_persona(GID_A, QQ_A)
        assert snap is not None
        persona = plugin._build_persona_text(plugin.state.active_target(), snap)
        assert "每次回复不超过两句话" in persona
        assert "# 注释" not in persona
        await plugin.storage.close()

    asyncio.run(scenario())


def test_extra_constraint_reaches_analyzer_prompt() -> None:
    """追加蒸馏约束必须真的进到发给 LLM 的提示词里。"""
    from core.collector import RuntimeState
    from core.distiller import Distiller
    from core.storage import MessageRecord, Storage
    from core.targets import TargetSpec

    async def scenario():
        st = Storage(_db("extra_prompt.db"))
        await st.init()
        await st.insert_message(
            MessageRecord("m1", GID_A, QQ_A, "老王", "今天的话", timestamp=int(time.time()))
        )
        provider = RecordingProvider(_payload_for())
        d = Distiller(
            st,
            {"custom_prompt_extra": "# 注释行\n- 多关注他吐槽的语气"},
            RuntimeState(enabled=True, targets=[TargetSpec(GID_A, QQ_A, "老王")]),
            _context_with(provider),
        )
        await d._distill_once("umo")
        assert provider.prompts, "没调 LLM"
        assert "多关注他吐槽的语气" in provider.prompts[0]
        assert "# 注释行" not in provider.prompts[0]
        await st.close()

    asyncio.run(scenario())


def main() -> int:
    """直接运行时的入口，逐个执行测试函数。"""
    tests = [
        test_analyzer_prompt_demands_detail,
        test_analyzer_prompt_build_accepts_new_budget,
        test_distill_max_chars_config,
        test_speech_samples_merge,
        test_speech_samples_capped,
        test_trigger_avoid_preserved,
        test_persona_template_includes_samples_and_context,
        test_persona_summary_counts_samples,
        test_profile_and_export_show_samples,
        test_daily_digest_syncs_persona_by_default,
        test_daily_digest_sync_can_be_disabled,
        test_daily_digest_sync_ignores_completeness_threshold,
        test_daily_digest_sync_skips_targets_without_persona,
        test_schema_defaults_match_code_templates,
        test_default_templates_are_inert_until_enabled,
        test_strip_template_comments_rules,
        test_uncommented_template_takes_effect,
        test_extra_constraint_reaches_analyzer_prompt,
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
        shutil.rmtree(QA_TMP, ignore_errors=True)
        shutil.rmtree(PLUGIN_DIR / "data_local", ignore_errors=True)
