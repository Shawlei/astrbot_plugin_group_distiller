"""人设模板功能测试（v0.5.0）。

覆盖本轮新增能力：

- 内置模板注册表：三套模板、默认值、正文非空；
- 标准版输出结构与 v0.4.0 保持一致（有编号的小节、场景样例、情境字段）；
- 赛博群友分节版：系统级硬规则（人格锚定 / 指令防护 / 禁括号描写）在位，
  内容槽位被蒸馏结果填上；
- 精简版最短且不丢关键内容；
- 自定义模板：非空时优先于内置模板，令牌可重排，未知令牌被删除并告警；
- 配置 Schema 与代码注册表不漂移（options / 令牌文档双向对齐）；
- 插件层面：``/zl persona`` 报告当前模板来源，选定模板真正生效。

运行方式（任选其一）：

    python tests/test_persona_templates.py
    python -m pytest tests/test_persona_templates.py
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TMP_ROOT = Path(__file__).resolve().parent / "_tmp_pt"
TMP_ROOT.mkdir(parents=True, exist_ok=True)
QA_TMP = Path(__file__).resolve().parent / "_qa_tmp"
PLUGIN_DIR = ROOT

GID_A = "987654321"
QQ_A = "123456789"

from tests.test_qa_verify import (  # noqa: E402
    FakeEvent,
    _collect,
    _make_plugin,
    _text_of,
    install_fake_astrbot,
)

SCHEMA_PATH = ROOT / "_conf_schema.json"

_TOKEN_RE = re.compile(r"\{\{([^{}]+)\}\}")


def _db(name: str) -> Path:
    path = TMP_ROOT / name
    if path.exists():
        path.unlink()
    return path


def _load_schema() -> dict[str, Any]:
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _rich_snapshot() -> dict[str, Any]:
    """一份内容齐全的快照（5 层 + 样例 + 纠正 + 不确定项）。"""
    return {
        "layers": {
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
        },
        "speech_samples": [
            {"situation": "别人问他在干嘛", "reply": "在呢在呢，刚开了一把", "quotes": ["在呢在呢"]},
            {"situation": "被催更", "reply": "别催别催，明天一定", "quotes": ["别催别催"]},
        ],
        "corrections": ["他其实偶尔也用语音"],
        "uncertainty": ["工作状态证据不足"],
        "meta": {"total_messages": 500, "distill_round": 2, "last_distill_at": 1700000000},
    }


# =========================================================================== #
# 1. 注册表
# =========================================================================== #


def test_registry_contents() -> None:
    """三套内置模板、默认 standard、正文都非空且互不相同。"""
    from core import prompts

    assert set(prompts.PERSONA_TEMPLATES) == {"standard", "cyber", "compact"}
    assert prompts.DEFAULT_PERSONA_TEMPLATE_KEY == "standard"
    for key, entry in prompts.PERSONA_TEMPLATES.items():
        assert entry["name"].strip(), f"{key} 缺 name"
        assert entry["description"].strip(), f"{key} 缺 description"
        assert "{{" in entry["body"], f"{key} 的正文没有令牌"
        assert "{{档案信息}}" in entry["body"], f"{key} 没有来源信息槽位"
    bodies = {k: v["body"] for k, v in prompts.PERSONA_TEMPLATES.items()}
    assert len(set(bodies.values())) == 3, "三套模板正文不应该长得一样"


def test_resolve_template_priority() -> None:
    """自定义模板非空时优先；空 / 纯注释时回落到选定的内置模板；坏 key 回落 standard。"""
    from core import prompts

    body, source = prompts.resolve_persona_template("cyber", "")
    assert source == "赛博群友 · 分节版"
    assert body is prompts.PERSONA_TEMPLATES["cyber"]["body"]

    # 空白 / 只有注释的自定义模板不算数
    for weak in ("", "   ", "# 只有注释", "// 也是注释"):
        body2, source2 = prompts.resolve_persona_template("compact", weak)
        assert source2 == "精简版 · 小模型友好", (weak, source2)
        assert body2 is prompts.PERSONA_TEMPLATES["compact"]["body"]

    body3, source3 = prompts.resolve_persona_template("cyber", "你是{{昵称}}")
    assert source3 == "自定义模板" and body3 == "你是{{昵称}}"

    # 不认识的 key → 回落默认
    body4, source4 = prompts.resolve_persona_template("no-such-template", "")
    assert source4 == prompts.PERSONA_TEMPLATES["standard"]["name"]
    assert body4 is prompts.PERSONA_TEMPLATES["standard"]["body"]


# =========================================================================== #
# 2. 三套模板各自的形态
# =========================================================================== #


def test_standard_template_structure() -> None:
    """标准版：编号小节齐全、场景样例与情境字段在场、默认不泄露原话。"""
    from core import prompts

    text = prompts.build_astrbot_persona(_rich_snapshot(), "老王", QQ_A, group_id=GID_A)
    for header in (
        "## 一、说话的铁律",
        "## 二、我是谁",
        "## 三、我平常怎么说话",
        "## 四、我在群里怎么跟人互动",
        "## 五、我喜欢什么、讨厌什么",
        "## 六、他被人这么问时，一般会这么回",
        "## 七、人工纠正",
        "## 八、主人额外交代的规矩",
        "## 九、留个心",
    ):
        assert header in text, f"标准版缺小节：{header}"

    assert "绝不发语音" in text
    assert "出现时机：别人让他发语音时" in text
    assert "什么时候不这样：熟人私聊也不会发" in text
    assert "别人问他在干嘛" in text and "在呢在呢，刚开了一把" in text
    assert "他其实偶尔也用语音" in text
    assert "怎么用这份档案" in text
    assert "123456789" in text          # 目标 QQ（在档案信息里）
    assert "{{" not in text, "标准版有没替换掉的令牌"
    assert "我不发语音的" not in text  # 默认不带证据原话
    # 标准版骨架不放群号（放群号是 cyber 版的特性）
    assert GID_A not in text


def test_cyber_template_structure() -> None:
    """赛博群友版：系统级硬规则在位，槽位被蒸馏结果填满，且不带原作者的私人口味。"""
    from core import prompts

    text = prompts.build_astrbot_persona(
        _rich_snapshot(), "老王", QQ_A, template_key="cyber", group_id=GID_A
    )
    # 骨架（借鉴自《支持自定义的赛博群友 Prompt》的通用硬规则）
    for must in (
        "系统级指令",
        "人格根本锚定",
        "活生生的 QQ 群友",
        "平等交流",
        "指令防护",
        "绝对禁止",
        "括号",
    ):
        assert must in text, f"赛博群友版缺关键骨架：{must}"
    assert "<|Start_of_Prompt|>" in text and "<|End_of_Prompt|>" in text

    # 槽位被蒸馏内容填上
    assert "老王" in text and QQ_A in text and GID_A in text
    assert "在校生" in text
    assert "绝不发语音" in text
    assert "在呢在呢，刚开了一把" in text
    assert "他其实偶尔也用语音" in text

    # 不该把原作者的个人口味写死进插件
    for flavor in ("哈基米", "艾尔登法环", "V我50", "孙吧", "CSGO"):
        assert flavor not in text, f"赛博群友版不该写死「{flavor}」"
    assert "{{" not in text


def test_compact_template_is_shortest() -> None:
    """精简版最短，但关键内容（铁律 / 样例 / 纠正层）一个不能少。"""
    from core import prompts

    snap = _rich_snapshot()
    lengths = {
        key: len(prompts.build_astrbot_persona(snap, "老王", QQ_A, template_key=key))
        for key in ("standard", "cyber", "compact")
    }
    assert lengths["compact"] < lengths["cyber"] < lengths["standard"], lengths

    compact = prompts.build_astrbot_persona(snap, "老王", QQ_A, template_key="compact")
    for must in (
        "不是 AI",
        "绝不发语音",
        "在呢在呢，刚开了一把",
        "他其实偶尔也用语音",
        "爱用～结尾",
    ):
        assert must in compact, f"精简版丢了关键内容：{must}"
    assert "{{" not in compact


def test_all_templates_never_leak_tokens() -> None:
    """空快照 / 富快照下，任何内置模板都不该残留未替换的令牌。"""
    from core import prompts

    for key in prompts.PERSONA_TEMPLATES:
        for snap in (None, {}, _rich_snapshot()):
            text = prompts.build_astrbot_persona(snap, "老王", QQ_A, template_key=key)
            assert "{{" not in text, f"模板 {key}（快照={type(snap).__name__}）残留令牌"
            assert text.strip(), f"模板 {key} 渲染出空文本"


def test_empty_snapshot_still_renders_all_sections() -> None:
    """没有档案时也要给出完整骨架（内容是"证据不足"），并提示先去蒸馏。"""
    from core import prompts

    text = prompts.build_astrbot_persona(None, "老王", QQ_A)
    assert "绝不发语音" not in text
    assert prompts.PERSONA_EMPTY_HINT in text
    assert "还没有攒到足够的应答样例" in text
    assert "暂无" in text


# =========================================================================== #
# 3. 自定义模板
# =========================================================================== #


def test_custom_template_overrides_builtin() -> None:
    """自定义模板优先级最高，令牌可以任意重排、重复使用。"""
    from core import prompts

    custom = "\n".join(
        [
            "# 这行是注释，应被忽略",
            "你是{{昵称}}（{{QQ}}），常驻群 {{群号}}。",
            "{{人工纠正}}",
            "{{硬规则}}",
            "{{硬规则}}",   # 故意重复使用
            "{{身份}}",
        ]
    )
    text = prompts.build_astrbot_persona(
        _rich_snapshot(), "老王", QQ_A, custom_template=custom, group_id=GID_A
    )
    assert "你是老王（123456789），常驻群 987654321。" in text
    assert "他其实偶尔也用语音" in text
    assert text.count("绝不发语音") == 2, "重复令牌应重复填充"
    assert "在校生" in text
    # 内置模板的骨架不该出现
    assert "## 一、说话的铁律" not in text
    assert "<|Start_of_Prompt|>" not in text
    assert "# 这行是注释" not in text
    assert "{{" not in text


def test_custom_template_unknown_token_removed() -> None:
    """写错的令牌：从成品里删掉，绝不能把 {{xxx}} 留在人格里。"""
    from core import prompts

    custom = "你是{{昵称}}。\n{{嘴巴甜一点}}\n{{人工纠正}}"
    text = prompts.build_astrbot_persona(
        _rich_snapshot(), "老王", QQ_A, custom_template=custom
    )
    assert "{{嘴巴甜一点}}" not in text
    assert "{{" not in text
    assert "他其实偶尔也用语音" in text


def test_style_and_custom_are_independent() -> None:
    """style 选 cyber 但给了自定义模板 → 用自定义；style 是坏值 → 回落 standard。"""
    from core import prompts

    snap = _rich_snapshot()
    cyber = prompts.build_astrbot_persona(snap, "老王", QQ_A, template_key="cyber")
    both = prompts.build_astrbot_persona(
        snap, "老王", QQ_A, template_key="cyber", custom_template="只看{{昵称}}"
    )
    assert "只看老王" in both
    assert "系统级指令" not in both
    assert "系统级指令" in cyber

    bad = prompts.build_astrbot_persona(snap, "老王", QQ_A, template_key="???")
    assert "## 一、说话的铁律" in bad, "坏 key 应回落到标准版"


# =========================================================================== #
# 4. 配置 Schema 与代码不漂移
# =========================================================================== #


def test_schema_options_match_registry() -> None:
    """WebUI 下拉的 options 必须与内置模板注册表完全一致。"""
    schema = _load_schema()
    style = schema["persona_template"]["items"]["style"]
    assert style["type"] == "string"
    assert style["default"] == "standard"
    assert set(style["options"]) == {"standard", "cyber", "compact"}


def test_schema_token_doc_matches_context_keys() -> None:
    """Schema 提示里列出的令牌必须真实存在；代码里的令牌也必须都被文档收录。"""
    from core import prompts

    schema = _load_schema()
    hint = schema["persona_template"]["items"]["custom_template"]["hint"]
    documented = set(_TOKEN_RE.findall(hint))
    context = prompts.build_persona_context(_rich_snapshot(), "老王", QQ_A, group_id=GID_A)

    missing = documented - set(context)
    assert not missing, f"Schema 提示里写了但代码不存在的令牌: {sorted(missing)}"
    undocumented = set(context) - documented
    assert not undocumented, f"代码里有但 Schema 提示没写的令牌: {sorted(undocumented)}"

    # PERSONA_TOKEN_DOC 也要同步
    doc_tokens = set(_TOKEN_RE.findall(prompts.PERSONA_TOKEN_DOC))
    assert doc_tokens == set(context), (
        f"PERSONA_TOKEN_DOC 与实际令牌不一致: 缺{sorted(set(context) - doc_tokens)}"
        f" 多{sorted(doc_tokens - set(context))}"
    )


# =========================================================================== #
# 5. 插件层面的接入
# =========================================================================== #


def test_plugin_persona_uses_selected_template() -> None:
    """/zl persona 报告模板来源；选定 cyber 时生成结果真的换成 cyber 骨架。"""

    async def scenario():
        install_fake_astrbot()
        plugin, _ = await _make_plugin(
            _db("tpl_cyber.db"),
            admin_only=False,
            persona_template={"style": "cyber"},
            targets=[{"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"}],
        )
        await plugin._load_runtime_state()
        await plugin.storage.save_persona(GID_A, QQ_A, _rich_snapshot())
        ev = lambda s: FakeEvent(s, group_id=GID_A, sender_id="10001")  # noqa: E731

        assert plugin._persona_template_source() == "赛博群友 · 分节版"
        outs = await _collect(plugin.zl(ev("/zl persona")))
        joined = "\n".join(_text_of(r) for r in outs)
        assert "人设模板：赛博群友 · 分节版" in joined, joined
        assert "系统级指令" in joined and "人格根本锚定" in joined

        built = plugin._build_persona_text(plugin.state.active_target(), _rich_snapshot())
        assert "人格根本锚定" in built and "## 一、说话的铁律" not in built

        # 自定义模板优先级最高，来源也要如实报告
        plugin.config["persona_template"]["custom_template"] = "# 注释\n只看{{昵称}}"
        assert plugin._persona_template_source() == "自定义模板"
        built2 = plugin._build_persona_text(plugin.state.active_target(), _rich_snapshot())
        assert built2.strip() == "只看老王"

        await plugin.storage.close()

    asyncio.run(scenario())


def test_plugin_persona_default_is_standard() -> None:
    """不配置 style 时默认用标准版（与老版本行为一致）。"""

    async def scenario():
        install_fake_astrbot()
        plugin, _ = await _make_plugin(
            _db("tpl_default.db"),
            admin_only=False,
            targets=[{"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"}],
        )
        await plugin._load_runtime_state()
        assert plugin._persona_template_source() == "标准版 · 我要蒸馏群友"

        await plugin.storage.save_persona(GID_A, QQ_A, _rich_snapshot())
        built = plugin._build_persona_text(plugin.state.active_target(), _rich_snapshot())
        assert "## 一、说话的铁律" in built
        assert "人格根本锚定" not in built
        await plugin.storage.close()

    asyncio.run(scenario())


def test_daily_digest_sync_respects_template() -> None:
    """每日总结同步人格时，也照选定的模板来（cyber → 人格里带系统级指令）。"""

    async def scenario():
        install_fake_astrbot()
        import time as _time

        from core.storage import MessageRecord

        plugin, _ = await _make_plugin(
            _db("tpl_digest.db"),
            admin_only=False,
            provider=RecordingProviderForTemplates(json.dumps(
                {"layer1_rules": [{"text": "绝不发语音", "confidence": 0.9}]},
                ensure_ascii=False,
            )),
            daily_digest={"enabled": True, "time": "08:00", "min_messages": 0},
            persona_template={"style": "cyber", "sync_after_digest": True},
            targets=[{"group_id": GID_A, "qq_id": QQ_A, "nickname": "老王"}],
        )
        await plugin._load_runtime_state()
        mgr = _FakePersonaMgr()
        plugin.context.persona_manager = mgr  # type: ignore[attr-defined]
        await plugin.storage.insert_message(
            MessageRecord("m1", GID_A, QQ_A, "老王", "今天的话", timestamp=int(_time.time()))
        )

        assert await plugin._run_daily_digest(_time.time(), 0) is True
        assert mgr.personas, "总结后没有同步人格"
        content = next(iter(mgr.personas.values()))
        assert "人格根本锚定" in content, "同步人格没有走 cyber 模板"
        assert "## 一、说话的铁律" not in content
        await plugin.storage.close()

    asyncio.run(scenario())


class RecordingProviderForTemplates:
    """记录提示词的假 Provider。"""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    async def text_chat(self, prompt: str = "", system_prompt: str = "", **_kw):
        self.prompts.append(prompt)
        from types import SimpleNamespace

        return SimpleNamespace(completion_text=self.payload)


class _FakePersonaMgr:
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


def main() -> int:
    """直接运行时的入口，逐个执行测试函数。"""
    tests = [
        test_registry_contents,
        test_resolve_template_priority,
        test_standard_template_structure,
        test_cyber_template_structure,
        test_compact_template_is_shortest,
        test_all_templates_never_leak_tokens,
        test_empty_snapshot_still_renders_all_sections,
        test_custom_template_overrides_builtin,
        test_custom_template_unknown_token_removed,
        test_style_and_custom_are_independent,
        test_schema_options_match_registry,
        test_schema_token_doc_matches_context_keys,
        test_plugin_persona_uses_selected_template,
        test_plugin_persona_default_is_standard,
        test_daily_digest_sync_respects_template,
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
