"""人格快照压缩相关测试（v0.8.0，不依赖 AstrBot 本体，也不依赖 pytest）。

覆盖「多轮蒸馏无限膨胀」修复的两层防御 + 渲染端截断：

- ``cap_snapshot``：每层超 16 条时按 evidence 裁剪、保留 evidence 最高的、
  维持相对顺序、meta/corrections 原样、幂等；
- ``merge_snapshot``：连续多轮语义近似但 text 不同的条目最终每层有界；
- ``snapshot_item_count``：条目总数统计正确；
- ``rehydrate_compressed``：evidence 保留、corrections 用传入值、
  ``meta.compress_count`` 自增、结果有界；
- ``build_compressor_user``：输出含 5 层 JSON 且不含 meta/corrections 字段；
- ``_persona_bullets``：>16 条时渲染端截断并附「已省略」提示。

运行方式（任选其一）：

    python tests/test_compression.py
    python -m pytest tests/test_compression.py   # 若环境有 pytest
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import prompts  # noqa: E402
from core.distiller import (  # noqa: E402
    cap_snapshot,
    empty_snapshot,
    merge_snapshot,
    parse_analysis,
    rehydrate_compressed,
    snapshot_item_count,
)


def _layer_items(n: int) -> list[dict[str, Any]]:
    """构造 n 条 evidence 递增（1..n）、confidence 相同的结论条目。"""
    return [
        {"text": f"特征{i}", "confidence": 0.5, "evidence": i + 1}
        for i in range(n)
    ]


def test_cap_snapshot_trims_layers_and_preserves_meta() -> None:
    """某层超过 16 条时，保留 evidence 最高的 16 条，且 meta/corrections 不变。"""
    snap: dict[str, Any] = {
        "layers": {key: [] for key in prompts.LAYER_KEYS},
        "speech_samples": [],
        "uncertainty": [],
        "corrections": ["人工纠正一条"],
        "meta": {"distill_round": 5, "total_messages": 999},
    }
    snap["layers"]["layer3_style"] = _layer_items(20)

    capped = cap_snapshot(snap)

    kept = capped["layers"]["layer3_style"]
    assert len(kept) == prompts.MAX_ITEMS_PER_LAYER, len(kept)
    # evidence 最高的 16 条是 evidence 5..20，即原顺序里的 特征4..特征19
    assert [it["text"] for it in kept] == [f"特征{i}" for i in range(4, 20)]
    # 入参不被原地修改
    assert len(snap["layers"]["layer3_style"]) == 20
    # meta / corrections 原样保留
    assert capped["corrections"] == ["人工纠正一条"]
    assert capped["meta"] == {"distill_round": 5, "total_messages": 999}

    # 幂等：再裁一次结果不变
    assert cap_snapshot(capped) == capped


def test_cap_snapshot_bounds_samples_and_uncertainty() -> None:
    """samples 封顶 40，uncertainty 去重后封顶 12。"""
    snap: dict[str, Any] = {
        "layers": {key: [] for key in prompts.LAYER_KEYS},
        "speech_samples": [
            {"situation": f"情境{i}", "reply": f"回复{i}", "evidence": i + 1}
            for i in range(prompts.MAX_SPEECH_SAMPLES + 10)
        ],
        "uncertainty": [f"不确定项{i}" for i in range(30)] + ["不确定项0"],
        "corrections": [],
        "meta": {},
    }
    capped = cap_snapshot(snap)
    assert len(capped["speech_samples"]) == prompts.MAX_SPEECH_SAMPLES
    assert len(capped["uncertainty"]) == prompts.MAX_UNCERTAINTY
    # 去重：重复的「不确定项0」不应出现两次
    assert capped["uncertainty"].count("不确定项0") == 1


def test_merge_snapshot_bounded_after_many_rounds() -> None:
    """连续多轮 merge 语义近似但 text 不同的条目，最终每层有界。"""
    snap: dict[str, Any] | None = None
    for round_no in range(30):
        analysis = {
            "layer3_style": [{"text": f"他说话很有特点变体{round_no}", "confidence": 0.6}],
            "speech_samples": [{"situation": f"情境{round_no}", "reply": f"回答{round_no}"}],
            "uncertainty": [f"不确定项变体{round_no}"],
        }
        snap = merge_snapshot(snap, analysis, ["纠正"])

    assert snap is not None
    assert len(snap["layers"]["layer3_style"]) <= prompts.MAX_ITEMS_PER_LAYER
    assert len(snap["speech_samples"]) <= prompts.MAX_SPEECH_SAMPLES
    assert len(snap["uncertainty"]) <= prompts.MAX_UNCERTAINTY
    assert snap["meta"]["distill_round"] == 30


def test_snapshot_item_count() -> None:
    """条目总数 = 5 层 + speech_samples + uncertainty。"""
    snap: dict[str, Any] = {
        "layers": {
            "layer1_rules": [{"text": "a"}],
            "layer2_identity": [{"text": "b"}, {"text": "c"}],
            "layer3_style": [],
            "layer4_behavior": [{"text": "d"}],
            "layer5_interests": [{"text": "e"}, {"text": "f"}, {"text": "g"}],
        },
        "speech_samples": [
            {"situation": "x", "reply": "y"},
            {"situation": "z", "reply": "w"},
        ],
        "uncertainty": ["u1"],
    }
    assert snapshot_item_count(snap) == 1 + 2 + 0 + 1 + 3 + 2 + 1
    assert snapshot_item_count(None) == 0
    assert snapshot_item_count({}) == 0
    assert snapshot_item_count("bad") == 0


def test_rehydrate_compressed() -> None:
    """压缩器 JSON 规整：evidence 保留、corrections 用传入值、compress_count 自增、结果有界。"""
    existing: dict[str, Any] = {
        "layers": {
            key: _layer_items(20) if key == "layer1_rules" else []
            for key in prompts.LAYER_KEYS
        },
        "speech_samples": [],
        "uncertainty": ["旧不确定"],
        "corrections": ["旧纠正"],
        "meta": {"distill_round": 7, "total_messages": 999, "compress_count": 2},
    }
    compressed: dict[str, Any] = {
        "layer1_rules": [
            {
                "text": "合并后的结论",
                "confidence": 0.8,
                "evidence": 12,
                "quotes": ["原话1", "原话2"],
                "conflict": True,
            }
        ],
        "layer2_identity": [],
        "layer3_style": [],
        "layer4_behavior": [],
        "layer5_interests": [],
        "speech_samples": [{"situation": "s", "reply": "r", "evidence": 3}],
        "uncertainty": ["新不确定"],
    }

    out = rehydrate_compressed(existing, compressed, ["权威纠正"])

    item = out["layers"]["layer1_rules"][0]
    assert item["text"] == "合并后的结论"
    assert item["evidence"] == 12, "压缩器给的 evidence 未被保留"
    assert item["conflict"] is True
    assert out["speech_samples"][0]["evidence"] == 3

    assert out["corrections"] == ["权威纠正"], "corrections 应使用传入值而非 LLM 输出"
    assert out["meta"]["compress_count"] == 3, "compress_count 应自增"
    assert out["meta"]["distill_round"] == 7, "meta 应复制 existing"
    assert out["meta"]["total_messages"] == 999

    # 结果有界（最终兜底 cap_snapshot）
    for key in prompts.LAYER_KEYS:
        assert len(out["layers"][key]) <= prompts.MAX_ITEMS_PER_LAYER
    assert len(out["speech_samples"]) <= prompts.MAX_SPEECH_SAMPLES
    assert len(out["uncertainty"]) <= prompts.MAX_UNCERTAINTY


def test_build_compressor_user() -> None:
    """压缩器用户提示词：含 5 层 JSON、不含 meta/corrections 字段。"""
    snap: dict[str, Any] = {
        "layers": {
            "layer1_rules": [
                {
                    "text": "绝不发语音",
                    "confidence": 0.9,
                    "evidence": 2,
                    "quotes": ["我不发语音的"],
                    "trigger": "被要求发语音时",
                    "avoid": "熟人也不会",
                    "conflict": False,
                }
            ],
            "layer2_identity": [],
            "layer3_style": [],
            "layer4_behavior": [],
            "layer5_interests": [],
        },
        "speech_samples": [
            {"situation": "别人问他在干嘛", "reply": "在呢在呢", "evidence": 1, "quotes": ["在呢在呢"]}
        ],
        "uncertainty": ["工作状态证据不足"],
        "corrections": ["人工纠正"],
        "meta": {"distill_round": 3},
    }

    text = prompts.build_compressor_user(snap, "老王", "123456789")
    # 不含 meta / corrections 字段（既不在 JSON 里，也不该出现在说明文字里）
    assert '"meta"' not in text
    assert '"corrections"' not in text

    data = parse_analysis(text)
    assert isinstance(data, dict), "提示词里应能解析出 JSON"
    assert set(data.keys()) == set(prompts.LAYER_KEYS) | {
        prompts.SAMPLE_KEY,
        "uncertainty",
    }
    assert data["layer1_rules"][0]["text"] == "绝不发语音"
    assert data["layer1_rules"][0]["evidence"] == 2
    assert data["speech_samples"][0]["reply"] == "在呢在呢"
    # 目标信息要带上
    assert "老王" in text and "123456789" in text


def test_persona_bullets_truncates() -> None:
    """>16 条的层渲染时被截断，并附「已省略」提示。"""
    items = _layer_items(20)
    lines = prompts._persona_bullets(items, include_evidence=False)

    # 16 条结论 + 1 行省略提示
    assert len(lines) == prompts.MAX_ITEMS_PER_LAYER + 1, lines
    assert lines[0] == "- 特征4", "应保留 evidence 最高的 16 条且维持相对顺序"
    assert any("已省略 4 条" in ln for ln in lines), lines
    assert lines[-1] == "- （已省略 4 条低证据条目，见 /zl export）"


def main() -> int:
    """直接运行时的入口，逐个执行测试函数。"""
    tests = [
        test_cap_snapshot_trims_layers_and_preserves_meta,
        test_cap_snapshot_bounds_samples_and_uncertainty,
        test_merge_snapshot_bounded_after_many_rounds,
        test_snapshot_item_count,
        test_rehydrate_compressed,
        test_build_compressor_user,
        test_persona_bullets_truncates,
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
    raise SystemExit(main())
