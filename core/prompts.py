"""蒸馏提示词与 5 层 Persona 结构定义。

本模块集中管理与 LLM 交互的全部提示词常量，设计参考 pig-skill 的
「分析师 → 合并器」两阶段思路，并针对群聊语料场景补充了取证与防幻觉约束。

所有提示词共同遵循三条原则：

1. **只依据语料证据**：找不到证据的维度输出 ``null``，并写入 ``uncertainty``，
   明确禁止编造。
2. **输出严格 JSON**：便于程序化解析与增量 merge。
3. **诚实抽样**：语料超长时按信息量抽样，并在提示词中如实声明抽样情况，
   避免让使用者误以为全量投喂。

本模块零第三方依赖，可安全地在无 AstrBot 环境下被单元测试导入。
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

# --------------------------------------------------------------------------- #
# 5 层 Persona 结构的键名与中文标题（全项目统一，避免各模块各写一份）
# --------------------------------------------------------------------------- #

LAYER_KEYS: list[str] = [
    "layer1_rules",
    "layer2_identity",
    "layer3_style",
    "layer4_behavior",
    "layer5_interests",
]

LAYER_TITLES: dict[str, str] = {
    "layer1_rules": "Layer 1 · 硬规则",
    "layer2_identity": "Layer 2 · 身份",
    "layer3_style": "Layer 3 · 表达风格",
    "layer4_behavior": "Layer 4 · 聊天行为模式",
    "layer5_interests": "Layer 5 · 兴趣偏好",
}

LAYER_DESCRIPTIONS: dict[str, str] = {
    "layer1_rules": "不可违背的铁律：口头禅、说话长度上限、绝不出现的行为。",
    "layer2_identity": "身份画像：昵称、年龄段、性别倾向、职业、身份感、自我称呼。",
    "layer3_style": "表达风格：语气、句长、标点习惯、口头禅、表情包习惯、错别字习惯、语速感。",
    "layer4_behavior": "聊天行为：何时秒回、何时潜水、如何起话头、如何收尾、吵架与吐槽方式、如何称呼他人。",
    "layer5_interests": "兴趣偏好：话题清单、黑话、梗、活跃时段、雷点。",
}

# 供提示词插入的结构说明文本（拼一次，多处复用）
LAYER_DEFINITION_TEXT: str = "\n".join(
    f"- {LAYER_TITLES[key]}：{LAYER_DESCRIPTIONS[key]}" for key in LAYER_KEYS
)

# --------------------------------------------------------------------------- #
# 提示词常量
# --------------------------------------------------------------------------- #

# 分析师系统提示：设计意图 —— 把 LLM 约束成"只认证据的取证员"而非"小说家"。
# 明确要求输出 null + uncertainty，从机制上抑制幻觉；限制隐私推断，规避风险。
ANALYZER_SYSTEM = """你是一名资深人格分析师兼语料取证员。你的唯一任务：根据给定的群聊聊天记录，推断目标对象的语言风格与人格特征，并按 5 层 Persona 结构输出结构化结论。

【铁律】
1. 只依据语料证据，禁止编造。任何结论都必须能在语料中找到对应原话。
2. 找不到足够证据的维度，值输出 null，并在 uncertainty 列表里说明"哪一项、为什么证据不足"。
3. 不臆测真实姓名、住址、电话、工作单位等隐私信息；身份层只做模糊画像（年龄段、性别倾向、身份感）。
4. 关注"语言习惯"胜过"内容评价"：口头禅、句长、标点、表情、错别字、语气助词等。
5. 输出必须是一个合法 JSON 对象，禁止输出任何 JSON 之外的解释文字，禁止使用 Markdown 代码块包裹。

【输出 JSON 结构】
{
  "layer1_rules":     [{"text": "硬规则描述", "confidence": 0.0, "quotes": ["原话1", "原话2"]}],
  "layer2_identity":  [{"text": "身份特征",   "confidence": 0.0, "quotes": ["原话1"]}],
  "layer3_style":     [{"text": "风格特征",   "confidence": 0.0, "quotes": ["原话1"]}],
  "layer4_behavior":  [{"text": "行为模式",   "confidence": 0.0, "quotes": ["原话1"]}],
  "layer5_interests": [{"text": "兴趣偏好",   "confidence": 0.0, "quotes": ["原话1"]}],
  "uncertainty": ["证据不足的说明1", "证据不足的说明2"]
}

【字段约定】
- text：一句话描述，尽量精炼。
- confidence：0.0 ~ 1.0，表示你对该结论的把握程度。
- quotes：最多 3 条，必须是语料中原样出现的片段，不得改写。
- 某一层没有把握时给空数组 []，并在 uncertainty 里说明原因。
"""

# 分析师用户提示模板：只使用 {name} 占位符，模板内不得出现裸花括号，以免 format 崩溃。
ANALYZER_USER_TEMPLATE = """请分析以下群聊记录中的「目标对象」，并输出结构化 JSON。

【目标对象】
- 昵称：{nickname}
- QQ：{qq}

【语料时间跨度】
{span}

【5 层 Persona 结构定义（请严格按这些 key 输出）】
{layer_def}

【已有 Persona 摘要（用于增量对比；若显示"暂无"表示首次蒸馏）】
{existing}

【待分析语料】
{note}
{transcript}

【额外约束】
{extra}
"""

# 构建器系统提示：把分析 JSON 汇成可读的 5 层 Markdown。样板版默认走确定性
# Python merge，本提示词保留用于"需要自然语言润色档案"的进阶场景。
BUILDER_SYSTEM = """你是一名人格档案编辑。给定一份结构化的 5 层 Persona JSON 分析结果，
请把它整理成一份通顺、克制、便于阅读的 Markdown 档案。

要求：
1. 严格保留 5 层结构，每层用三级标题，条目用无序列表。
2. 忠于 JSON 内容，不得新增 JSON 中不存在的结论，也不得删减。
3. 证据引用放在条目末尾，形如 `（依据：<原话>）`，最多保留 2 条。
4. 中文输出，语气客观中性，不要点评、不要抒情。
"""

# 合并器系统提示：定义增量 merge 的"宪法"。
# 关键设计：新证据默认只做补充与加权，绝不轻易推翻旧的高置信结论；
# 一旦出现冲突，保留双方并标注 conflict，交由人工（纠正层）裁决。
MERGE_SYSTEM = """你是人格档案的增量合并器。你会拿到「已有 Persona」和「新一轮分析结果」，
请输出合并后的完整 Persona。

合并规则（务必严格遵守）：
1. 新证据只做补充与加权，不推翻已有高置信结论。已有 confidence >= 0.7 的条目一律保留。
2. 同义或高度相似的条目应合并为一条，并累加其证据强度。
3. 若新旧结论冲突（例如旧说"爱用句号"、新说"从不用句号"），
   保留双方，并给两条都标注 "conflict": true，交由人工裁决，不得擅自删除。
4. 语言风格类结论优先保留在 Layer 3；行为模式类结论优先保留在 Layer 4。
5. 输出严格 JSON，结构与输入一致，禁止输出 JSON 之外的任何文字。
"""

# 导出模板：导出的 persona_<qq>.md 骨架说明（实际渲染见 render_export_markdown）。
EXPORT_TEMPLATE = """# {nickname} 的人格档案

> 由「我要蒸馏群友」自动蒸馏生成 · 目标 QQ：{qq}

## Layer 1 · 硬规则
## Layer 2 · 身份
## Layer 3 · 表达风格
## Layer 4 · 聊天行为模式
## Layer 5 · 兴趣偏好

## 证据索引
## 人工纠正层
## Meta 元信息
"""


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #


def _fmt_ts(ts: int) -> str:
    """把 Unix 时间戳格式化为 ``月-日 时:分``，供语料行前缀使用。"""
    try:
        return time.strftime("%m-%d %H:%M", time.localtime(int(ts)))
    except (TypeError, ValueError, OSError):
        return "未知时间"


def _score_line(line: str) -> float:
    """给一行语料打信息量分，用于超长语料抽样。

    评分策略：基础分为长度；带疑问/感叹等情绪标点加分；带笑/梗词再加分。
    目标是优先保留"话多且带情绪"的消息，它们对人格刻画最有价值。
    """
    score = float(len(line))
    if any(ch in line for ch in "？?！!～~"):
        score += 20.0
    if any(meme in line for meme in ("哈哈", "笑死", "草", "绷", "xs", "🤣", "😭", "😅")):
        score += 15.0
    return score


def format_transcript(
    records: Sequence[dict[str, Any]], max_chars: int = 8000
) -> tuple[str, str]:
    """把语料记录拼装成提示词可用的文本，并返回 (文本, 抽样说明)。

    Args:
        records: 语料列表，每条含 ``timestamp`` / ``speaker_name`` /
            ``content`` / ``is_context`` 等字段。
        max_chars: 字符预算上限，超出即触发信息量抽样。

    Returns:
        ``(transcript, note)``。``note`` 会如实告知本轮是全量还是抽样，
        避免使用者误解投喂范围。
    """
    lines: list[str] = []
    for rec in records:
        content = str(rec.get("content") or "").strip()
        if not content:
            continue
        ts = rec.get("timestamp") or 0
        name = str(rec.get("speaker_name") or "群友").strip() or "群友"
        prefix = f"[{_fmt_ts(ts)}] " if ts else "[未知时间] "
        if rec.get("is_context"):
            prefix += "(语境) "
        lines.append(f"{prefix}{name}: {content}")

    total = len(lines)
    if total == 0:
        return "", "（本轮无语料）"

    joined = "\n".join(lines)
    if len(joined) <= max_chars:
        return joined, f"（本次为全量语料，共 {total} 条）"

    # 超预算：按信息量降序挑行，挑满预算后按原始时间顺序还原
    order = sorted(range(total), key=lambda i: _score_line(lines[i]), reverse=True)
    budget = max_chars
    picked: set[int] = set()
    for idx in order:
        cost = len(lines[idx]) + 1
        if budget - cost < 0 and picked:
            break
        picked.add(idx)
        budget -= cost

    if not picked:  # 极端情况：单行就超预算，至少保留最长的 1 条
        picked.add(order[0])

    ordered = [lines[i] for i in sorted(picked)]
    note = (
        f"（语料较长，已按信息量抽样保留 {len(ordered)}/{total} 条，"
        f"优先保留长消息与含情绪/梗的消息）"
    )
    return "\n".join(ordered), note


def summarize_snapshot_for_prompt(
    snapshot: Optional[dict[str, Any]], max_items: int = 4
) -> str:
    """把已有 Persona 压缩成简短摘要，供增量对比使用。"""
    if not isinstance(snapshot, dict):
        return "（暂无，首次蒸馏）"
    layers = snapshot.get("layers")
    if not isinstance(layers, dict):
        return "（暂无，首次蒸馏）"

    out: list[str] = []
    for key in LAYER_KEYS:
        items = layers.get(key) or []
        if not items:
            continue
        out.append(f"【{LAYER_TITLES[key]}】")
        for item in items[:max_items]:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
            else:
                text = str(item).strip()
            if text:
                out.append(f"- {text}")
    return "\n".join(out) if out else "（暂无，首次蒸馏）"


def build_analyzer_user(
    *,
    nickname: str,
    qq: str,
    records: Sequence[dict[str, Any]],
    existing_snapshot: Optional[dict[str, Any]] = None,
    extra: str = "",
    max_chars: int = 8000,
) -> str:
    """组装分析师的用户提示词。

    Args:
        nickname: 目标昵称。
        qq: 目标 QQ。
        records: 本轮待分析语料。
        existing_snapshot: 已有 Persona（用于增量对比）。
        extra: 使用者自定义的额外约束。
        max_chars: 语料字符预算。

    Returns:
        填充完毕的用户提示词字符串。
    """
    times = [int(r.get("timestamp") or 0) for r in records]
    valid_times = [t for t in times if t]
    if valid_times:
        span = (
            f"{_fmt_ts(min(valid_times))} ~ {_fmt_ts(max(valid_times))}"
            f"（共 {len(records)} 条）"
        )
    else:
        span = f"未知（共 {len(records)} 条）"

    transcript, note = format_transcript(records, max_chars=max_chars)
    existing = summarize_snapshot_for_prompt(existing_snapshot)
    extra_text = (extra or "").strip() or "（无）"

    return ANALYZER_USER_TEMPLATE.format(
        nickname=nickname or "未知",
        qq=qq or "未知",
        span=span,
        layer_def=LAYER_DEFINITION_TEXT,
        existing=existing,
        note=note,
        transcript=transcript or "（空）",
        extra=extra_text,
    )


def render_export_markdown(
    snapshot: Optional[dict[str, Any]],
    nickname: str,
    qq: str,
    plugin_name: str = "我要蒸馏群友",
) -> str:
    """把 Persona 快照渲染成可导出的 Markdown 文本。

    严格遵循 ``EXPORT_TEMPLATE`` 的骨架：5 层 + 证据索引 + 纠正层 + Meta。
    """
    snap = snapshot if isinstance(snapshot, dict) else {}
    layers = snap.get("layers") if isinstance(snap.get("layers"), dict) else {}
    meta = snap.get("meta") if isinstance(snap.get("meta"), dict) else {}
    corrections = snap.get("corrections") or []
    uncertainty = snap.get("uncertainty") or []

    md: list[str] = []
    md.append(f"# {nickname or '未知昵称'} 的人格档案")
    md.append("")
    md.append(f"> 由「{plugin_name}」自动蒸馏生成 · 目标 QQ：{qq}")
    md.append("")

    for key in LAYER_KEYS:
        md.append(f"## {LAYER_TITLES[key]}")
        md.append("")
        md.append(f"_{LAYER_DESCRIPTIONS[key]}_")
        md.append("")
        items = layers.get(key) or []
        if not items:
            md.append("- （暂无足够证据）")
        else:
            for item in items:
                if isinstance(item, dict):
                    text = str(item.get("text") or "").strip()
                    conf = item.get("confidence", 0.0)
                    evidence = item.get("evidence", 1)
                    quotes = item.get("quotes") or []
                    flag = " ⚠️冲突" if item.get("conflict") else ""
                    line = f"- {text}{flag} 「置信 {conf} · 证据 {evidence} 条」"
                    if quotes:
                        joined = " / ".join(str(q) for q in quotes[:3])
                        line += f"　（依据：{joined}）"
                else:
                    line = f"- {item}"
                md.append(line)
        md.append("")

    md.append("## 证据索引")
    md.append("")
    md.append("| 层级 | 结论 | 证据条数 |")
    md.append("| --- | --- | --- |")
    for key in LAYER_KEYS:
        for item in layers.get(key) or []:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                evidence = item.get("evidence", 1)
                md.append(f"| {LAYER_TITLES[key]} | {text} | {evidence} |")
    md.append("")

    md.append("## 人工纠正层（优先级高于 LLM 推断）")
    md.append("")
    if corrections:
        for c in corrections:
            md.append(f"- {c}")
    else:
        md.append("- （暂无人工纠正）")
    md.append("")

    md.append("## 不确定项（证据不足）")
    md.append("")
    if uncertainty:
        for u in uncertainty:
            md.append(f"- {u}")
    else:
        md.append("- （无）")
    md.append("")

    md.append("## Meta 元信息")
    md.append("")
    md.append(f"- 蒸馏轮次：{meta.get('distill_round', 0)}")
    md.append(f"- 语料总条数：{meta.get('total_messages', 0)}")
    md.append(f"- 去重后条数：{meta.get('deduped_messages', 0)}")
    md.append(f"- 最后蒸馏：{_fmt_ts(meta.get('last_distill_at', 0)) if meta.get('last_distill_at') else '尚未'}")
    md.append("")
    md.append("> ⚠️ 本档案由 AI 依据有限语料推断，仅供娱乐，可能不准确，")
    md.append("> 也可能涉及个人特征。请勿用于侵犯他人隐私或任何歧视性用途。")
    md.append("")

    return "\n".join(md)
