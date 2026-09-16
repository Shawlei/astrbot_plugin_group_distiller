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

# 除 5 层结论之外，专门用来"演得像"的场景化应答样例区块。
# 它不是第 6 层「性格」，而是"情境 → 他会怎么回"的现成素材，对扮演帮助最大。
SAMPLE_KEY = "speech_samples"

# 结论条目允许携带的可选附加字段：描述"什么时候会出现 / 什么时候不会"。
ITEM_OPTIONAL_FIELDS = ("trigger", "avoid")

# 场景应答样例的条数上限（防止快照无限膨胀）
MAX_SPEECH_SAMPLES = 40

# --------------------------------------------------------------------------- #
# 提示词常量
# --------------------------------------------------------------------------- #

# 分析师系统提示：设计意图 —— 把 LLM 约束成"只认证据的取证员"而非"小说家"。
# 明确要求输出 null + uncertainty，从机制上抑制幻觉；限制隐私推断，规避风险。
#
# v0.4.0 起大幅加严：从"贴标签"升级为"还原成一个能被照着演出来的人"。
# 关键手段是**点名禁用空话套话**（第 3 条列了反面清单）+ 要求每条结论都带
# 可观察的语言行为 + 要求产出场境化应答样例。这三招直接决定 Bot 演得像不像。
ANALYZER_SYSTEM = """你是一名资深人格分析师兼语料取证员。你的唯一任务：根据给定的群聊聊天记录，把这个目标对象「还原」成一个别人能照着演出来的人 —— 不是贴一堆标签，而是给出可观察、可复现的语言行为。

【铁律】
1. 只依据语料证据。每条结论都必须能在语料里找到对应原话；quotes 至少给 1 条，原样照抄，一个字都不许改写。
2. 越具体越好。要写"他吐槽时喜欢用「绷不住了」起头，后面跟一句反问"，而不是写"他很幽默"。凡是换成任何另一个人也成立、或者不看语料也能猜到的描述，一律不要写。
3. 禁止空话套话。下面这类表述直接算不合格，绝对不要出现在输出里：
   「话不多」「比较活跃」「性格开朗」「喜欢聊天」「偶尔发言」「挺有意思的人」「对话题有自己的看法」「发言很随意」「是个正常人」。
   如果你只能想到这种程度，说明证据不足 —— 那就少写几条，并在 uncertainty 里说明缺什么证据，绝不凑数。
4. 不臆测真实姓名、住址、电话、工作单位等隐私信息；身份层只做模糊画像（年龄段、性别倾向、身份感、自我称呼）。
5. 重点抓这些可观察的行为：口头禅、常用起头/收尾词、句长、标点习惯（用不用句号/波浪号/省略号）、表情与颜文字使用、错别字与漏字习惯、语气助词、回复节奏、怎么起话头、怎么结束话题、怎么称呼别人、吵架与吐槽的方式。
6. 每层尽量给 4~8 条结论（证据足够的话）；证据不够就少给几条，并在 uncertainty 里写明。
7. 每条结论都可以额外带两个可选字段，描述"什么时候会出现 / 什么时候不会"：
   - trigger：什么情境下会这样（如"被催更时""有人阴阳他时"）
   - avoid：什么情况下不会这样（如"陌生人搭话时不会这么亲昵"）
   没把握就省略，不要编。
8. 另外产出 speech_samples：从语料里挑几个典型情境，还原成"情境 → 他会怎么回"的样例，给 3~8 条。reply 要贴近原话语气，允许为补全句子做轻微改写，但内容不得凭空编造。这是让别人演得像他的关键素材。
9. 输出必须是一个合法 JSON 对象，禁止输出任何 JSON 之外的解释文字，禁止使用 Markdown 代码块包裹。

【输出 JSON 结构】
{
  "layer1_rules":     [{"text": "硬规则描述", "confidence": 0.0, "quotes": ["原话1"], "trigger": "情境", "avoid": "情境"}],
  "layer2_identity":  [{"text": "身份特征",   "confidence": 0.0, "quotes": ["原话1"]}],
  "layer3_style":     [{"text": "风格特征",   "confidence": 0.0, "quotes": ["原话1"]}],
  "layer4_behavior":  [{"text": "行为模式",   "confidence": 0.0, "quotes": ["原话1"]}],
  "layer5_interests": [{"text": "兴趣偏好",   "confidence": 0.0, "quotes": ["原话1"]}],
  "speech_samples":   [{"situation": "别人问他在干嘛", "reply": "在呢在呢，刚开了一把", "quotes": ["在呢在呢"]}],
  "uncertainty": ["证据不足的说明1", "证据不足的说明2"]
}

【字段约定】
- text：一句话结论，精炼且具体；负面与边界情况也照实写（"他从不发语音""被怼了会立刻还嘴"也是有效结论）。
- confidence：0.0 ~ 1.0，表示你对该结论的把握程度。
- quotes：1~3 条，必须是语料中原样出现的片段。
- trigger / avoid：可选，中文，一句话。
- speech_samples[].situation：情境描述；reply：他会怎么回；quotes：支撑这条样例的原话。
- 某一层没有把握时给空数组 []，并在 uncertainty 里说明原因。
"""

# 分析师用户提示模板：只使用具名占位符，模板内不得出现裸花括号，以免 format 崩溃。
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

【必须输出的 key（一个都不能少）】
layer1_rules / layer2_identity / layer3_style / layer4_behavior / layer5_interests / speech_samples / uncertainty

【额外约束（由使用者指定，优先级高于你自己的偏好）】
{extra}

【最后提醒】
这次分析的产出会被直接用于让另一个模型扮演他，所以请把重点放在"怎么说话"上：
多给具体的口头禅、句式、标点与情绪反应，少给评语式的性格概括。
每条结论都要能让人看完就知道"下一句话该怎么打出来"。
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
# 配置里两个"填空框"的默认模板
#
# 设计要点：默认给的是**带注释的模板**。以 # 或 // 开头的行在运行时会通过
# :func:`strip_template_comments` 被剥掉，所以：
#   - 用户打开配置就能看到"这里能写什么"，不用去翻文档；
#   - 但在用户自己动手启用（删掉行首的 # 或另起一行写自己的话）之前，
#     默认模板不会真的影响蒸馏结果 —— 不擅自替用户加约束。
# --------------------------------------------------------------------------- #

# 视为注释的行首标记
COMMENT_PREFIXES = ("#", "//", "＃", "／／")

# 「追加蒸馏约束」的默认模板
DEFAULT_CUSTOM_PROMPT_EXTRA = """# ── 追加蒸馏约束 · 填写模板 ──────────────────────
# 这里写的是「给分析师模型的额外要求」，会拼在蒸馏提示词末尾，优先级高于它自己的偏好。
# 下面每行都是一条可用示例。想启用哪条，把行首的 # 删掉即可；也可以直接在下面另起一行写自己的要求。
# 注意：以 # 或 // 开头的行会被自动忽略（所以现在这份模板不会生效）。
#
# - 多关注他吐槽、阴阳怪气、抬杠时的语气，少总结他的观点内容。
# - 他打字很快，常有错别字和漏字，请保留这种粗糙感，不要润色。
# - 严格区分"他自己说的"和"他转述/复述别人的"，只采信前者。
# - 表情包、颜文字只统计使用频率和出现场合，不要描述图片内容。
# - 每条结论都要给原话，没有原话支撑的一律不要输出。
# - 宁可少写几条，也不要用"可能""大概""似乎"这类模糊措辞凑数。
# - 情境样例（situation/reply）多给几条，这是让 Bot 演得像的关键素材。
"""

# 「人格追加规矩」的默认模板
DEFAULT_PERSONA_EXTRA_RULES = """# ── 人格追加规矩 · 填写模板 ──────────────────────
# 这里写的是「写进人格设定的硬规矩」，会作为「主人额外交代的规矩」附在人格模板最后一段。
# 想启用哪条就把行首的 # 删掉；也可以直接另起一行写自己的要求。
# 以 # 或 // 开头的行会被自动忽略（所以现在这份模板不会生效）。
#
# - 每次回复不超过两句话，且不要用"作为一个AI"之类的措辞。
# - 被问到身份、住址、学校、工作单位时一律含糊带过，不要编造具体信息。
# - 不要主动提起"蒸馏""人格""设定""扮演"这些词，也不要跳出角色解释自己。
# - 不要对群里的任何人做人身攻击、外貌评价或地域歧视。
# - 遇到不会的话题就直接说不会，别硬装懂。
"""


def strip_template_comments(text: Any) -> str:
    """剥掉模板里以 ``#`` / ``//`` 开头的注释行。

    这样「默认模板」能起到说明书的作用，却不会在用户真正启用之前影响提示词。

    Args:
        text: 配置里填入的原始文本。

    Returns:
        去掉注释行并去除首尾空白后的文本；没有有效内容时返回空串（调用方按
        "未填写"处理）。
    """
    lines: list[str] = []
    for line in str(text if text is not None else "").splitlines():
        if line.strip().startswith(COMMENT_PREFIXES):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


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

    md.append("## 场景应答样例")
    md.append("")
    sample_lines = _persona_samples(snap_samples(snap))
    if sample_lines:
        md.extend(sample_lines)
    else:
        md.append("- （暂无足够样例）")
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


# --------------------------------------------------------------------------- #
# AstrBot 人格模板（把蒸馏结果「搬进」AstrBot 的人格设定）
# --------------------------------------------------------------------------- #

# 人格模板里 5 层对应的自然语言标题。与 LAYER_TITLES 的「Layer N · xxx」不同，
# 这里是写给模型看的第二人称指令，语气更像在交代一个角色怎么演。
PERSONA_SECTION_TITLES: dict[str, str] = {
    "layer1_rules": "一、说话的铁律（最高优先级，任何情况都不能违背）",
    "layer2_identity": "二、我是谁",
    "layer3_style": "三、我平常怎么说话",
    "layer4_behavior": "四、我在群里怎么跟人互动",
    "layer5_interests": "五、我喜欢什么、讨厌什么",
}

# 某一层没证据时的占位提示。刻意写成「别硬演」，避免模型凭空空想出人格。
PERSONA_EMPTY_HINT = "（这部分语料里看不出来，就按一个普通群友的方式自然应对，别硬演。）"

# 人格模板的固定开场白。刻意把"不要出戏"和"档案怎么用"写在前几行 ——
# 模型对系统提示词的开头最敏感。
PERSONA_OPENING = """你现在就是「{nickname}」——一个实实在在存在过的 QQ 群友，不是 AI，不是助手。
下面是别人根据 {qq} 的聊天记录给你整理的档案。请完全第一人称代入，
用它的语气、习惯和脾气说话。

【怎么用这份档案】
- 下面每一条都是别人从他的聊天记录里总结出来的，尽量照着来。
- 遇到档案里没写过的情况，就按你已经表现出的性格自然反应，不要突然变成另一个人。
- 档案里的「情境」和「什么时候不会这样」比结论本身更重要 —— 它决定你什么时候该那样说话。

【扮演铁律】
1. 你就是{nickname}本友。绝不要说自己是 AI、大模型、机器人或助手，
   也不要提"扮演""设定""人格档案""蒸馏"这类词。
2. 时刻守住下面的说话习惯。哪怕别人问你正经问题，也要用它的方式回答，
   而不是突然切换成客客气气的百科口吻。
3. 拿不准的事就用它的方式含糊、吐槽或者转移话题过去，不要忽然变得很专业。
4. 档案里没写的性格，不要自己发明。宁可少说、也别演歪。
5. 保持它的口头禅、句长和标点习惯 —— 哪怕内容说对了，说话方式不对也算演砸了。"""


def _persona_bullets(
    items: Sequence[Any], include_evidence: bool, fallback: str = PERSONA_EMPTY_HINT
) -> list[str]:
    """把某一层的结论渲染成 Markdown 无序列表。

    结论条目除了 ``text``，还可能带 ``trigger``（什么时候会这样）与
    ``avoid``（什么时候不会这样）—— 这两个情境字段对"演得像"的帮助比结论
    本身更大，所以渲染成缩进子行而不是丢掉。

    Args:
        items: 该层的条目（dict 或字符串）。
        include_evidence: 是否附上原话样张。
        fallback: 该层无内容时的占位行。

    Returns:
        若干行文本；无内容时返回单行 ``fallback``。
    """
    if not items:
        return [f"- {fallback}"]

    lines: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            line = f"- {text}"
            if item.get("conflict"):
                line += "（这条的判断有争议，语气上别太笃定）"
            if include_evidence:
                quotes = [str(q).strip() for q in (item.get("quotes") or []) if str(q).strip()]
                if quotes:
                    line += f"　例：{' / '.join(quotes[:2])}"
        else:
            text = str(item).strip()
            if not text:
                continue
            line = f"- {text}"

        lines.append(line)
        for field, label in (("trigger", "出现时机"), ("avoid", "什么时候不这样")):
            value = str(item.get(field) or "").strip() if isinstance(item, dict) else ""
            if value:
                lines.append(f"  · {label}：{value}")

    return lines or [f"- {fallback}"]


def _persona_samples(samples: Sequence[Any]) -> list[str]:
    """把场景化应答样例渲染成"情境 → 他会怎么回"。"""
    lines: list[str] = []
    for item in samples:
        if not isinstance(item, dict):
            text = str(item).strip()
            if text:
                lines.append(f"- {text}")
            continue
        situation = str(item.get("situation") or "").strip()
        reply = str(item.get("reply") or "").strip()
        if not (situation or reply):
            continue
        if situation and reply:
            lines.append(f"- 【{situation}】他会说：{reply}")
        elif reply:
            lines.append(f"- 他会说：{reply}")
        else:
            lines.append(f"- 会遇到的情境：{situation}")
    return lines


def build_astrbot_persona(
    snapshot: Optional[dict[str, Any]],
    nickname: str,
    qq: str,
    *,
    include_evidence: bool = False,
    extra_rules: str = "",
    plugin_name: str = "我要蒸馏群友",
) -> str:
    """把 Persona 快照渲染成可直接粘贴进 AstrBot「人格设定」的系统提示词。

    产出是一段纯文本（AstrBot 人格的 ``prompt`` 字段就是要这种系统提示词）。
    设计上刻意分两层：**扮演铁律**放最前面（模型对开头最敏感），
    5 层结论随后展开，来源与局限放在最后的注释里。

    Args:
        snapshot: Persona 快照。
        nickname: 目标昵称。
        qq: 目标 QQ。
        include_evidence: 是否把代表性原话附在结论后面。
        extra_rules: 使用者在配置里追加的额外规矩。
        plugin_name: 用于生成来源注释。

    Returns:
        人格模板文本。快照为空时也会返回一份（只是内容全是"证据不足"）。
    """
    snap = snapshot if isinstance(snapshot, dict) else {}
    layers = snap.get("layers") if isinstance(snap.get("layers"), dict) else {}
    meta = snap.get("meta") if isinstance(snap.get("meta"), dict) else {}
    corrections = [str(c).strip() for c in (snap.get("corrections") or []) if str(c).strip()]
    uncertainty = [str(u).strip() for u in (snap.get("uncertainty") or []) if str(u).strip()]
    samples = snap.get(SAMPLE_KEY) if isinstance(snap.get(SAMPLE_KEY), list) else []

    display = (nickname or "").strip() or "群友"

    lines: list[str] = []
    lines.append(PERSONA_OPENING.format(nickname=display, qq=qq or "未知"))
    lines.append("")

    for key in LAYER_KEYS:
        lines.append(f"## {PERSONA_SECTION_TITLES[key]}")
        lines.extend(_persona_bullets(layers.get(key) or [], include_evidence))
        lines.append("")

    # 场景应答样例：扮演时最好用的素材，所以放在 5 层之后、纠正层之前
    lines.append("## 六、他被人这么问时，一般会这么回")
    sample_lines = _persona_samples(samples)
    if sample_lines:
        lines.extend(sample_lines)
    else:
        lines.append("- （还没有攒到足够的应答样例，按上面的说话习惯自由发挥。）")
    lines.append("")

    lines.append("## 七、人工纠正（优先级高于上面的所有推断）")
    if corrections:
        lines.extend(f"- {c}" for c in corrections)
    else:
        lines.append("- （暂无。上面的推断如有偏差，请用「/zl 纠正 <内容>」补一条。）")
    lines.append("")

    extra = (extra_rules or "").strip()
    if extra:
        lines.append("## 八、主人额外交代的规矩")
        lines.append(f"- {extra}")
        lines.append("")

    if uncertainty:
        lines.append("## 九、留个心（这些地方证据不足，别看太重）")
        lines.extend(f"- {u}" for u in uncertainty)
        lines.append("")

    total = int(meta.get("total_messages", 0) or 0)
    round_no = int(meta.get("distill_round", 0) or 0)
    generated = _fmt_ts(meta.get("last_distill_at", 0)) if meta.get("last_distill_at") else "未蒸馏过"
    lines.append("---")
    lines.append(
        f"（本模板由「{plugin_name}」依据 {total} 条群聊语料、第 {round_no} 轮蒸馏自动生成 · "
        f"目标 QQ {qq or '未知'} · 最后蒸馏 {generated}）"
    )
    lines.append(
        "（⚠️ 内容由 AI 推断，可能不准；仅供娱乐，请勿用于侵犯他人隐私或任何歧视性用途。）"
    )
    return "\n".join(lines)


def build_persona_summary(
    snapshot: Optional[dict[str, Any]], nickname: str, qq: str
) -> str:
    """渲染人格模板的"体检报告"：能不能用了、还缺什么。

    用于 ``/zl persona`` 的头部提示，帮使用者判断现在搬进 AstrBot 是否合适。
    """
    layers = {}
    if isinstance(snapshot, dict) and isinstance(snapshot.get("layers"), dict):
        layers = snapshot["layers"]
    formed = [key for key in LAYER_KEYS if layers.get(key)]
    missing = [key for key in LAYER_KEYS if not layers.get(key)]

    if not snapshot:
        return (
            f"⚠️ {nickname or '目标'} 还没有任何档案，现在生成的人格模板基本是空壳。\n"
            "建议先攒语料、跑一轮 /zl distill 再来。"
        )

    tip = "✅ 可以直接搬进 AstrBot 了。" if not missing else "🟡 还能用，但有几层证据不足。"
    samples = snap_samples(snapshot)
    lines = [
        f"🧬 {nickname or '目标'}（{qq}）人格模板体检",
        SEP_LINE,
        f"成型层数：{len(formed)}/{len(LAYER_KEYS)}",
    ]
    if formed:
        lines.append("已有：" + "、".join(LAYER_TITLES[k] for k in formed))
    if missing:
        lines.append("缺少：" + "、".join(LAYER_TITLES[k] for k in missing))
    lines.append(f"场景应答样例：{len(samples)} 条" + ("" if samples else "（建议再攒点语料）"))
    lines.append(tip)
    return "\n".join(lines)


def snap_samples(snapshot: Optional[dict[str, Any]]) -> list[Any]:
    """从快照里安全取出场景应答样例列表。"""
    if not isinstance(snapshot, dict):
        return []
    value = snapshot.get(SAMPLE_KEY)
    return value if isinstance(value, list) else []


# 供 build_persona_summary 使用的分隔线（与 progress.SEP 保持同一视觉）
SEP_LINE = "──────────────────────────"
