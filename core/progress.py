"""进度面板与档案渲染。

纯文本渲染，不依赖任何平台富媒体能力（除了调用方自行拼 @ 组件），
因此本模块零 AstrBot 依赖，可被单元测试直接导入、直接断言输出字符串。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

try:
    from . import prompts
except ImportError:  # pragma: no cover - 兼容顶层导入
    import prompts  # type: ignore

PLUGIN_DISPLAY = "我要蒸馏群友"
BAR_FULL = "█"
BAR_EMPTY = "░"
BAR_WIDTH = 10
SEP = "──────────────────────────"


def make_bar(current: int, total: int, width: int = BAR_WIDTH) -> str:
    """生成进度条字符串。

    Args:
        current: 当前值。
        total: 饱和基准（<=0 时视为满）。
        width: 进度条宽度（字符数）。

    Returns:
        由 ``█`` 与 ``░`` 拼成的进度条。
    """
    if width <= 0:
        return ""
    if total <= 0:
        ratio = 1.0
    else:
        ratio = max(0.0, min(1.0, current / total))
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    return BAR_FULL * filled + BAR_EMPTY * (width - filled)


def fmt_int(number: Any) -> str:
    """整数千分位格式化。"""
    try:
        return f"{int(number):,}"
    except (TypeError, ValueError):
        return "0"


def fmt_dt(ts: Any) -> str:
    """Unix 时间戳 → 本地 ``月-日 时:分``。"""
    try:
        value = int(ts)
    except (TypeError, ValueError):
        return "—"
    if value <= 0:
        return "—"
    try:
        return time.strftime("%m-%d %H:%M", time.localtime(value))
    except (OSError, ValueError):
        return "—"


def fmt_span(min_ts: Any, max_ts: Any) -> str:
    """时间跨度 → ``09-01 ~ 09-16（15 天）``。"""
    try:
        lo = int(min_ts)
        hi = int(max_ts)
    except (TypeError, ValueError):
        return "尚无数据"
    if lo <= 0 and hi <= 0:
        return "尚无数据"
    days = max(0, int((hi - lo) // 86400))
    return f"{fmt_dt(lo)} ~ {fmt_dt(hi)}（{days} 天）"


@dataclass
class PanelData:
    """进度面板的渲染数据。"""

    plugin_name: str = PLUGIN_DISPLAY
    has_target: bool = False
    nickname: str = ""
    qq: str = ""
    group_name: str = ""
    group_id: str = ""
    distilling: bool = False
    listen: bool = True
    total: int = 0
    saturation: int = 1500
    round_no: int = 0
    last_distill_ts: int = 0
    queue: int = 0
    min_ts: int = 0
    max_ts: int = 0
    completeness: int = 0
    formed_layers: int = 0
    total_layers: int = len(prompts.LAYER_KEYS)


def render_no_target() -> str:
    """未设定目标时的友好引导面板。"""
    lines = [
        f"🧪 群友蒸馏 · {PLUGIN_DISPLAY}",
        SEP,
        "❗ 还没有设定蒸馏目标",
        "请在 WebUI 插件配置里填写「目标群号 / 目标群友 QQ」，",
        "或直接在群里用指令：/zl set <群号> <QQ号>",
        SEP,
        "💡 /zl help 查看全部指令",
    ]
    return "\n".join(lines)


def render_panel(data: PanelData) -> str:
    """渲染 ``/zl`` 默认进度面板。"""
    if not data.has_target:
        return render_no_target()

    saturation = max(1, int(data.saturation or 1))
    percent = 0 if data.total <= 0 else min(100, int(round(data.total / saturation * 100)))
    bar = make_bar(data.total, saturation)

    # 状态
    if not data.listen:
        status_emoji, status_text = "⚪", "已暂停"
    elif data.distilling:
        status_emoji, status_text = "🟠", "蒸馏中"
    else:
        status_emoji, status_text = "🟢", "待命"
    listen_text = "开启" if data.listen else "关闭"

    group_display = data.group_name or "目标群"
    nickname = data.nickname or "未知昵称"
    last = fmt_dt(data.last_distill_ts) if data.last_distill_ts else "尚未"

    lines = [
        f"🧪 群友蒸馏进度 · {data.plugin_name}",
        SEP,
        f"👤 目标    {nickname} ({data.qq})",
        f"👥 群聊    {group_display} ({data.group_id})",
        f"🎚 状态    {status_emoji} {status_text}  |  采集：{listen_text}",
        f"📦 语料    {fmt_int(data.total)} 条  [{bar}] {percent}%",
        f"🔁 蒸馏    {data.round_no} 轮 · 上次 {last} · 排队 {fmt_int(data.queue)} 条待蒸馏",
        f"📅 跨度    {fmt_span(data.min_ts, data.max_ts)}",
        f"🧩 档案    完整度 {data.completeness}% · 已成型 {data.formed_layers}/{data.total_layers} 层",
        SEP,
        "💡 /zl profile 看档案 · /zl help 看全部指令",
    ]
    return "\n".join(lines)


def render_profile(
    snapshot: Optional[dict[str, Any]], nickname: str, qq: str
) -> str:
    """渲染 ``/zl profile`` 的 5 层档案摘要。"""
    if not isinstance(snapshot, dict):
        return (
            f"🧬 {nickname or '未知昵称'} 还没有档案。\n"
            "先攒点语料，再用 /zl distill 手动蒸馏一轮吧。"
        )

    layers = snapshot.get("layers") if isinstance(snapshot.get("layers"), dict) else {}
    meta = snapshot.get("meta") if isinstance(snapshot.get("meta"), dict) else {}
    corrections = snapshot.get("corrections") or []

    from .distiller import compute_completeness  # 局部导入避免循环

    lines = [
        f"🧬 {nickname or '未知昵称'} 的人格档案（{qq}）",
        f"第 {meta.get('distill_round', 0)} 轮 · 完整度 {compute_completeness(snapshot)}% "
        f"· 语料 {fmt_int(meta.get('total_messages', 0))} 条",
        SEP,
    ]

    for key in prompts.LAYER_KEYS:
        lines.append(f"【{prompts.LAYER_TITLES[key]}】")
        items = layers.get(key) or []
        if not items:
            lines.append("- （暂无足够证据）")
        else:
            for item in items[:5]:
                if isinstance(item, dict):
                    text = str(item.get("text") or "").strip()
                    flag = " ⚠️冲突" if item.get("conflict") else ""
                    lines.append(f"- {text}{flag}")
                else:
                    lines.append(f"- {item}")
        lines.append("")

    lines.append("【⚠️ 人工纠正层（优先级最高）】")
    if corrections:
        for c in corrections:
            lines.append(f"- {c}")
    else:
        lines.append("- （暂无）")

    lines.append(SEP)
    lines.append("💡 /zl export 导出完整 Markdown 档案")
    return "\n".join(lines)


def render_help(prefix: str = "/zl") -> str:
    """渲染帮助文本。"""
    p = prefix
    lines = [
        f"🧪 {PLUGIN_DISPLAY} · 指令帮助",
        SEP,
        f"{p}                查看进度面板",
        f"{p} on / off       开启 / 关闭采集",
        f"{p} set <群号> <QQ> 设定蒸馏目标（也可只给 QQ，用当前群）",
        f"{p} now            查看当前目标",
        f"{p} distill        手动触发一轮蒸馏",
        f"{p} profile        查看 5 层人格档案摘要",
        f"{p} export         导出 persona_<QQ>.md 到数据目录",
        f"{p} correct <内容>  追加人工纠正（别名：{p} 纠正 <内容>）",
        f"{p} reset confirm  清空该目标语料（需二次确认）",
        f"{p} help           显示本帮助",
        SEP,
        "⚠️ 仅供娱乐，请勿用于侵犯他人隐私。",
    ]
    return "\n".join(lines)
