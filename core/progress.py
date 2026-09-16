"""进度面板与档案渲染。

纯文本渲染，不依赖任何平台富媒体能力（除了调用方自行拼 @ 组件），
因此本模块零 AstrBot 依赖，可被单元测试直接导入、直接断言输出字符串。

面板分三块：**当前目标详情** → **目标清单**（多目标时才显示）→ **操作提示**。
只有 1 个目标时输出与 v0.1.0 完全一致，避免单目标用户看到多余的噪音。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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

# 群聊单条消息的安全长度：超过就分条发，避免被平台截断
CHUNK_LIMIT = 1500


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


def chunk_text(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    """把长文本按行切成若干块，每块长度尽量不超过 ``limit``。

    优先在换行处切；单行本身就超长时按硬长度截断，保证一定能切开。

    Args:
        text: 原始文本。
        limit: 单块长度上限（按字符计）。

    Returns:
        分块列表；输入为空时返回空列表。
    """
    raw = str(text or "")
    if not raw.strip():
        return []
    if limit <= 0:
        return [raw]

    chunks: list[str] = []
    buf = ""
    for line in raw.splitlines():
        while len(line) > limit:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{buf}\n{line}" if buf else line
        if len(candidate) > limit:
            chunks.append(buf)
            buf = line
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


@dataclass
class TargetRow:
    """目标清单里的一行。"""

    key: str = ""
    nickname: str = ""
    qq: str = ""
    group_id: str = ""
    total: int = 0
    undistilled: int = 0
    round_no: int = 0
    completeness: int = 0
    active: bool = False
    distilling: bool = False
    has_persona: bool = False

    def render(self) -> str:
        """渲染成一行文本。"""
        mark = "▶" if self.active else "·"
        name = self.nickname or "未知昵称"
        spin = " 🟠" if self.distilling else ""
        persona = " · 已有人格" if self.has_persona else ""
        return (
            f"{mark} {name}({self.qq})@{self.group_id}{spin}\n"
            f"   {fmt_int(self.total)} 条 · {self.round_no} 轮 · "
            f"档案 {self.completeness}% · 待蒸馏 {fmt_int(self.undistilled)}{persona}"
        )


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
    targets: list[TargetRow] = field(default_factory=list)
    # 每日定时总结
    daily_enabled: bool = False
    daily_time: str = ""
    daily_last: str = ""


def render_no_target() -> str:
    """未设定目标时的友好引导面板。"""
    lines = [
        f"🧪 群友蒸馏 · {PLUGIN_DISPLAY}",
        SEP,
        "❗ 还没有设定蒸馏目标",
        "请在 WebUI 插件配置里添加「蒸馏目标」（可加多个），",
        "或直接在群里用指令：/zl add <群号> <QQ号>",
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
    ]

    if data.daily_enabled:
        daily = f"⏰ 每日总结  {data.daily_time}"
        if data.daily_last:
            daily += f" · {data.daily_last}"
        lines.append(daily)

    # 多目标时才追加清单，避免单目标用户被多余信息干扰
    if len(data.targets) > 1:
        lines.append(SEP)
        lines.append(f"🎯 目标清单（{len(data.targets)} 个，▶ 为当前）")
        lines.extend(row.render() for row in data.targets)

    lines.append(SEP)
    lines.append("💡 /zl list 目标清单 · /zl profile 看档案 · /zl persona 生成人格 · /zl help 全部指令")
    return "\n".join(lines)


def render_target_list(rows: list[TargetRow], active_key: str = "") -> str:
    """渲染 ``/zl list`` 的目标清单。"""
    if not rows:
        return (
            "📭 还没有任何蒸馏目标。\n"
            "用 /zl add <群号> <QQ号> [昵称] 添加，\n"
            "或在 WebUI 插件配置的「蒸馏目标清单」里添加。"
        )

    lines = [
        f"🎯 蒸馏目标清单（共 {len(rows)} 个）",
        SEP,
    ]
    for row in rows:
        row.active = row.active or (bool(active_key) and row.key == active_key)
        lines.append(row.render())
    lines.append(SEP)
    lines.append("💡 /zl use <QQ号> 切换当前目标 · /zl del <QQ号> 删除")
    lines.append("   同一群可以挂多个群友，多个群也可以各挂几个。")
    return "\n".join(lines)


def render_profile(
    snapshot: Optional[dict[str, Any]],
    nickname: str,
    qq: str,
    group_id: str = "",
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

    header = f"🧬 {nickname or '未知昵称'} 的人格档案（{qq}）"
    if group_id:
        header += f" @ 群 {group_id}"
    samples = prompts.snap_samples(snapshot)
    lines = [
        header,
        f"第 {meta.get('distill_round', 0)} 轮 · 完整度 {compute_completeness(snapshot)}% "
        f"· 语料 {fmt_int(meta.get('total_messages', 0))} 条 "
        f"· 应答样例 {len(samples)} 条",
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
                    trigger = str(item.get("trigger") or "").strip()
                    if trigger:
                        lines.append(f"    · 出现时机：{trigger}")
                else:
                    lines.append(f"- {item}")
        lines.append("")

    lines.append("【场景应答样例】")
    if samples:
        for item in samples[:5]:
            if isinstance(item, dict):
                situation = str(item.get("situation") or "").strip()
                reply = str(item.get("reply") or "").strip()
                if situation and reply:
                    lines.append(f"- 【{situation}】{reply}")
                elif reply:
                    lines.append(f"- {reply}")
            else:
                lines.append(f"- {item}")
        if len(samples) > 5:
            lines.append(f"……（共 {len(samples)} 条，完整内容见 /zl export）")
    else:
        lines.append("- （暂无）")
    lines.append("")

    lines.append("【⚠️ 人工纠正层（优先级最高）】")
    if corrections:
        for c in corrections:
            lines.append(f"- {c}")
    else:
        lines.append("- （暂无）")

    lines.append(SEP)
    lines.append("💡 /zl persona 生成人格模板 · /zl export 导出 Markdown 档案")
    return "\n".join(lines)


def render_persona_header(
    nickname: str, qq: str, group_id: str, persona_id: str, chunk_count: int
) -> str:
    """渲染 ``/zl persona`` 的头部说明，后面跟着人格模板正文。"""
    lines = [
        f"🪪 人格模板 · {nickname or '未知昵称'}（{qq}）@ {group_id}",
        SEP,
        f"建议人格 ID：{persona_id}",
        "用法：把下面的内容整段复制，粘到 AstrBot WebUI 的「人格设定」里即可；",
        "或者直接发 /zl push，让它自动帮你创建/更新人格。",
        SEP,
    ]
    if chunk_count > 1:
        lines.append(f"（内容较长，已切成 {chunk_count} 条发送，按顺序拼起来用）")
        lines.append(SEP)
    return "\n".join(lines)


def render_help(prefix: str = "/zl") -> str:
    """渲染帮助文本。"""
    p = prefix
    lines = [
        f"🧪 {PLUGIN_DISPLAY} · 指令帮助",
        SEP,
        "— 目标管理 —",
        f"{p} list            查看全部目标及各自进度",
        f"{p} add <群号> <QQ> [昵称]   添加一个蒸馏目标",
        f"{p} del <QQ号> [purge]      删除目标（加 purge 连语料档案一起删）",
        f"{p} use <QQ号>      切换当前目标",
        "— 采集与蒸馏 —",
        f"{p}                查看当前目标进度面板",
        f"{p} on / off       开启 / 关闭采集",
        f"{p} distill [all]  手动蒸馏（加 all 则依次蒸馏全部目标）",
        f"{p} digest [all]   立即跑一次「每日总结」（用当天的全部对话蒸馏）",
        "— 档案与人格 —",
        f"{p} profile        查看当前目标的 5 层人格档案",
        f"{p} persona        生成可粘贴进 AstrBot 的人格模板",
        f"{p} push           把人格模板直接写进 AstrBot 人格设定",
        f"{p} export         导出 persona_<QQ>.md 与人格模板到数据目录",
        f"{p} correct <内容>  追加人工纠正（别名：{p} 纠正 <内容>）",
        f"{p} reset confirm  清空当前目标的语料（需二次确认）",
        f"{p} help           显示本帮助",
        SEP,
        "⚠️ 仅供娱乐，请勿用于侵犯他人隐私。",
    ]
    return "\n".join(lines)
