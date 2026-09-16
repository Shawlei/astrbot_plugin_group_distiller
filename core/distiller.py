"""蒸馏引擎与 Persona 增量合并逻辑。

职责：

- **触发管理**：``maybe_auto`` 负责自动触发，``trigger`` 负责手动触发；
  任一时刻只允许一轮蒸馏在跑（``asyncio.Lock`` + 运行标志），避免并发烧 token。
- **LLM 调用**：整体 try/except 包住，失败必须降级而不是让插件崩溃。
- **增量 merge**：新证据只做补充/加权，不推翻已有高置信结论；冲突保留双方
  并标注 ``conflict``（呼应 pig-skill 的思路）。
- **纠正层**：人工纠正（``/zl 纠正``）与快照一并存储，优先级高于 LLM 推断。

本模块零 AstrBot 强依赖，可被单元测试直接导入。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

try:  # 允许在未安装 AstrBot 的环境（如单元测试）下导入
    from astrbot.api import logger
except ImportError:  # pragma: no cover - 仅在无 AstrBot 环境触发
    import logging

    logger = logging.getLogger("astrbot_plugin_group_distiller")

try:
    from . import prompts
    from .collector import RuntimeState
    from .storage import Storage
except ImportError:  # pragma: no cover - 兼容顶层导入
    import prompts  # type: ignore
    from collector import RuntimeState  # type: ignore
    from storage import Storage  # type: ignore

PLUGIN_NAME = "astrbot_plugin_group_distiller"


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass
class DistillResult:
    """一次触发操作的结果。"""

    ok: bool
    message: str


def empty_snapshot() -> dict[str, Any]:
    """返回一份空的 Persona 快照。"""
    return {
        "layers": {key: [] for key in prompts.LAYER_KEYS},
        "uncertainty": [],
        "corrections": [],
        "meta": {
            "distill_round": 0,
            "last_distill_at": 0,
            "total_messages": 0,
            "deduped_messages": 0,
            "span_start": 0,
            "span_end": 0,
            "created_at": int(time.time()),
        },
    }


def normalize_text(text: Any) -> str:
    """归一化文本用于去重：去空白 + 转小写。"""
    return re.sub(r"\s+", "", str(text or "")).lower()


def _as_list(value: Any) -> list[Any]:
    """把"设计上是数组"的字段归一化为 list，杜绝字符串被逐字符拆散。

    归一化规则：

    - ``None`` → ``[]``
    - ``list`` → 原样返回
    - ``str`` → 单元素列表（例如 ``"abc"`` → ``["abc"]``，而不是 ``["a","b","c"]``）
    - ``tuple`` / ``set`` → 转成 ``list``
    - 其它类型 → ``[]``，并记录一次 debug

    这样即便 LLM 把 ``uncertainty`` / ``quotes`` 之类字段返回成字符串，
    也不会污染快照。
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    if isinstance(value, (tuple, set)):
        return list(value)
    logger.debug(
        "[%s] 期望数组字段但收到 %s，已忽略。", PLUGIN_NAME, type(value).__name__
    )
    return []


def _coerce_item(raw: Any) -> Optional[dict[str, Any]]:
    """把 LLM 返回的任意条目规整为统一结构，非法则返回 None。"""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        return {"text": text, "confidence": 0.5, "evidence": 1, "quotes": [], "conflict": False}
    if not isinstance(raw, dict):
        return None

    text = str(raw.get("text") or "").strip()
    if not text:
        return None

    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    quotes: list[str] = []
    for q in _as_list(raw.get("quotes")):
        qs = str(q).strip()
        if qs and qs not in quotes:
            quotes.append(qs)
        if len(quotes) >= 3:
            break

    return {
        "text": text,
        "confidence": confidence,
        "evidence": 1,
        "quotes": quotes,
        "conflict": bool(raw.get("conflict", False)),
    }


def _copy_layers(existing: Optional[dict[str, Any]]) -> dict[str, Any]:
    """安全复制已有快照（规整条目、保留 evidence 与 meta）。"""
    snap = empty_snapshot()
    if not isinstance(existing, dict):
        return snap

    ex_layers = existing.get("layers") if isinstance(existing.get("layers"), dict) else {}
    for key in prompts.LAYER_KEYS:
        cleaned: list[dict[str, Any]] = []
        for it in _as_list(ex_layers.get(key)):
            coerced = _coerce_item(it)
            if coerced is None:
                continue
            if isinstance(it, dict):
                try:
                    coerced["evidence"] = int(it.get("evidence", 1) or 1)
                except (TypeError, ValueError):
                    coerced["evidence"] = 1
                coerced["conflict"] = bool(it.get("conflict", coerced["conflict"]))
            cleaned.append(coerced)
        snap["layers"][key] = cleaned

    snap["uncertainty"] = [str(u) for u in _as_list(existing.get("uncertainty"))]
    snap["corrections"] = [str(c) for c in _as_list(existing.get("corrections"))]
    if isinstance(existing.get("meta"), dict):
        snap["meta"].update(existing["meta"])
    return snap


def merge_snapshot(
    existing: Optional[dict[str, Any]],
    analysis: dict[str, Any],
    corrections: Optional[list[str]] = None,
    *,
    now: Optional[int] = None,
    meta_patch: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """把一轮增量分析合并进已有 Persona 快照。

    合并规则（确定性实现，无需二次 LLM 调用）：

    1. 同层内按归一化文本去重；命中已有条目则累加证据、取较大置信、补 quotes。
    2. 新条目直接追加，保留出现顺序（旧在前、新在后）。
    3. 冲突标注不互相覆盖：任一方标记 ``conflict`` 则整体标记冲突。
    4. ``distill_round`` 自增；``meta_patch`` 覆盖 meta 字段。
    5. 纠正层由调用方提供（来自数据库），以保持"优先级高于 LLM 推断"。

    Args:
        existing: 已有快照（可为 None）。
        analysis: 本轮 LLM 返回的分析 JSON。
        corrections: 当前全部人工纠正条目。
        now: 本轮蒸馏时间戳，默认取当前时间。
        meta_patch: 需要覆盖写入 meta 的字段（如 total_messages）。

    Returns:
        合并后的完整快照 dict。
    """
    snap = _copy_layers(existing)
    layers: dict[str, list[dict[str, Any]]] = snap["layers"]

    for key in prompts.LAYER_KEYS:
        incoming = _as_list(analysis.get(key) if isinstance(analysis, dict) else None)
        index: dict[str, dict[str, Any]] = {
            normalize_text(item.get("text")): item for item in layers.get(key, [])
        }
        for raw in incoming:
            item = _coerce_item(raw)
            if item is None:
                continue
            norm = normalize_text(item["text"])
            if not norm:
                continue
            if norm in index:
                target = index[norm]
                try:
                    target["evidence"] = int(target.get("evidence", 1)) + 1
                except (TypeError, ValueError):
                    target["evidence"] = 2
                target["confidence"] = max(
                    float(target.get("confidence", 0.0)), item["confidence"]
                )
                quotes = target.setdefault("quotes", [])
                for q in item["quotes"]:
                    if q and q not in quotes and len(quotes) < 3:
                        quotes.append(q)
                if item.get("conflict"):
                    target["conflict"] = True
            else:
                layers.setdefault(key, []).append(item)
                index[norm] = item

    # 累积不确定项（analysis 里若误传字符串，也只会当成一条，不拆字符）
    uncertainty = _as_list(snap.get("uncertainty"))
    for u in _as_list(analysis.get("uncertainty") if isinstance(analysis, dict) else None):
        su = str(u).strip()
        if su and su not in uncertainty:
            uncertainty.append(su)
    snap["uncertainty"] = uncertainty

    # 纠正层（权威来源是调用方传入的 corrections；LLM 输出里的同名字段一律忽略）
    snap["corrections"] = [str(c) for c in _as_list(corrections)]

    # meta
    meta = snap.get("meta", {})
    meta["distill_round"] = int(meta.get("distill_round", 0) or 0) + 1
    meta["last_distill_at"] = int(now or time.time())
    if meta_patch:
        meta.update(meta_patch)
    snap["meta"] = meta

    return snap


def count_formed_layers(snapshot: Optional[dict[str, Any]]) -> int:
    """统计已"成型"的层数（该层至少有一条结论）。"""
    if not isinstance(snapshot, dict):
        return 0
    layers = snapshot.get("layers")
    if not isinstance(layers, dict):
        return 0
    return sum(1 for key in prompts.LAYER_KEYS if layers.get(key))


def compute_completeness(snapshot: Optional[dict[str, Any]]) -> int:
    """档案完整度（0~100）：成型层数 / 总层数。"""
    total = len(prompts.LAYER_KEYS)
    if total == 0:
        return 0
    return int(round(count_formed_layers(snapshot) / total * 100))


def parse_analysis(text: Any) -> Optional[dict[str, Any]]:
    """从 LLM 输出里稳健地抽取 JSON 对象。

    支持剥离 Markdown 代码块、忽略 JSON 前后的解释文字。

    Returns:
        解析成功的 dict，失败返回 None。
    """
    if not text:
        return None
    raw = str(text).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class Distiller:
    """蒸馏引擎。"""

    def __init__(
        self,
        storage: Storage,
        config: Any,
        state: RuntimeState,
        context: Any,
        plugin_name: str = PLUGIN_NAME,
    ) -> None:
        """构造蒸馏引擎（不做 IO）。

        Args:
            storage: 存储层。
            config: AstrBot 配置对象。
            state: 共享运行时状态。
            context: AstrBot ``Context``，用于获取 LLM Provider。
            plugin_name: 日志前缀。
        """
        self.storage = storage
        self.config = config
        self.state = state
        self.context = context
        self.plugin_name = plugin_name
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # 触发
    # ------------------------------------------------------------------ #

    def is_running(self) -> bool:
        """是否有蒸馏在进行中。"""
        if self._running:
            return True
        return self._task is not None and not self._task.done()

    async def maybe_auto(self, umo: str) -> None:
        """自动蒸馏检查：攒够语料且开启自动蒸馏时触发一轮。"""
        try:
            if not bool(self.config.get("auto_distill", True)):
                return
            if self.is_running() or not self.state.has_target():
                return
            interval = max(1, int(self.config.get("distill_interval_messages", 200) or 200))
            if self.state.undistilled < interval:
                return
            await self.trigger(umo, manual=False)
        except Exception as exc:  # noqa: BLE001 - 自动路径必须兜底
            logger.error("[%s] 自动蒸馏检查异常: %s", self.plugin_name, exc)

    async def trigger(self, umo: str, manual: bool = False) -> DistillResult:
        """触发一轮蒸馏（异步后台执行，立即返回）。

        Args:
            umo: 会话来源（``unified_msg_origin``），用于选 Provider 与发消息。
            manual: 是否为手动触发。

        Returns:
            :class:`DistillResult`，告诉调用方是否成功排队。
        """
        if self.is_running():
            return DistillResult(False, "⏳ 已有一轮蒸馏正在进行，请稍候再试。")
        if not self.state.has_target():
            return DistillResult(False, "⚠️ 还没有设定蒸馏目标，先用 /zl set 或 WebUI 配置。")
        self._task = asyncio.create_task(self._run(umo))
        if manual:
            return DistillResult(True, "🔬 已开始手动蒸馏，完成后可用 /zl 查看进度。")
        return DistillResult(True, "🔬 已达到阈值，自动蒸馏已开始。")

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #

    async def _run(self, umo: str) -> None:
        """执行一轮完整蒸馏（内部方法，异常全部兜底）。"""
        async with self._lock:
            self._running = True
            try:
                await self._distill_once(umo)
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] 蒸馏失败: %s", self.plugin_name, exc, exc_info=True)
            finally:
                self._running = False

    async def _distill_once(self, umo: str) -> None:
        group_id = self.state.group_id
        qq = self.state.qq_id
        batch_limit = max(1, int(self.config.get("distill_batch_messages", 300) or 300))

        records = await self.storage.fetch_undistilled(group_id, qq, batch_limit)
        if not records:
            logger.info("[%s] 无待蒸馏语料，跳过本轮。", self.plugin_name)
            return

        existing = await self.storage.get_persona(group_id, qq)
        provider = await self._get_provider(umo)
        if provider is None:
            logger.error("[%s] 未获取到 LLM Provider，本轮蒸馏中止。", self.plugin_name)
            return

        extra = str(self.config.get("custom_prompt_extra", "") or "")
        user_prompt = prompts.build_analyzer_user(
            nickname=self.state.nickname,
            qq=qq,
            records=records,
            existing_snapshot=existing,
            extra=extra,
        )

        text = await self._chat(provider, prompts.ANALYZER_SYSTEM, user_prompt)
        analysis = parse_analysis(text)
        if analysis is None:
            logger.error("[%s] LLM 返回无法解析为 JSON，本轮中止（语料保持未蒸馏）。", self.plugin_name)
            return

        corrections = await self.storage.get_corrections(group_id, qq)
        total = await self.storage.count_messages(group_id, qq)
        span_lo, span_hi = await self.storage.get_time_span(group_id, qq)
        merged = merge_snapshot(
            existing,
            analysis,
            corrections,
            meta_patch={
                "total_messages": total,
                "deduped_messages": total,
                "span_start": span_lo,
                "span_end": span_hi,
            },
        )

        await self.storage.save_persona(group_id, qq, merged)
        await self.storage.mark_distilled([int(r["id"]) for r in records])
        self.state.undistilled = await self.storage.count_undistilled(group_id, qq)
        logger.info(
            "[%s] 蒸馏完成：本轮 %d 条，轮次 %d，剩余未蒸馏 %d 条。",
            self.plugin_name,
            len(records),
            merged["meta"].get("distill_round", 0),
            self.state.undistilled,
        )

    # ------------------------------------------------------------------ #
    # LLM 交互
    # ------------------------------------------------------------------ #

    async def _get_provider(self, umo: str) -> Any:
        """获取 LLM Provider：优先配置指定，回退默认。失败返回 None。"""
        provider_id = str(self.config.get("llm_provider_id", "") or "").strip()
        if provider_id:
            try:
                provider = self.context.get_provider_by_id(provider_id)
                if provider is not None:
                    return provider
                logger.warning("[%s] 未找到指定 Provider %s，回退默认。", self.plugin_name, provider_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] 获取指定 Provider 失败: %s", self.plugin_name, exc)
        try:
            return await self.context.get_using_provider_async(umo=umo)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 获取默认 Provider 失败: %s", self.plugin_name, exc)
            return None

    async def _chat(self, provider: Any, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM 并返回文本；任何异常都降级为空串。"""
        try:
            resp = await provider.text_chat(prompt=user_prompt, system_prompt=system_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] LLM 调用失败: %s", self.plugin_name, exc)
            return ""
        return str(
            getattr(resp, "completion_text", "")
            or getattr(resp, "text", "")
            or ""
        )
