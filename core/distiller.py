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
    from .targets import TargetSpec
except ImportError:  # pragma: no cover - 兼容顶层导入
    import prompts  # type: ignore
    from collector import RuntimeState  # type: ignore
    from storage import Storage  # type: ignore
    from targets import TargetSpec  # type: ignore

PLUGIN_NAME = "astrbot_plugin_group_distiller"


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass
class DistillResult:
    """一次触发操作的结果。"""

    ok: bool
    message: str


@dataclass
class DailyItem:
    """每日总结里单个目标的结果。

    Attributes:
        spec: 目标。
        ok: 是否处理完毕（False 表示遇到可重试的故障）。
        detail: 人话说明。
        snapshot: 该目标总结后的人格快照；没有档案时为 None。播报与"总结后同步
            人格"都要用它。
    """

    spec: TargetSpec
    ok: bool
    detail: str
    snapshot: Optional[dict[str, Any]] = None


def empty_snapshot() -> dict[str, Any]:
    """返回一份空的 Persona 快照。"""
    return {
        "layers": {key: [] for key in prompts.LAYER_KEYS},
        "speech_samples": [],
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


def _with_optional_fields(item: dict[str, Any], raw: Any) -> dict[str, Any]:
    """把条目里可选的情境字段（``trigger`` / ``avoid``）带进规整后的结构。

    这两个字段描述"什么时候会这样 / 什么时候不会"，对"演得像"的帮助比结论
    本身更大，所以在合并过程中必须保留而不是丢掉。
    """
    if not isinstance(raw, dict):
        return item
    for field in prompts.ITEM_OPTIONAL_FIELDS:
        value = str(raw.get(field) or "").strip()
        if value:
            item[field] = value[:120]
    return item


def _coerce_sample(raw: Any) -> Optional[dict[str, Any]]:
    """把 LLM 返回的场景应答样例规整成统一结构，非法返回 None。"""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        return {"situation": "", "reply": text, "quotes": [], "evidence": 1}
    if not isinstance(raw, dict):
        return None

    situation = str(raw.get("situation") or raw.get("scene") or "").strip()
    reply = str(raw.get("reply") or raw.get("answer") or "").strip()
    if not situation and not reply:
        return None

    quotes: list[str] = []
    for q in _as_list(raw.get("quotes")):
        qs = str(q).strip()
        if qs and qs not in quotes:
            quotes.append(qs)
        if len(quotes) >= 3:
            break

    return {
        "situation": situation[:120],
        "reply": reply[:300],
        "quotes": quotes,
        "evidence": 1,
    }


def sample_key_of(sample: Any) -> str:
    """场景应答样例的去重键：情境 + 回复一起归一化。"""
    if not isinstance(sample, dict):
        return normalize_text(sample)
    return normalize_text(
        f"{sample.get('situation') or ''}|{sample.get('reply') or ''}"
    )


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
                _with_optional_fields(coerced, it)
            cleaned.append(coerced)
        snap["layers"][key] = cleaned

    # 场景应答样例：保留原有证据强度
    samples: list[dict[str, Any]] = []
    for raw in _as_list(existing.get(prompts.SAMPLE_KEY)):
        sample = _coerce_sample(raw)
        if sample is None:
            continue
        if isinstance(raw, dict):
            try:
                sample["evidence"] = int(raw.get("evidence", 1) or 1)
            except (TypeError, ValueError):
                sample["evidence"] = 1
        samples.append(sample)
    snap[prompts.SAMPLE_KEY] = samples

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
            _with_optional_fields(item, raw)
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
                # 已有的条目缺情境字段时，用新证据补上（情境信息越全越像）
                for field in prompts.ITEM_OPTIONAL_FIELDS:
                    if not target.get(field) and item.get(field):
                        target[field] = item[field]
            else:
                layers.setdefault(key, []).append(item)
                index[norm] = item

    # 场景应答样例：按「情境 + 回复」去重，重复出现只累加证据
    samples: list[dict[str, Any]] = _as_list(snap.get(prompts.SAMPLE_KEY))
    sample_index = {
        sample_key_of(item): item for item in samples if isinstance(item, dict)
    }
    for raw in _as_list(analysis.get(prompts.SAMPLE_KEY) if isinstance(analysis, dict) else None):
        sample = _coerce_sample(raw)
        if sample is None:
            continue
        norm = sample_key_of(sample)
        if not norm:
            continue
        if norm in sample_index:
            target = sample_index[norm]
            try:
                target["evidence"] = int(target.get("evidence", 1)) + 1
            except (TypeError, ValueError):
                target["evidence"] = 2
        else:
            samples.append(sample)
            sample_index[norm] = sample
    # 超出上限时保留证据最强的那些（保持原有相对顺序）
    if len(samples) > prompts.MAX_SPEECH_SAMPLES:
        keep = sorted(
            range(len(samples)),
            key=lambda i: int(samples[i].get("evidence", 1) or 1),
            reverse=True,
        )[: prompts.MAX_SPEECH_SAMPLES]
        keep_set = set(keep)
        samples = [s for i, s in enumerate(samples) if i in keep_set]
    snap[prompts.SAMPLE_KEY] = samples

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
        # 当前正在处理的目标；由 _run 在逐个目标时写入，_distill_once 读取
        self._current: Optional[TargetSpec] = None
        # 单轮蒸馏成功后的回调（由 main 注入，用于「蒸馏达标自动写入人格」）
        self._on_distilled: Optional[Any] = None

    def set_on_distilled(self, callback: Optional[Any]) -> None:
        """注册蒸馏成功回调：``await callback(target, snapshot)``。

        回调异常会被吞掉并记日志，绝不影响蒸馏主流程。
        """
        self._on_distilled = callback

    def current_target(self) -> Optional[TargetSpec]:
        """当前正在蒸馏的目标（空闲时为 None）。"""
        return self._current

    # ------------------------------------------------------------------ #
    # 触发
    # ------------------------------------------------------------------ #

    def is_running(self) -> bool:
        """是否有蒸馏在进行中。"""
        if self._running:
            return True
        return self._task is not None and not self._task.done()

    def _pick_auto_candidate(self) -> Optional[TargetSpec]:
        """挑一个"最欠蒸馏"的目标（未蒸馏语料最多）。

        只看内存计数，保证在每条群消息的热路径上是 O(目标数) 而非查库。
        """
        interval = max(
            1, self._safe_int(self.config.get("distill_interval_messages", 200), 200)
        )
        best: Optional[TargetSpec] = None
        best_count = 0
        for spec in self.state.collect_targets():
            count = int(self.state.counts_for(spec).get("undistilled", 0))
            if count >= interval and count > best_count:
                best, best_count = spec, count
        return best

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        """把任意配置值安全转换为 int。"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    async def maybe_auto(self, umo: str) -> None:
        """自动蒸馏检查：某个目标的语料攒够了就触发一轮。"""
        try:
            if not bool(self.config.get("auto_distill", True)):
                return
            if self.is_running():
                return
            spec = self._pick_auto_candidate()
            if spec is None:
                return
            await self.trigger(umo, manual=False, target=spec)
        except Exception as exc:  # noqa: BLE001 - 自动路径必须兜底
            logger.error("[%s] 自动蒸馏检查异常: %s", self.plugin_name, exc)

    async def trigger(
        self,
        umo: str,
        manual: bool = False,
        target: Optional[TargetSpec] = None,
        all_targets: bool = False,
    ) -> DistillResult:
        """触发蒸馏（异步后台执行，立即返回）。

        Args:
            umo: 会话来源（``unified_msg_origin``），用于选 Provider 与发消息。
            manual: 是否为手动触发（只影响返回文案）。
            target: 指定要蒸馏的目标；为 None 时用当前选中目标。
            all_targets: 为 True 时依次蒸馏全部目标（此时忽略 ``target``）。

        Returns:
            :class:`DistillResult`，告诉调用方是否成功排队。
        """
        if self.is_running():
            return DistillResult(False, "⏳ 已有一轮蒸馏正在进行，请稍候再试。")

        if all_targets:
            queue = self.state.collect_targets()
            if not queue:
                return DistillResult(False, "⚠️ 还没有设定蒸馏目标，先用 /zl add 或 WebUI 配置。")
        else:
            if target is None:
                target = self.state.active_target()
            if target is None:
                return DistillResult(False, "⚠️ 还没有设定蒸馏目标，先用 /zl add 或 WebUI 配置。")
            queue = [target]

        self._task = asyncio.create_task(self._run(umo, queue))
        if all_targets:
            return DistillResult(
                True, f"🔬 已开始蒸馏全部 {len(queue)} 个目标，完成后可用 /zl list 查看进度。"
            )
        if manual:
            return DistillResult(
                True,
                f"🔬 已开始手动蒸馏：{queue[0].label()}，完成后可用 /zl 查看进度。",
            )
        return DistillResult(
            True, f"🔬 {queue[0].label()} 已达到阈值，自动蒸馏已开始。"
        )

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #

    async def _run(self, umo: str, targets: Optional[list[TargetSpec]] = None) -> None:
        """执行一轮（可含多个目标）蒸馏（内部方法，异常全部兜底）。"""
        queue = targets if targets else None
        if queue is None:
            active = self.state.active_target()
            queue = [active] if active is not None else []

        async with self._lock:
            self._running = True
            try:
                for spec in queue:
                    self._current = spec
                    try:
                        await self._distill_once(umo)
                    except Exception as exc:  # noqa: BLE001 - 单个目标失败不影响其它目标
                        logger.error(
                            "[%s] 目标 %s 蒸馏失败: %s",
                            self.plugin_name,
                            spec.label(),
                            exc,
                            exc_info=True,
                        )
            finally:
                self._running = False
                self._current = None

    async def _distill_once(self, umo: str, target: Optional[TargetSpec] = None) -> None:
        """对单个目标执行一轮「未蒸馏语料」蒸馏。

        Args:
            umo: 会话来源。
            target: 目标；为 None 时依次回退到「当前正在处理的目标」→「选中目标」。
        """
        spec = target or self._current or self.state.active_target()
        if spec is None:
            logger.info("[%s] 无目标可蒸馏，跳过。", self.plugin_name)
            return

        records = await self.storage.fetch_undistilled(
            spec.group_id, spec.qq_id, self._batch_limit()
        )
        if not records:
            logger.info("[%s] %s 无待蒸馏语料，跳过本轮。", self.plugin_name, spec.label())
            return

        await self._distill_records(umo, spec, records)

    def _batch_limit(self) -> int:
        """单轮投喂语料条数上限。"""
        return max(
            1, self._safe_int(self.config.get("distill_batch_messages", 300), 300)
        )

    def _max_chars(self) -> int:
        """单轮投喂语料的字符预算（超出会按信息量抽样）。

        语料给得越多，分析出的人格越细；代价是 token。默认 12000 是"信息量"
        与"成本"之间比较舒服的位置，想更细可以往上调。
        """
        return max(
            2000, self._safe_int(self.config.get("distill_max_chars", 12000), 12000)
        )

    async def _distill_records(
        self,
        umo: str,
        spec: TargetSpec,
        records: list[dict[str, Any]],
        *,
        notify: bool = True,
    ) -> tuple[bool, str, Optional[dict[str, Any]]]:
        """对**给定的这批语料**跑一轮 LLM 蒸馏并落库。

        盘中自动蒸馏与每日定时总结共用这一条链路，保证两条路径的提示词、
        合并规则、计数刷新、回调行为完全一致。

        Args:
            umo: 会话来源（用于选 Provider）。
            spec: 目标。
            records: 待蒸馏的语料行（来自 storage）。
            notify: 是否触发 ``on_distilled`` 回调。每日总结会传 False ——
                因为总结结束后由它自己统一同步人格，避免同一次总结写两遍。

        Returns:
            ``(是否成功, 说明, 合并后的快照)``。失败时快照为 None，且语料保持
            未蒸馏状态，下一轮可以重来。
        """
        if not records:
            return False, "没有语料", None

        group_id = spec.group_id
        qq = spec.qq_id
        existing = await self.storage.get_persona(group_id, qq)
        provider = await self._get_provider(umo)
        if provider is None:
            logger.error("[%s] 未获取到 LLM Provider，本轮蒸馏中止。", self.plugin_name)
            return False, "未获取到 LLM Provider", None

        # 追加约束走"模板注释剥离"：配置默认给的是带 # 的模板，用户启用后才生效
        extra = prompts.strip_template_comments(
            self.config.get("custom_prompt_extra", "")
        )
        user_prompt = prompts.build_analyzer_user(
            nickname=spec.nickname,
            qq=qq,
            records=records,
            existing_snapshot=existing,
            extra=extra,
            max_chars=self._max_chars(),
        )

        text = await self._chat(provider, prompts.ANALYZER_SYSTEM, user_prompt)
        analysis = parse_analysis(text)
        if analysis is None:
            logger.error(
                "[%s] LLM 返回无法解析为 JSON，本轮中止（语料保持未蒸馏）。", self.plugin_name
            )
            return False, "LLM 返回无法解析", None

        corrections = await self.storage.get_corrections(group_id, qq)
        total = await self.storage.count_messages(group_id, qq)
        undistilled = await self.storage.count_undistilled(group_id, qq)
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
                "group_id": group_id,
                "target_qq": qq,
                "nickname": spec.nickname,
            },
        )

        await self.storage.save_persona(group_id, qq, merged)
        await self.storage.mark_distilled([int(r["id"]) for r in records])
        # 用数据库的真实统计刷新计数缓存（本轮之后 undistilled 已归零一批）
        self.state.set_counts(spec.key, total, undistilled)

        round_no = int(merged["meta"].get("distill_round", 0) or 0)
        logger.info(
            "[%s] %s 蒸馏完成：本轮 %d 条，轮次 %d，剩余未蒸馏 %d 条。",
            self.plugin_name,
            spec.label(),
            len(records),
            round_no,
            undistilled,
        )

        # 通知外部（例如「达到完整度阈值就写入 AstrBot 人格」）
        if notify and self._on_distilled is not None:
            try:
                await self._on_distilled(spec, merged)
            except Exception as exc:  # noqa: BLE001 - 回调失败不影响蒸馏结果
                logger.error("[%s] 蒸馏回调执行失败: %s", self.plugin_name, exc)

        return True, f"本轮 {len(records)} 条，第 {round_no} 轮", merged

    # ------------------------------------------------------------------ #
    # 每日定时总结
    # ------------------------------------------------------------------ #

    async def digest_day(
        self,
        umo: str,
        spec: TargetSpec,
        start_ts: int,
        end_ts: int,
        *,
        min_messages: int = 0,
    ) -> tuple[bool, str]:
        """把某个目标在 ``[start_ts, end_ts]`` 区间内的**全部语料**蒸一遍。

        与 :meth:`_distill_once` 的区别：不限定"未蒸馏"，当天聊过的内容全部
        重新过一遍 —— 这正是"总结今天"的语义。

        Args:
            umo: 会话来源。
            spec: 目标。
            start_ts: 区间起点（一般是当天 00:00）。
            end_ts: 区间终点（一般是当前时刻）。
            min_messages: 当天语料少于这个数就跳过（0 表示不设限）。

        Returns:
            ``(是否处理完毕, 说明)``。返回 True 表示不必重试 —— 包括
            "当天没语料""低于阈值"这类正常的跳过。返回 False 表示遇到
            可重试的故障（如 LLM 不可用）。
        """
        target = spec or self.state.active_target()
        if target is None:
            return True, "没有配置目标"

        day_count = await self.storage.count_range(
            target.group_id, target.qq_id, start_ts, end_ts
        )
        if day_count <= 0:
            return True, "当天没有语料"
        if min_messages > 0 and day_count < min_messages:
            return True, f"当天仅 {day_count} 条（低于阈值 {min_messages}）"

        records = await self.storage.fetch_range(
            target.group_id, target.qq_id, start_ts, end_ts, self._batch_limit()
        )
        if not records:
            return True, "当天没有语料"

        ok, detail, _ = await self._distill_records(umo, target, records, notify=False)
        return ok, detail

    async def run_daily_digest_detailed(
        self,
        umo_for: Any,
        start_ts: int,
        end_ts: int,
        *,
        min_messages: int = 0,
    ) -> list[DailyItem]:
        """对全部目标跑一次每日总结，返回**逐目标**的结果。

        逐目标返回是为了播报时能各发各的群：把 A 群的结果发到 B 群，等于
        平白泄露了"B 群某人是蒸馏目标"这件事。

        Args:
            umo_for: 字符串，或 ``(spec) -> str`` 的可调用对象。
            start_ts: 区间起点。
            end_ts: 区间终点。
            min_messages: 当天语料少于这个数就跳过该目标。

        Returns:
            每个目标一条 :class:`DailyItem`；没有目标时返回空列表。
        """
        specs = self.state.collect_targets()
        items: list[DailyItem] = []
        for spec in specs:
            try:
                umo = umo_for(spec) if callable(umo_for) else str(umo_for)
            except Exception:  # noqa: BLE001 - 取会话来源失败也要继续其它目标
                umo = str(umo_for)
            try:
                ok, detail = await self.digest_day(
                    umo, spec, start_ts, end_ts, min_messages=min_messages
                )
            except Exception as exc:  # noqa: BLE001 - 单个目标失败不影响其它目标
                logger.error(
                    "[%s] %s 每日总结失败: %s",
                    self.plugin_name,
                    spec.label(),
                    exc,
                    exc_info=True,
                )
                ok, detail = False, f"异常：{exc}"
            items.append(DailyItem(spec=spec, ok=ok, detail=detail))
        # 补齐快照：总结结束后要拿它去同步 AstrBot 人格
        for item in items:
            if item.snapshot is not None or not item.ok:
                continue
            try:
                item.snapshot = await self.storage.get_persona(
                    item.spec.group_id, item.spec.qq_id
                )
            except Exception as exc:  # noqa: BLE001 - 读快照失败不影响总结本身
                logger.error(
                    "[%s] 读取 %s 的快照失败: %s", self.plugin_name, item.spec.label(), exc
                )
        return items

    async def run_daily_digest(
        self,
        umo_for: Any,
        start_ts: int,
        end_ts: int,
        *,
        min_messages: int = 0,
    ) -> tuple[bool, str]:
        """对全部目标跑一次每日总结，返回 ``(是否全部处理完毕, 汇总说明)``。

        需要逐目标结果（例如要分群播报）时请用
        :meth:`run_daily_digest_detailed`。
        """
        items = await self.run_daily_digest_detailed(
            umo_for, start_ts, end_ts, min_messages=min_messages
        )
        if not items:
            return True, "没有配置任何目标"
        detail = "；".join(
            f"{item.spec.display_name}({item.spec.qq_id})：{item.detail}"
            for item in items
        )
        return all(item.ok for item in items), detail

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
