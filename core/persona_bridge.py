"""把蒸馏出的人格模板接进 AstrBot 的「人格设定」。

两条路径：

- **手动复制**：``build_astrbot_persona`` 生成一段系统提示词，用户自己粘到
  AstrBot 的「人格设定」里（任何版本都能用）。
- **一键写入**：调用 ``Context.persona_manager``，直接创建/更新一个 AstrBot
  人格（需要 AstrBot ≥ v4.0.0，且该版本暴露了 ``persona_manager``）。

本模块对 AstrBot 的接口调用全部做了「接口缺失 / 签名不符 / 抛异常」的兜底，
失败时退回"让用户手动复制"，绝不因为版本差异把插件搞崩。
"""

from __future__ import annotations

import asyncio
import inspect
import re
from pathlib import Path
from typing import Any, Optional

try:  # 允许在未安装 AstrBot 的环境（如单元测试）下导入
    from astrbot.api import logger
except ImportError:  # pragma: no cover - 仅在无 AstrBot 环境触发
    import logging

    logger = logging.getLogger("astrbot_plugin_group_distiller")

try:
    from .targets import TargetSpec
except ImportError:  # pragma: no cover - 兼容顶层导入
    from targets import TargetSpec  # type: ignore

PLUGIN_NAME = "astrbot_plugin_group_distiller"

# 默认的人格 ID 前缀，可在 WebUI 配置里改
DEFAULT_PERSONA_PREFIX = "群友蒸馏"

# 人格 ID 里不允许出现的字符：文件名非法字符 + 空白。
# persona_id 会同时被 AstrBot 当作人格名与 v3 配置里的 name，尽量干净些。
_ID_BAD_CHARS = re.compile(r'[\\/:*?"<>|\s\u3000]+')

# 人格 ID 长度上限，防止昵称很长时把 ID 撑爆
_MAX_ID_LEN = 60


async def _maybe_await(value: Any) -> Any:
    """兼容同步/异步两种接口：是 awaitable 就 await，否则直接返回。"""
    if inspect.isawaitable(value):
        return await value
    return value


def make_persona_id(prefix: str, nickname: str, qq: str, group_id: str = "") -> str:
    """按「前缀-昵称-QQ」拼一个人格 ID，并清理掉不适合做 ID 的字符。

    Args:
        prefix: 用户配置的前缀，为空时用默认前缀。
        nickname: 目标昵称。
        qq: 目标 QQ。
        group_id: 群号（仅在昵称为空、需要区分同名目标时使用）。

    Returns:
        清理后的短 ID。极端情况下也不会返回空串。
    """
    head = (prefix or "").strip() or DEFAULT_PERSONA_PREFIX
    tail = (nickname or "").strip() or (f"群{group_id}" if group_id else "群友")
    raw = f"{head}-{tail}-{qq}"
    clean = _ID_BAD_CHARS.sub("", raw).strip("-")
    if not clean:
        clean = f"{DEFAULT_PERSONA_PREFIX}-{qq}"
    return clean[:_MAX_ID_LEN]


def persona_manager_of(context: Any) -> Optional[Any]:
    """取出 AstrBot 的 persona_manager；取不到返回 None。"""
    try:
        return getattr(context, "persona_manager", None)
    except Exception:  # noqa: BLE001
        return None


def persona_supported(context: Any) -> bool:
    """当前 AstrBot 是否支持一键写入人格。"""
    mgr = persona_manager_of(context)
    if mgr is None:
        return False
    return all(
        hasattr(mgr, name)
        for name in ("get_persona", "create_persona", "update_persona")
    )


async def persona_exists(context: Any, persona_id: str) -> Optional[bool]:
    """判断人格是否已存在；接口不可用时返回 None（表示"说不准"）。"""
    mgr = persona_manager_of(context)
    if mgr is None or not hasattr(mgr, "get_persona"):
        return None
    try:
        await _maybe_await(mgr.get_persona(persona_id))
        return True
    except Exception:  # noqa: BLE001 - AstrBot 用 ValueError 表示不存在
        return False


async def push_persona(
    context: Any, persona_id: str, system_prompt: str
) -> tuple[bool, str]:
    """创建或更新一个 AstrBot 人格。

    已存在则更新 ``system_prompt``（保留它的其它设置），不存在则新建。
    任何异常都转成 ``(False, 人话原因)``，调用方据此提示用户改用手动复制。

    Args:
        context: AstrBot ``Context``。
        persona_id: 人格 ID（同时也是展示名）。
        system_prompt: 人格系统提示词。

    Returns:
        ``(是否成功, 结果说明)``。
    """
    prompt = (system_prompt or "").strip()
    if not prompt:
        return False, "人格模板是空的，先生成档案再来。"

    mgr = persona_manager_of(context)
    if mgr is None:
        return False, (
            "当前 AstrBot 没有暴露 persona_manager 接口（版本可能低于 v4.0.0），"
            "请改用 /zl persona 手动复制粘贴。"
        )

    exists = await persona_exists(context, persona_id)

    try:
        if exists and hasattr(mgr, "update_persona"):
            await _maybe_await(mgr.update_persona(persona_id, system_prompt=prompt))
            return True, f"已更新 AstrBot 人格「{persona_id}」。"
        if hasattr(mgr, "create_persona"):
            await _maybe_await(mgr.create_persona(persona_id, prompt))
            return True, f"已新建 AstrBot 人格「{persona_id}」。"
    except Exception as exc:  # noqa: BLE001 - 版本差异/数据库异常都兜住
        logger.error("[%s] 写入 AstrBot 人格失败: %s", PLUGIN_NAME, exc, exc_info=True)
        return False, f"写入 AstrBot 人格失败：{exc}（可改用 /zl persona 手动复制）"

    return False, "当前 AstrBot 的 persona_manager 不支持创建/更新人格，请手动复制。"


async def export_persona_file(
    out_dir: Path, spec: TargetSpec, text: str
) -> Optional[Path]:
    """把人格模板落一份纯文本到数据目录，便于备份与手动复制。

    Returns:
        写出的文件路径；失败返回 None。
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"astrbot_persona_{spec.qq_id}.txt"
        await asyncio.to_thread(path.write_text, text, "utf-8")
        return path
    except OSError as exc:
        logger.error("[%s] 导出人格模板失败: %s", PLUGIN_NAME, exc)
        return None
