"""核心子包：存储、采集、蒸馏、提示词与进度渲染。

本包内的模块刻意**不依赖 AstrBot 本体**（仅在缺失时回退到标准库 logger），
以便在无 AstrBot 环境下也能被单元测试直接导入。

``__all__`` 汇总了对外可用的公共对象，方便 ``from core import ...`` 使用。
"""

from __future__ import annotations

from . import collector, distiller, progress, prompts, storage
from .collector import Collector, RuntimeState
from .distiller import (
    DistillResult,
    Distiller,
    compute_completeness,
    count_formed_layers,
    empty_snapshot,
    merge_snapshot,
    parse_analysis,
)
from .progress import (
    PLUGIN_DISPLAY,
    PanelData,
    render_help,
    render_no_target,
    render_panel,
    render_profile,
)
from .storage import MessageRecord, Storage, resolve_plugin_data_dir

__all__ = [
    # 子模块
    "collector",
    "distiller",
    "progress",
    "prompts",
    "storage",
    # 采集
    "Collector",
    "RuntimeState",
    # 蒸馏
    "Distiller",
    "DistillResult",
    "merge_snapshot",
    "empty_snapshot",
    "parse_analysis",
    "compute_completeness",
    "count_formed_layers",
    # 进度渲染
    "PanelData",
    "PLUGIN_DISPLAY",
    "render_panel",
    "render_profile",
    "render_help",
    "render_no_target",
    # 存储
    "Storage",
    "MessageRecord",
    "resolve_plugin_data_dir",
]
