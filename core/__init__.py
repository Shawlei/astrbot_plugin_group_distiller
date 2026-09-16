"""核心子包：目标、存储、采集、蒸馏、提示词、人格桥接与进度渲染。

本包内的模块刻意**不依赖 AstrBot 本体**（仅在缺失时回退到标准库 logger），
以便在无 AstrBot 环境下也能被单元测试直接导入。

``__all__`` 汇总了对外可用的公共对象，方便 ``from core import ...`` 使用。
"""

from __future__ import annotations

from . import (
    collector,
    distiller,
    persona_bridge,
    progress,
    prompts,
    storage,
    targets,
)
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
from .persona_bridge import (
    export_persona_file,
    make_persona_id,
    persona_exists,
    persona_supported,
    push_persona,
)
from .progress import (
    PLUGIN_DISPLAY,
    PanelData,
    TargetRow,
    chunk_text,
    render_help,
    render_no_target,
    render_panel,
    render_profile,
    render_target_list,
)
from .storage import MessageRecord, Storage, resolve_plugin_data_dir
from .targets import (
    TargetSpec,
    collect_config_targets,
    dedupe,
    is_valid_id,
    make_target,
    parse_targets_text,
    targets_from_json,
    targets_to_json,
)

__all__ = [
    # 子模块
    "collector",
    "distiller",
    "persona_bridge",
    "progress",
    "prompts",
    "storage",
    "targets",
    # 目标
    "TargetSpec",
    "collect_config_targets",
    "parse_targets_text",
    "make_target",
    "is_valid_id",
    "dedupe",
    "targets_to_json",
    "targets_from_json",
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
    # 人格桥接
    "make_persona_id",
    "push_persona",
    "persona_supported",
    "persona_exists",
    "export_persona_file",
    # 进度渲染
    "PanelData",
    "TargetRow",
    "PLUGIN_DISPLAY",
    "render_panel",
    "render_profile",
    "render_target_list",
    "render_help",
    "render_no_target",
    "chunk_text",
    # 存储
    "Storage",
    "MessageRecord",
    "resolve_plugin_data_dir",
]
