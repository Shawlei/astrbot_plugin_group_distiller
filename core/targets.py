"""蒸馏目标（群 + 群友）的定义、解析与序列化。

一个「蒸馏目标」= **某个群里的某个群友**。插件支持同时挂多个目标：
同一个群的多个群友、多个群各若干群友，混着配都行。

配置来源有三条路径，按顺序合并（同 ``群号:QQ号`` 去重，先到先得）：

1. ``targets``：WebUI 的可视化清单（``template_list`` 类型，AstrBot ≥ v4.10.4）；
2. ``targets_text``：纯文本清单，一行一个目标，任何版本都能用；
3. ``target_group_id`` / ``target_qq_id``：v0.1.0 的遗留扁平字段，仅作兼容。

运行期通过 ``/zl add``、``/zl del`` 修改的目标会持久化到数据库 ``state`` 表，
下次启动时优先于配置文件生效（用户手改的意图高于静态配置）。

本模块零 AstrBot 依赖，可被单元测试直接导入。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

# QQ 号 / 群号都是纯数字，且长度至少 5 位（QQ 目前最短 5 位）。
# 设下限是为了拦住「123 4」这类把备注写进号码列的误输入。
MIN_ID_LEN = 5
MAX_ID_LEN = 20

# 备注昵称的长度上限，防止有人把整段话填进去撑爆面板
MAX_NICKNAME_LEN = 24

# 文本清单里允许出现的分隔符（中英文逗号、顿号、竖线、分号、空白）
_SPLIT_RE = re.compile(r"[\s,，、|｜;；]+")

# 注释行前缀：以这些字符开头的行整行忽略
_COMMENT_PREFIXES = ("#", "//", "－", "—")


@dataclass(frozen=True)
class TargetSpec:
    """一个蒸馏目标：某群里的某个群友。

    Attributes:
        group_id: 目标所在群号（纯数字字符串）。
        qq_id: 被蒸馏群友的 QQ 号（纯数字字符串）。
        nickname: 备注昵称，仅用于展示与提示词，可为空。
    """

    group_id: str
    qq_id: str
    nickname: str = ""

    @property
    def key(self) -> str:
        """目标的唯一键，形如 ``群号:QQ号``。"""
        return f"{self.group_id}:{self.qq_id}"

    @property
    def display_name(self) -> str:
        """展示用昵称，缺省时给一个占位串。"""
        return self.nickname or "未知昵称"

    def label(self) -> str:
        """一行式描述，形如 ``张三(123456789)@987654321``。"""
        return f"{self.display_name}({self.qq_id})@{self.group_id}"

    def to_dict(self) -> dict[str, str]:
        """转为可 JSON 序列化的 dict。"""
        return {
            "group_id": self.group_id,
            "qq_id": self.qq_id,
            "nickname": self.nickname,
        }

    @staticmethod
    def from_dict(raw: Any) -> Optional[TargetSpec]:
        """从 dict / WebUI 清单条目构造目标，非法返回 None。

        ``template_list`` 生成的条目里混有 ``__template_key`` 之类的内部字段，
        这里只挑需要的三个键，其余一律忽略。
        """
        if not isinstance(raw, dict):
            return None
        return make_target(
            raw.get("group_id"),
            raw.get("qq_id"),
            raw.get("nickname", ""),
        )


def _clean_id(value: Any) -> str:
    """把任意输入规整成纯数字 ID 字符串；不合法返回空串。

    会剥掉常见的复制粘贴噪声：首尾空白、``群号:`` 前缀、``@`` 前缀。
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    # 容忍「群号:123456」「@123456」这类从别处粘过来的写法
    text = re.sub(r"^(?:群号|群聊|群|qq号|qq|QQ号|QQ|＠|@)\s*[:：]?\s*", "", text)
    text = text.strip()
    if not text.isdigit():
        return ""
    if not (MIN_ID_LEN <= len(text) <= MAX_ID_LEN):
        return ""
    return text


def _clean_nickname(value: Any) -> str:
    """规整备注昵称：去空白、限长。"""
    text = str(value if value is not None else "").strip()
    return text[:MAX_NICKNAME_LEN]


def is_valid_id(value: Any) -> bool:
    """判断一个群号/QQ 号是否合法（纯数字且长度在合理区间）。"""
    return bool(_clean_id(value))


def make_target(
    group_id: Any, qq_id: Any, nickname: Any = ""
) -> Optional[TargetSpec]:
    """构造一个目标，任一必填项非法则返回 None。"""
    group = _clean_id(group_id)
    qq = _clean_id(qq_id)
    if not group or not qq:
        return None
    return TargetSpec(group_id=group, qq_id=qq, nickname=_clean_nickname(nickname))


def normalize_targets(raw: Any) -> list[TargetSpec]:
    """把任意「目标集合」输入规整成去重后的目标列表。

    接受：``TargetSpec`` 列表、``dict`` 列表（WebUI 配置形态）、``None``。
    """
    if not raw:
        return []
    if isinstance(raw, (TargetSpec, dict)):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []

    out: list[TargetSpec] = []
    seen: set[str] = set()
    for item in raw:
        spec = item if isinstance(item, TargetSpec) else TargetSpec.from_dict(item)
        if spec is None or spec.key in seen:
            continue
        seen.add(spec.key)
        out.append(spec)
    return out


def dedupe(targets: Iterable[TargetSpec]) -> list[TargetSpec]:
    """按 ``key`` 去重并保持原有顺序。"""
    out: list[TargetSpec] = []
    seen: set[str] = set()
    for spec in targets:
        if spec is None or spec.key in seen:
            continue
        seen.add(spec.key)
        out.append(spec)
    return out


def parse_targets_text(text: Any) -> tuple[list[TargetSpec], list[str]]:
    """解析纯文本目标清单。

    每行一个目标，格式为 ``群号 QQ号 [备注昵称]``，分隔符可用空格、逗号、
    顿号、竖线或分号；以 ``#`` / ``//`` 开头的行视为注释整行忽略。

    Args:
        text: 多行文本。

    Returns:
        ``(targets, warnings)``。``warnings`` 逐行说明哪一行为什么被跳过，
        便于用户在群里/日志里定位自己填错的那一行。
    """
    targets: list[TargetSpec] = []
    warnings: list[str] = []
    raw_text = str(text or "")
    if not raw_text.strip():
        return targets, warnings

    for lineno, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(_COMMENT_PREFIXES):
            continue

        parts = _SPLIT_RE.split(stripped, maxsplit=2)
        # 备注昵称可能自带空格，所以第三段直接用剩余整串，不再继续切
        if len(parts) < 2:
            warnings.append(
                f"第 {lineno} 行「{stripped}」格式不对：至少要有「群号 QQ号」两段。"
            )
            continue

        spec = make_target(parts[0], parts[1], parts[2] if len(parts) > 2 else "")
        if spec is None:
            warnings.append(
                f"第 {lineno} 行「{stripped}」的群号或 QQ 号不合法"
                f"（须为 {MIN_ID_LEN}~{MAX_ID_LEN} 位纯数字）。"
            )
            continue
        targets.append(spec)

    return dedupe(targets), warnings


def collect_config_targets(config: Any) -> tuple[list[TargetSpec], list[str]]:
    """从插件配置里汇总全部目标（三条来源合并 + 去重）。

    Args:
        config: AstrBot 配置对象（dict 语义）。

    Returns:
        ``(targets, warnings)``。配置问对象可能为 None，此时返回空列表。
    """
    if not isinstance(config, dict):
        return [], []

    out: list[TargetSpec] = []

    # ① 可视化清单（template_list）
    out.extend(normalize_targets(config.get("targets")))

    # ② 文本清单
    text_targets, warnings = parse_targets_text(config.get("targets_text"))
    out.extend(text_targets)

    # ③ 遗留扁平字段
    legacy = make_target(
        config.get("target_group_id"),
        config.get("target_qq_id"),
        config.get("target_nickname", ""),
    )
    if legacy is not None:
        out.append(legacy)

    return dedupe(out), warnings


def targets_to_json(targets: Iterable[TargetSpec]) -> str:
    """把目标列表序列化为 JSON 字符串（供写入 state 表）。"""
    payload = [spec.to_dict() for spec in dedupe(targets)]
    return json.dumps(payload, ensure_ascii=False)


def targets_from_json(raw: Any) -> list[TargetSpec]:
    """从 state 表里的 JSON 字符串还原目标列表；坏数据返回空列表。"""
    if not raw:
        return []
    try:
        data = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    return normalize_targets(data)
