"""UTF-8 JSON 输出小工具，数据层与服务层共享。"""

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    """拒绝 NaN/Infinity，保证输出为可跨语言读取的标准 JSON。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
