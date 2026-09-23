"""为历史源码契约读取拆分后的前端实现，排除测试与声明文件。"""

from pathlib import Path

STUDIO_SOURCE = Path(__file__).resolve().parents[1] / "MFQStudio" / "src"


def read_studio_sources(*relative_paths: str) -> str:
    """按给定顺序读取文件或业务目录，供尚未迁移为行为测试的契约复用。"""
    sources = []
    for relative_path in relative_paths:
        location = STUDIO_SOURCE / relative_path
        paths = sorted(location.rglob("*.ts*")) if location.is_dir() else [location]
        sources.extend(
            path.read_text(encoding="utf-8")
            for path in paths
            if path.suffix in {".ts", ".tsx"}
            and not any(marker in path.name for marker in (".test.", ".spec.", ".d.ts"))
        )
    return "\n".join(sources)
