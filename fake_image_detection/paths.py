from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

_THIS_FILE = Path(__file__).resolve()
_DATA_MARKERS = ("README.md", "data/image_test", "image_test.csv")
# 优先使用用户指定的 sem_image 目录，再兼容原项目路径
_DATA_ROOT_CANDIDATES = (
    "dateset/sem_image",
    "dateset/image_data",
    "dataset/dataset/image_data",
    "dataset/image_data",
    "image_data",
)


def _looks_like_data_root(p: Path) -> bool:
    return all((p / m).exists() for m in _DATA_MARKERS)


def _looks_like_project_root(p: Path) -> bool:
    return any((_looks_like_data_root(p / c) for c in _DATA_ROOT_CANDIDATES))


def resolve_code_root() -> Path:
    env = os.environ.get("CODE_ROOT")
    if env:
        cp = Path(env).resolve()
        if cp.is_dir():
            return cp
    return _THIS_FILE.parent.parent


def resolve_project_root() -> Path:
    env = os.environ.get("PROJECT_ROOT")
    if env:
        pp = Path(env).resolve()
        if pp.is_dir():
            return pp
    return resolve_code_root().parent


def resolve_data_root() -> Path:
    env = os.environ.get("DATA_ROOT")
    if env:
        dp = Path(env).resolve()
        if _looks_like_data_root(dp):
            return dp
        if dp.is_dir():
            return dp
    project = resolve_project_root()
    for c in _DATA_ROOT_CANDIDATES:
        cand = project / c
        if _looks_like_data_root(cand):
            return cand
    for parent in [project, *project.parents]:
        for c in _DATA_ROOT_CANDIDATES:
            cand = parent / c
            if _looks_like_data_root(cand):
                return cand
            # 兼容从原项目拆出的情况：数据在原项目 Detector/dateset/... 下
            cand_in_detector = parent / "Detector" / c
            if _looks_like_data_root(cand_in_detector):
                return cand_in_detector
    raise FileNotFoundError(
        "Could not auto-locate DATA_ROOT. Set DATA_ROOT env var. "
        "Expected a directory containing README.md, data/image_test and image_test.csv."
    )


def resolve_path(*parts: str, base: Optional[Path] = None) -> Path:
    root = base if base is not None else resolve_code_root()
    p = Path(parts[0])
    if p.is_absolute():
        return p.resolve()
    return (root / Path(*parts)).resolve()


def ensure_output_dirs() -> None:
    cr = resolve_code_root()
    for d in (
        "artifacts",
        "outputs",
        "outputs/checkpoints",
        "outputs/logs",
        "outputs/oof",
        "outputs/predictions",
        "outputs/submissions",
    ):
        (cr / d).mkdir(parents=True, exist_ok=True)


def sample_csv() -> Path:
    return resolve_data_root() / "image_sample_data.csv"


def test_csv() -> Path:
    return resolve_data_root() / "image_test.csv"


def submission_example_csv() -> Path:
    return resolve_data_root() / "image_submission_example.csv"


def sample_image_dir() -> Path:
    return resolve_data_root() / "data" / "image_sample_data"


def test_image_dir() -> Path:
    return resolve_data_root() / "data" / "image_test"
