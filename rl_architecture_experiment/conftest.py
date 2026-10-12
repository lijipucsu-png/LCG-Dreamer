from __future__ import annotations

from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_TESTS = EXPERIMENT_ROOT / "tests"


def pytest_ignore_collect(collection_path: Path, config: object) -> bool:
    """Keep the copied suite out of pytest runs launched from the parent project."""
    root_path = Path(str(getattr(config, "rootpath"))).resolve()
    if root_path == EXPERIMENT_ROOT:
        return False
    candidate = Path(str(collection_path)).resolve()
    return candidate == EXPERIMENT_TESTS or EXPERIMENT_TESTS in candidate.parents
