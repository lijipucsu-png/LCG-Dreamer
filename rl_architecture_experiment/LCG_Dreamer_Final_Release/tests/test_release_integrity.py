from pathlib import Path

from final_model.runtime import ReleasePaths
from final_model.verify import verify_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_critical_file_hashes() -> None:
    checks = verify_manifest(ROOT)
    assert checks
    assert all(row["exists"] and row["sha256_matches"] for row in checks)


def test_runtime_assets_are_self_contained() -> None:
    paths = ReleasePaths.discover(ROOT)
    paths.validate(include_lane_change=True)
    assert paths.longitudinal.is_relative_to(ROOT)
    assert paths.lateral.is_relative_to(ROOT)
    assert paths.predictors.is_relative_to(ROOT)
