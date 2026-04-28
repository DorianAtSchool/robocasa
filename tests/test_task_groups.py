from pathlib import Path

from data_generation.task_level.pipeline.task_groups import (
    count_unique_task_specs,
    copy_group_task_specs,
    groups_for_task,
    tasks_for_group,
)
from data_generation.task_level.tasks import supported_task_names


def test_batch1_bowls_group_includes_hotdog_portioning_overlap():
    assert "portionhotdogs" in tasks_for_group("bowls", batch="batch1")
    assert "portionhotdogs" in tasks_for_group("plates", batch="batch1")


def test_groups_for_task_returns_all_overlapping_batch1_groups():
    assert groups_for_task("portionhotdogs", batch="batch1") == ("bowls", "plates")


def test_copy_group_task_specs_copies_present_and_reports_missing(tmp_path: Path):
    source = tmp_path / "src"
    destination = tmp_path / "dst"
    source.mkdir()
    destination.mkdir()

    # Only seed a subset of the bowls group to exercise missing reporting.
    for task_name in ("bowlandcup", "setupbowls"):
        (source / f"{task_name}.json").write_text('{"ok": true}', encoding="utf-8")

    summary = copy_group_task_specs(
        group_name="bowls",
        batch="batch1",
        src_dir=source,
        dst_dir=destination,
    )

    assert set(summary["copied"]) == {"bowlandcup", "setupbowls"}
    assert set(summary["missing"]) >= {"setbowlsforsoup", "portionyogurt"}
    assert (destination / "bowlandcup.json").exists()
    assert (destination / "setupbowls.json").exists()


def test_copy_group_task_specs_respects_overwrite_flag(tmp_path: Path):
    source = tmp_path / "src"
    destination = tmp_path / "dst"
    source.mkdir()
    destination.mkdir()

    (source / "bowlandcup.json").write_text('{"version": 2}', encoding="utf-8")
    existing_path = destination / "bowlandcup.json"
    existing_path.write_text('{"version": 1}', encoding="utf-8")

    no_overwrite = copy_group_task_specs(
        group_name="bowls",
        batch="batch1",
        src_dir=source,
        dst_dir=destination,
        overwrite=False,
    )
    assert "bowlandcup" in no_overwrite["skipped_existing"]
    assert existing_path.read_text(encoding="utf-8") == '{"version": 1}'

    with_overwrite = copy_group_task_specs(
        group_name="bowls",
        batch="batch1",
        src_dir=source,
        dst_dir=destination,
        overwrite=True,
    )
    assert "bowlandcup" in with_overwrite["copied"]
    assert existing_path.read_text(encoding="utf-8") == '{"version": 2}'


def test_count_unique_task_specs_counts_duplicates_by_task_slug(tmp_path: Path):
    bowls_dir = tmp_path / "bowls"
    plates_dir = tmp_path / "plates"
    bowls_dir.mkdir()
    plates_dir.mkdir()

    (bowls_dir / "portionhotdogs.json").write_text("{}", encoding="utf-8")
    (plates_dir / "portionhotdogs.json").write_text("{}", encoding="utf-8")
    (bowls_dir / "bowlandcup.json").write_text("{}", encoding="utf-8")
    (plates_dir / "setupbutterplate.json").write_text("{}", encoding="utf-8")

    summary = count_unique_task_specs(tmp_path)

    assert summary["file_count"] == 4
    assert summary["unique_task_count"] == 3
    assert summary["duplicate_task_count"] == 1
    duplicates = summary["duplicates"]
    assert isinstance(duplicates, dict)
    assert duplicates["portionhotdogs"] == (
        "bowls/portionhotdogs.json",
        "plates/portionhotdogs.json",
    )


def test_supported_task_names_defaults_to_canonical_verified_inventory():
    supported = supported_task_names()

    assert "HotDogSetup" in supported
    assert "PrepareCoffee" in supported
