"""Named task groups for targeted TaskSpec and sweep validation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_VERIFIED_SPECS_DIR = (
    Path(__file__).resolve().parents[1] / "tasks" / "specs" / "verified"
)

# Batch 1 groups intentionally overlap. They are meant for systematic
# validation by behavior/fixture family rather than for partitioning the batch.
BATCH1_TASK_GROUPS: dict[str, tuple[str, ...]] = {
    "blender_mixer": (
        "fillblenderjug",
        "makebananamilkshake",
        "blendvegetablesauce",
        "blendsalsamix",
        "blendmarinade",
        "makechocolatemilk",
        "makecheesecakefilling",
        "prepareveggiedip",
        "cheesemixing",
        "colorfulsalsa",
    ),
    "bowls": (
        "bowlandcup",
        "setupbowls",
        "setbowlsforsoup",
        "portionyogurt",
        "portionhotdogs",
        "veggiedipprep",
        "arrangebreadbowl",
    ),
    "trays": (
        "dessertassembly",
        "displaymeatvariety",
        "veggiedipprep",
    ),
    "plates": (
        "addsugarcubes",
        "portionhotdogs",
        "hotdogsetup",
        "portionyogurt",
        "setupbutterplate",
        "platestoredinner",
        "preparesausagecheese",
        "servesteak",
    ),
    "garnishes": (
        "garnishcake",
        "garnishpancake",
        "garnishcupcake",
        "addlemontofish",
        "lemonseasoningfish",
    ),
    "beverages_drinkware": (
        "placebeveragestogether",
        "preparecoffee",
        "sweetencoffee",
        "matchcupanddrink",
        "preparecocktailstation",
        "preparedrinkstation",
        "setupsodabowl",
        "servemealjuice",
        "alcoholservingprep",
        "arrangedrinkware",
        "setupwineglasses",
        "deliverstraw",
        "beverageorganization",
        "datenight",
        "arrangetea",
    ),
    "condiments_spices": (
        "lineupcondiments",
        "organizecondiments",
        "condimentcollection",
        "setupspicestation",
        "gathermarinadeingredients",
        "spicymarinade",
        "seasoningsteak",
        "lemonseasoningfish",
    ),
    "toaster_heating": (
        "arrangebreadbowl",
        "heatkebabsandwich",
        "toastheatableingredients",
        "kettleboiling",
    ),
    "sink_cleaning": (
        "candlecleanup",
        "clearreceptaclesforcleaning",
        "cleanboard",
        "washlettuce",
        "fillblenderjug",
    ),
    "meat_pan_serving": (
        "displaymeatvariety",
        "meatskewerassembly",
        "distributechicken",
        "seasoningsteak",
        "servesteak",
        "addlemontofish",
        "lemonseasoningfish",
        "preparesausagecheese",
    ),
    "buffet_station_setup": (
        "cutbuffetpizza",
        "tongbuffetsetup",
        "preparecheesestation",
        "preparesandwichstation",
        "preparesoupserving",
    ),
    "table_silverware": (
        "alignsilverware",
        "setupbowls",
        "setbowlsforsoup",
    ),
    "clearing_clustering": (
        "clusteritemsforclearing",
        "clearreceptaclesforcleaning",
        "gathercuttingtools",
    ),
}


BATCH2_TASK_GROUPS: dict[str, tuple[str, ...]] = {
    "ice_cold_drinks": (
        "makeicelemonade",
        "placeequalicecubes",
        "placeiceincup",
        "retrieveicetray",
        "addicecubes",
        "placestraw",
    ),
    "boiling_kettle_water": (
        "boilcorn",
        "boileggs",
        "boilpot",
        "coolkettle",
        "fillkettle",
        "heatmultiplewater",
        "placelidtoboil",
        "startelectrickettle",
    ),
    "broiling_fish": (
        "ovenbroilfish",
        "preparebroilingstation",
        "removebroiledfish",
        "toasterovenbroilfish",
        "washfish",
        "steamfish",
    ),
    "pan_stove_cooking": (
        "defrostbycategory",
        "distributesteakonpans",
        "fryingpanadjustment",
        "presschicken",
        "setupfrying",
        "butteronpan",
        "placevegetablesevenly",
        "stirvegetables",
        "turnoffsimmeredsauceheat",
        "beginslowcooking",
        "stopslowcooking",
        "simmeringsauce",
    ),
    "tea_hot_drinks": (
        "arrangeteaaccompaniments",
        "servetea",
        "strainersetup",
        "addmarshmallow",
        "sweetenhotchocolate",
        "heatmug",
    ),
    "toaster_reheat": (
        "makeloadedpotato",
        "wafflereheat",
        "warmcroissant",
        "gettoastedbread",
        "pjsandwichprep",
        "servewarmcroissant",
        "toastbagel",
        "toastbaguette",
        "toastoneslotpair",
    ),
    "sink_sanitizing": (
        "cleanblenderjug",
        "rinsesinkbasin",
        "rinsecuttingboard",
        "sanitizeprepcuttingboard",
        "scrubcuttingboard",
        "arrangesinksanitization",
        "cleanmicrowave",
        "prepforsanitizing",
        "sanitizesink",
        "wipetable",
    ),
    "dishwashing_cleanup": (
        "collectwashingsupplies",
        "drydrinkware",
        "placedishesbysink",
        "placeondishrack",
        "prerinsestation",
        "presoakpan",
        "returnwashingsupplies",
        "rinsebowls",
        "rinsefragileitem",
        "scrubbowl",
        "soaksponge",
        "sortingcleanup",
        "stackbowlsinsink",
        "transportcookware",
    ),
}


BATCH3_TASK_GROUPS: dict[str, tuple[str, ...]] = {
    "table_cabinet_reset": (
        "gathertableware",
        "resetcabinetdoors",
        "stackcans",
    ),
    "baking_prep_cleanup": (
        "cookiedoughprep",
        "coolbakedcake",
        "coolbakedcookies",
        "cupcakecleanup",
        "mixcakefrosting",
        "organizebakingingredients",
        "pastrydisplay",
    ),
    "dishwasher_dishrack": (
        "loaddishwasher",
        "preparedishwasher",
        "emptydishrack",
    ),
    "fridge_freezer": (
        "loadcondimentsinfridge",
        "rearrangefridgeitems",
        "breadselection",
        "freezebottledwaters",
        "freezeicetray",
        "freezecookedfood",
    ),
    "mugs_bowls_storage": (
        "organizemugsbyhandle",
        "stackbowlscabinet",
        "restockbowls",
    ),
    "recycling_sorting": (
        "recyclebottlesbysize",
        "recyclesodacans",
        "recyclestackedyogurt",
        "snacksorting",
    ),
    "utensil_drawers": (
        "arrangeutensilsbytype",
        "clusterutensilsindrawer",
        "organizemetallicutensils",
        "drawerutensilsort",
        "utensilshuffle",
    ),
    "restocking_supplies": (
        "restockcannedfood",
        "restocksinksupplies",
        "organizecleaningsupplies",
    ),
}


TASK_GROUPS_BY_BATCH: dict[str, dict[str, tuple[str, ...]]] = {
    "batch1": BATCH1_TASK_GROUPS,
    "batch2": BATCH2_TASK_GROUPS,
    "batch3": BATCH3_TASK_GROUPS,
}


def tasks_for_group(group_name: str, *, batch: str = "batch1") -> tuple[str, ...]:
    """Return task slugs for one named validation group."""

    try:
        batch_groups = TASK_GROUPS_BY_BATCH[batch]
    except KeyError as exc:
        known_batches = ", ".join(sorted(TASK_GROUPS_BY_BATCH))
        raise KeyError(f"Unknown task batch {batch!r}; known: {known_batches}") from exc

    try:
        return batch_groups[group_name]
    except KeyError as exc:
        known_groups = ", ".join(sorted(batch_groups))
        raise KeyError(
            f"Unknown {batch} task group {group_name!r}; known: {known_groups}"
        ) from exc


def tasks_for_groups(
    group_names: str | list[str] | tuple[str, ...],
    *,
    batch: str = "batch1",
) -> tuple[str, ...]:
    """Return de-duplicated task slugs for multiple validation groups."""

    if isinstance(group_names, str):
        names = tuple(
            group_name
            for group_name in group_names.replace(",", " ").split()
            if group_name
        )
    else:
        names = tuple(str(group_name) for group_name in group_names)

    ordered_tasks: list[str] = []
    seen: set[str] = set()
    for group_name in names:
        for task in tasks_for_group(group_name, batch=batch):
            if task in seen:
                continue
            seen.add(task)
            ordered_tasks.append(task)
    return tuple(ordered_tasks)


def copy_group_task_specs(
    *,
    group_name: str,
    src_dir: str | Path,
    dst_dir: str | Path,
    batch: str = "batch1",
    overwrite: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Copy TaskSpec JSON files for one group from src_dir to dst_dir.

    Returns a summary dict with `copied`, `skipped_existing`, and `missing`.
    """

    source_root = Path(src_dir).expanduser().resolve()
    destination_root = Path(dst_dir).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    skipped_existing: list[str] = []
    missing: list[str] = []

    for task_name in tasks_for_group(group_name, batch=batch):
        source_path = source_root / f"{task_name}.json"
        destination_path = destination_root / source_path.name
        if not source_path.exists():
            missing.append(task_name)
            continue
        if destination_path.exists() and not overwrite:
            skipped_existing.append(task_name)
            continue
        shutil.copy2(source_path, destination_path)
        copied.append(task_name)

    return {
        "copied": tuple(copied),
        "skipped_existing": tuple(skipped_existing),
        "missing": tuple(missing),
    }


def count_unique_task_specs(
    specs_dir: str | Path = DEFAULT_VERIFIED_SPECS_DIR,
) -> dict[str, object]:
    """Count unique TaskSpecs in a directory tree, accounting for duplicates.

    Uniqueness is determined by JSON filename stem (task slug), so duplicated
    specs across group folders are counted once in `unique_task_count`.
    """

    root = Path(specs_dir).expanduser().resolve()
    spec_files = sorted(path for path in root.rglob("*.json") if path.is_file())
    files_by_task: dict[str, list[str]] = {}
    for spec_path in spec_files:
        files_by_task.setdefault(spec_path.stem, []).append(
            str(spec_path.relative_to(root))
        )

    duplicates = {
        task_name: tuple(sorted(relative_paths))
        for task_name, relative_paths in files_by_task.items()
        if len(relative_paths) > 1
    }

    return {
        "specs_dir": str(root),
        "file_count": len(spec_files),
        "unique_task_count": len(files_by_task),
        "duplicate_task_count": len(duplicates),
        "duplicates": duplicates,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task-group utilities")
    subparsers = parser.add_subparsers(dest="command")

    copy_parser = subparsers.add_parser(
        "copy-specs",
        help="Copy TaskSpec files for one named group from source to destination.",
    )
    copy_parser.add_argument("--group", required=True, help="Task group name (e.g. bowls)")
    copy_parser.add_argument("--batch", default="batch1", help="Batch name (default: batch1)")
    copy_parser.add_argument("--src", required=True, help="Source directory containing *.json specs")
    copy_parser.add_argument("--dst", required=True, help="Destination directory")
    copy_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )

    count_parser = subparsers.add_parser(
        "count-unique-specs",
        help="Count unique TaskSpecs and duplicates across a directory tree.",
    )
    count_parser.add_argument(
        "--specs-dir",
        default=str(DEFAULT_VERIFIED_SPECS_DIR),
        help=(
            "Directory to scan recursively for TaskSpec JSON files "
            f"(default: {DEFAULT_VERIFIED_SPECS_DIR})."
        ),
    )
    return parser


def _main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.command == "copy-specs":
        summary = copy_group_task_specs(
            group_name=args.group,
            src_dir=args.src,
            dst_dir=args.dst,
            batch=args.batch,
            overwrite=bool(args.overwrite),
        )
        print(f"Copied: {len(summary['copied'])}")
        print(f"Skipped existing: {len(summary['skipped_existing'])}")
        print(f"Missing in source: {len(summary['missing'])}")
        if summary["missing"]:
            print("Missing tasks:", ", ".join(summary["missing"]))
        return 0

    if args.command == "count-unique-specs":
        summary = count_unique_task_specs(args.specs_dir)
        print(f"Scanned files: {summary['file_count']}")
        print(f"Unique task specs: {summary['unique_task_count']}")
        print(f"Duplicate task slugs: {summary['duplicate_task_count']}")
        duplicates = summary["duplicates"]
        if isinstance(duplicates, dict) and duplicates:
            print("Duplicates:")
            for task_name in sorted(duplicates):
                locations = ", ".join(duplicates[task_name])
                print(f"  - {task_name}: {locations}")
        return 0

    if args.command is None:
        parser.print_help()
        return 1
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
