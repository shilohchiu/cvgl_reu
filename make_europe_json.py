#!/usr/bin/env python3
"""
make_europe_json.py

Builds the Europe train.json and test.json Yuqi asked for, starting from the
official split files that OpenCVL's own create_splits.py wrote.

Which pairs go where (Yuqi, Oct 24):
    train.json = every Mapillary city pair from the official "train" split
    test.json  = every pair in the official "test_cross_area" split (all ZOD)

Each record in the output has:
    ground_id        unique ID of the ground photo
    lat, lon         GPS position of the ground photo
    heading          camera direction in degrees, 0 = up on the aerial image
    aerial_id        ID of the aerial patch this photo is matched to
    offset_east_m    how far the camera is east of the patch center (meters)
    offset_north_m   how far the camera is north of the patch center (meters)
plus a few helper fields (split, source, country, group, image paths,
resolution, position_source) so the result can be checked and traced.

This version is ONE-TO-ONE: each photo is matched to its own OpenCVL patch.
The many-to-one step will replace aerial_id and the two offsets later.

How the offset is computed:
    OpenCVL stores the camera's position on its patch as dx, dy in pixels
    from the patch center (dx > 0 = east, dy > 0 = north, in the map grid).
    offset in meters = pixels x resolution (meters per pixel).

Usage:
    python3 make_europe_json.py \
        --dataset-root /gpfs3tmp/pi/swshah/CVGL/OpenCVL \
        --splits-dir /gpfs3tmp/pi/swshah/mherman5/europe/official_splits \
        --output /gpfs3tmp/pi/swshah/mherman5/europe \
        --position raw
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath


# Official counts from the OpenCVL paper (Section 3.2) for what we keep.
EXPECTED_TRAIN = 341540   # Mapillary pairs inside the official train split
EXPECTED_TEST = 18504     # official cross-area test split


def parse_arguments():
    parser = argparse.ArgumentParser(description="Build Europe train.json and test.json.")

    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="OpenCVL dataset folder (contains mapillary/ and zod/).")
    parser.add_argument("--splits-dir", type=Path, required=True,
                        help="Folder with the official train.jsonl, test_cross_area.jsonl, etc.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Folder to write train.json, test.json and europe_summary.json into.")
    parser.add_argument("--position", choices=["raw", "opensfm"], default="raw",
                        help="Which Mapillary position to use (ZOD always uses its own).")

    return parser.parse_args()


# ----------------------------------------------------------------------
# Reading the official split files and the labels they point to
# ----------------------------------------------------------------------

def read_jsonl(path):
    """Read a file with one JSON object per line."""

    entries = []

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    return entries


def load_label_records(path):
    """Read a labels.json and return its list of records."""

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload

    for key in ("labels", "images", "samples", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]

    raise ValueError(f"Unexpected labels.json layout: {path}")


class LabelCache:
    """Keeps each labels.json in memory after the first read, since the
    official split files point into the same files thousands of times."""

    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.files = {}

    def record(self, label_file, index):
        if label_file not in self.files:
            self.files[label_file] = load_label_records(self.dataset_root / label_file)

        return self.files[label_file][index]


# ----------------------------------------------------------------------
# Turning one OpenCVL record into one output record
# ----------------------------------------------------------------------

def mapillary_pose(record, position):
    """Pick position, heading and pixel offset for a Mapillary record.

    Rule: position, heading and dx/dy always come from the SAME source,
    never mixed. If OpenSfM is requested but missing, fall back to raw
    for that record and say so in position_source.
    """

    if position == "opensfm" and record.get("pose_mapilopensfm_available"):
        return {
            "latlon": record["ground_latlon_mapilopensfm"],
            "heading": record["heading_mapilopensfm"],
            "dx": record["dx_mapilopensfm"],
            "dy": record["dy_mapilopensfm"],
            "position_source": "mapillary_opensfm",
        }

    source = "mapillary_raw" if position == "raw" else "mapillary_raw_fallback"

    return {
        "latlon": record["ground_latlon_mapilraw"],
        "heading": record["heading_mapilraw"],
        "dx": record["dx_mapilraw"],
        "dy": record["dy_mapilraw"],
        "position_source": source,
    }


def zod_pose(record):
    """ZOD has one precise position, so there is nothing to choose."""

    return {
        "latlon": record["ground_latlon"],
        "heading": record["heading"],
        "dx": record["dx"],
        "dy": record["dy"],
        "position_source": "zod",
    }


def build_record(entry, record, position):
    """Combine the official split entry with its labels.json record."""

    if entry["dataset"] == "mapillary":
        pose = mapillary_pose(record, position)
        ground_path = record["mapil_image"]
    else:
        pose = zod_pose(record)
        ground_path = record["ground_image"]

    aerial_path = record["aerial_image"]
    resolution = record["resolution"]

    # Image paths in labels.json are relative to the labels.json folder for
    # Mapillary, and relative to zod/ for ZOD. Store them relative to the
    # OpenCVL root so they can be opened the same way for both.
    label_folder = PurePosixPath(entry["label_file"]).parent
    if entry["dataset"] == "mapillary":
        ground_rel = str(label_folder / ground_path)
        aerial_rel = str(label_folder / aerial_path)
    else:
        ground_rel = str(PurePosixPath("zod") / ground_path)
        aerial_rel = str(PurePosixPath("zod") / aerial_path)

    lat, lon = pose["latlon"]

    return {
        "ground_id": entry["sample_id"],
        "lat": lat,
        "lon": lon,
        "heading": pose["heading"],
        "aerial_id": PurePosixPath(aerial_path).stem,
        "offset_east_m": round(pose["dx"] * resolution, 3),
        "offset_north_m": round(pose["dy"] * resolution, 3),
        "split": entry["official_split"],
        "source": entry["dataset"],
        "country": entry["country"],
        "group": entry["group"],
        "ground_image": ground_rel,
        "aerial_image": aerial_rel,
        "resolution": resolution,
        "position_source": pose["position_source"],
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_arguments()
    cache = LabelCache(args.dataset_root)

    # train.json: only the Mapillary part of the official train split
    train_entries = [
        entry for entry in read_jsonl(args.splits_dir / "train.jsonl")
        if entry["dataset"] == "mapillary"
    ]

    # test.json: the whole official cross-area split
    test_entries = read_jsonl(args.splits_dir / "test_cross_area.jsonl")

    train = [
        build_record(entry, cache.record(entry["label_file"], entry["label_index"]), args.position)
        for entry in train_entries
    ]
    test = [
        build_record(entry, cache.record(entry["label_file"], entry["label_index"]), args.position)
        for entry in test_entries
    ]

    # Checks: counts, unique IDs, and no photo in both files
    train_ids = Counter(r["ground_id"] for r in train)
    test_ids = Counter(r["ground_id"] for r in test)

    checks = {
        "train_count": {"found": len(train), "expected": EXPECTED_TRAIN},
        "test_count": {"found": len(test), "expected": EXPECTED_TEST},
    }
    for entry in checks.values():
        entry["matches"] = entry["found"] == entry["expected"]

    summary = {
        "position_setting": args.position,
        "checks": checks,
        "duplicate_ids_in_train": sum(1 for n in train_ids.values() if n > 1),
        "duplicate_ids_in_test": sum(1 for n in test_ids.values() if n > 1),
        "ids_in_both_train_and_test": len(set(train_ids) & set(test_ids)),
        "position_sources": {
            "train": dict(Counter(r["position_source"] for r in train)),
            "test": dict(Counter(r["position_source"] for r in test)),
        },
        "test_by_subset_country": dict(Counter(
            f"{e['subset']}/{e['country']}" for e in test_entries
        )),
    }

    args.output.mkdir(parents=True, exist_ok=True)

    with (args.output / "train.json").open("w", encoding="utf-8") as handle:
        json.dump(train, handle)
    with (args.output / "test.json").open("w", encoding="utf-8") as handle:
        json.dump(test, handle)
    with (args.output / "europe_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))

    all_good = (
        all(entry["matches"] for entry in checks.values())
        and summary["duplicate_ids_in_train"] == 0
        and summary["duplicate_ids_in_test"] == 0
        and summary["ids_in_both_train_and_test"] == 0
    )

    if not all_good:
        print("CHECK FAILED: see the summary above before using these files.")
        sys.exit(1)

    print("All checks passed. Files written to:", args.output.resolve())


if __name__ == "__main__":
    main()
