#!/usr/bin/env python3
"""
make_split_mappings.py

Writes the official OpenCVL train/test assignment for every image, as JSON
mapping files the WebDataset conversion pipeline can read.

How the split is decided (from OpenCVL's release_v1.json):
    mapillary/cities/...            -> "train"
    mapillary/in_the_wild_test/...  -> "test_in_the_wild"
The folder a labels.json lives in decides the split, not the record contents.

This script does NOT re-implement that rule. It calls OpenCVL's own
SplitAssigner, so the output always matches their official file.

Usage:
    python3 make_split_mappings.py \
        --opencvl-repo /path/to/opencvl \
        --dataset-root /gpfs3tmp/pi/swshah/CVGL/OpenCVL \
        --output ./split_mappings

Optional:
    --test-label test     (rename "test_in_the_wild" to "test" in the output)

Output:
    split_mappings/cities/<country>/<city>.json
    split_mappings/in_the_wild_test/<country>/<city or all>.json
        each file: { "<mapil_image_id>": "train", ... }
    split_mappings/summary.json
        counts per split and per city, plus a check against the paper's totals
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


# Totals from the OpenCVL README / paper (Mapillary portion only).
# train: 341,540 Mapillary city pairs. test_in_the_wild: 1,361 pairs.
EXPECTED_COUNTS = {
    "train": 341540,
    "test_in_the_wild": 1361,
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Export official OpenCVL split mappings.")

    parser.add_argument("--opencvl-repo", type=Path, required=True,
                        help="Folder of the cloned OpenCVL GitHub repo (contains opencvl_tools/).")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="OpenCVL dataset folder (contains mapillary/).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Folder to write the mapping files into.")
    parser.add_argument("--test-label", default=None,
                        help="Optional name to use instead of 'test_in_the_wild'.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Make OpenCVL's own code importable, then import the official split tools.
    sys.path.insert(0, str(args.opencvl_repo.resolve()))
    from opencvl_tools.dataset import iter_samples
    from opencvl_tools.splits import SplitAssigner

    assigner = SplitAssigner()

    # mappings[(subset, country, city)] = { image_id: split }
    mappings = defaultdict(dict)

    # Where each image ID was first seen, to catch duplicates / leakage.
    first_seen = {}
    duplicates = []

    split_counts = Counter()

    for sample in iter_samples(args.dataset_root, resolve_paths=False):

        # Only the Mapillary part matters for this dataset.
        if sample.dataset != "mapillary":
            continue

        split = assigner.assign(sample)

        if split is None:
            raise ValueError(
                f"No official split for {sample.label_file.relative_path} "
                f"record {sample.index}"
            )

        if split == "test_in_the_wild" and args.test_label:
            split = args.test_label

        image_id = sample.sample_id
        location = f"{sample.split}/{sample.country}/{sample.group}"

        if image_id in first_seen:
            duplicates.append(
                {"image_id": image_id, "first": first_seen[image_id], "again": location}
            )
        else:
            first_seen[image_id] = location

        key = (sample.split, sample.country or "unknown", sample.group or "all")
        mappings[key][image_id] = split
        split_counts[split] += 1

    # Write one JSON file per (subset, country, city).
    per_city_counts = {}

    for (subset, country, city), mapping in sorted(mappings.items()):
        out_file = args.output / subset / country / f"{city}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with out_file.open("w", encoding="utf-8") as handle:
            json.dump(mapping, handle, indent=1, sort_keys=True)

        per_city_counts[f"{subset}/{country}/{city}"] = len(mapping)

    # Compare totals against the paper (using the official split names).
    test_name = args.test_label or "test_in_the_wild"
    check = {
        "train": {
            "found": split_counts.get("train", 0),
            "expected": EXPECTED_COUNTS["train"],
        },
        test_name: {
            "found": split_counts.get(test_name, 0),
            "expected": EXPECTED_COUNTS["test_in_the_wild"],
        },
    }
    for entry in check.values():
        entry["matches"] = entry["found"] == entry["expected"]

    summary = {
        "split_counts": dict(split_counts),
        "check_against_paper": check,
        "duplicate_image_ids": len(duplicates),
        "duplicate_examples": duplicates[:20],
        "per_city_counts": per_city_counts,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Short printout for the terminal.
    print("Split counts:", dict(split_counts))
    for name, entry in check.items():
        status = "OK" if entry["matches"] else "MISMATCH"
        print(f"  {name}: {entry['found']} (expected {entry['expected']}) {status}")
    print("Duplicate image IDs:", len(duplicates))
    print("Files written to:", args.output.resolve())


if __name__ == "__main__":
    main()
