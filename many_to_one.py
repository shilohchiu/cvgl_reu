#!/usr/bin/env python3
"""
many_to_one.py   (DRAFT)

Turns the one-to-one Europe files (train.json, test.json) into many-to-one:
nearby photos share one OpenCVL aerial patch instead of each keeping its own.

THE RULE FOR SHARING
    Photo B may use photo A's patch if B's camera falls inside the middle
    40 m x 40 m of A's patch (at most 20 m east/west and 20 m north/south of
    A's center). That is the same +-20 m rule OpenCVL used when it made the
    patches, so the task stays exactly as hard as before.

HOW PATCHES ARE PICKED (greedy)
    1. For every patch, list which photos it could hold.
    2. Pick the patch that holds the most photos that don't have a patch yet.
    3. Give those photos that patch. Repeat until every photo has one.
    Every photo can always use its own patch, so this always finishes.

WHERE THE METERS COME FROM (the map-grid rule found with grid_check.py and
grid_zones.py on ZOD, where every grid fits to pixel rounding)
    Sweden       -> SWEREF99 TM (EPSG:3006)
    Norway       -> UTM zone 33 (EPSG:25833)
    Netherlands  -> UTM zone of the photo's longitude: 31 west of 6 E, 32 east
    Poland       -> UTM zone of the photo's longitude: 33 west of 18 E, 34 east
    Offsets are computed in the patch's own grid, because the patch's "up"
    is that grid's north, not true north.

SAFETY LIMITS
    - Train and test are matched separately, so no patch is shared across them.
    - Mapillary photos only share within their own city; ZOD within its country.
    - Two photos in different grids (the rare photos right at a zone border)
      never share, so headings never need converting between grids.

Usage:
    python many_to_one.py \
        --input /gpfs3tmp/pi/swshah/mherman5/europe \
        --output /gpfs3tmp/pi/swshah/mherman5/europe/many_to_one
"""

import argparse
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from pyproj import Transformer


HALF_BOX_M = 20.0   # a camera must be within +-20 m of its patch center


# ----------------------------------------------------------------------
# Map grids
# ----------------------------------------------------------------------

def grid_for(country, lon):
    """EPSG code of the grid a patch at this location is drawn in."""

    if country == "SE":
        return 3006
    if country == "NO":
        return 25833
    if country == "NL":
        return 25831 if lon < 6.0 else 25832
    if country == "PL":
        return 25833 if lon < 18.0 else 25834

    raise ValueError(f"No grid rule for country {country}")


_transformers = {}


def to_grid(epsg, lat, lon):
    """Latitude/longitude -> (east, north) in meters in the given grid."""

    if epsg not in _transformers:
        _transformers[epsg] = Transformer.from_crs(4326, epsg, always_xy=True)

    return _transformers[epsg].transform(lon, lat)


# ----------------------------------------------------------------------
# Matching one group of photos (one city, or ZOD in one country)
# ----------------------------------------------------------------------

def match_group(records):
    """Assign every photo in this group to a patch. Returns a list of
    (anchor index, member indices) pairs, one per patch kept."""

    count = len(records)

    epsg = []        # each photo's grid
    position = []    # each photo's camera position (east, north) in its grid
    center = []      # each photo's own patch center in that same grid

    for r in records:
        code = grid_for(r["country"], r["lon"])
        east, north = to_grid(code, r["lat"], r["lon"])

        epsg.append(code)
        position.append((east, north))
        center.append((east - r["offset_east_m"], north - r["offset_north_m"]))

    # Put cameras into 40 m buckets, so finding nearby photos only looks
    # at the bucket and its 8 neighbors instead of the whole city.
    bucket_size = 2 * HALF_BOX_M
    buckets = defaultdict(list)

    for i, (east, north) in enumerate(position):
        buckets[(epsg[i], math.floor(east / bucket_size), math.floor(north / bucket_size))].append(i)

    # Step 1: which photos could each patch hold?
    could_hold = []

    for a in range(count):
        cx, cy = center[a]
        bx, by = math.floor(cx / bucket_size), math.floor(cy / bucket_size)
        members = []

        for nx in (bx - 1, bx, bx + 1):
            for ny in (by - 1, by, by + 1):
                for b in buckets.get((epsg[a], nx, ny), []):
                    east, north = position[b]
                    if abs(east - cx) <= HALF_BOX_M and abs(north - cy) <= HALF_BOX_M:
                        members.append(b)

        could_hold.append(members)

    # Steps 2-3: greedy, using a heap so the biggest patch comes out first.
    # Counts go stale as photos get assigned, so each popped patch is
    # recounted and pushed back if it has shrunk ("lazy" greedy).
    assigned = [False] * count
    heap = [(-len(members), a) for a, members in enumerate(could_hold)]
    heapq.heapify(heap)
    kept = []

    while heap:
        negative_size, a = heapq.heappop(heap)
        fresh = [b for b in could_hold[a] if not assigned[b]]

        if not fresh:
            continue
        if len(fresh) < -negative_size:
            heapq.heappush(heap, (-len(fresh), a))
            continue

        for b in fresh:
            assigned[b] = True
        kept.append((a, fresh))

    assert all(assigned), "every photo should have a patch"

    return kept, position, center, epsg


# ----------------------------------------------------------------------
# Building the output for one split
# ----------------------------------------------------------------------

def group_key(r):
    if r["source"] == "mapillary":
        return ("mapillary", r["country"], r["group"])
    return ("zod", r["country"])


def convert(records):
    groups = defaultdict(list)
    for r in records:
        groups[group_key(r)].append(r)

    out_records = []
    aerials = []
    worst_offset = 0.0

    for key in sorted(groups):
        members_of_group = groups[key]
        kept, position, center, epsg = match_group(members_of_group)

        for a, members in kept:
            anchor = members_of_group[a]
            cx, cy = center[a]

            aerials.append({
                "aerial_id": anchor["aerial_id"],
                "aerial_image": anchor["aerial_image"],
                "resolution": anchor["resolution"],
                "grid_epsg": epsg[a],
                "center_east_m": round(cx, 3),
                "center_north_m": round(cy, 3),
                "ground_count": len(members),
                "ground_ids": [members_of_group[b]["ground_id"] for b in members],
            })

            for b in members:
                record = dict(members_of_group[b])
                east, north = position[b]

                record["aerial_id"] = anchor["aerial_id"]
                record["aerial_image"] = anchor["aerial_image"]
                record["resolution"] = anchor["resolution"]
                record["offset_east_m"] = round(east - cx, 3)
                record["offset_north_m"] = round(north - cy, 3)

                worst_offset = max(worst_offset, abs(record["offset_east_m"]), abs(record["offset_north_m"]))
                out_records.append(record)

    sizes = Counter(a["ground_count"] for a in aerials)

    summary = {
        "photos": len(out_records),
        "patches_before": len(records),
        "patches_after": len(aerials),
        "patches_saved_percent": round(100 * (1 - len(aerials) / len(records)), 1),
        "largest_shared_patch": max(sizes),
        "photos_per_patch_average": round(len(out_records) / len(aerials), 2),
        "patches_by_size": {str(k): sizes[k] for k in sorted(sizes)},
        "worst_offset_m": worst_offset,
    }

    return out_records, aerials, summary


def check_own_patch_offsets(original, converted):
    """Sanity check: a photo that kept its own patch must keep its offset
    (to within pixel rounding of the patch center, a few centimeters)."""

    by_id = {r["ground_id"]: r for r in original}
    worst = 0.0
    checked = 0

    for r in converted:
        before = by_id[r["ground_id"]]
        if r["aerial_id"] == before["aerial_id"]:
            checked += 1
            worst = max(worst,
                        abs(r["offset_east_m"] - before["offset_east_m"]),
                        abs(r["offset_north_m"] - before["offset_north_m"]))

    return {"photos_on_own_patch": checked, "largest_change_m": round(worst, 4)}


def main():
    parser = argparse.ArgumentParser(description="Many-to-one aerial matching for the Europe files (draft).")
    parser.add_argument("--input", type=Path, required=True, help="Folder with train.json and test.json.")
    parser.add_argument("--output", type=Path, required=True, help="Folder for the many-to-one files.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    report = {}

    for split in ("train", "test"):
        with open(args.input / f"{split}.json", encoding="utf-8") as handle:
            original = json.load(handle)

        converted, aerials, summary = convert(original)
        summary["own_patch_check"] = check_own_patch_offsets(original, converted)
        report[split] = summary

        with open(args.output / f"{split}.json", "w", encoding="utf-8") as handle:
            json.dump(converted, handle)
        with open(args.output / f"aerial_{split}.json", "w", encoding="utf-8") as handle:
            json.dump(aerials, handle)

    with open(args.output / "many_to_one_summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print("Files written to:", args.output.resolve())


if __name__ == "__main__":
    main()
