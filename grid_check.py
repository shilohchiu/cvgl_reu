#!/usr/bin/env python3
"""
grid_check.py

Finds which map grid each country's aerial patches are drawn in.

Why: an aerial patch's "up" is its map grid's north, not true north, so
offsets in meters must be computed in that same grid. ZOD records give
both the camera position (ground_latlon) and the patch center
(aerial_latlon), plus the camera's pixel offset (dx, dy). The right grid
is the one where (camera - center) in meters equals (dx, dy) x resolution.

For each ZOD labels.json and each candidate grid, this prints the average
error in meters over a sample of records. The right grid has an error
close to 0 (pixel rounding alone is a few centimeters).

Usage:
    python grid_check.py /gpfs3tmp/pi/swshah/CVGL/OpenCVL
"""

import json
import sys
from pathlib import Path

from pyproj import Transformer


# Candidate grids per country (EPSG codes), each with a short name.
CANDIDATES = {
    "SE": {3006: "SWEREF99 TM", 32633: "UTM 33N", 32632: "UTM 32N"},
    "NL": {28992: "RD New", 32631: "UTM 31N", 32632: "UTM 32N"},
    "NO": {25832: "EUREF89 UTM 32N", 25833: "EUREF89 UTM 33N", 25835: "EUREF89 UTM 35N"},
    "PL": {2180: "PL-1992", 2177: "PL-2000 zone 6", 2178: "PL-2000 zone 7", 32633: "UTM 33N", 32634: "UTM 34N"},
}

SAMPLE_SIZE = 500


def load_records(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload

    for key in ("labels", "images", "samples", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]

    raise ValueError(f"Unexpected labels.json layout: {path}")


def mean_error(records, transformer):
    """Average distance (m) between the grid offset and the pixel offset."""

    total = 0.0

    for record in records:
        ground_lat, ground_lon = record["ground_latlon"]
        center_lat, center_lon = record["aerial_latlon"]

        ground_x, ground_y = transformer.transform(ground_lon, ground_lat)
        center_x, center_y = transformer.transform(center_lon, center_lat)

        grid_east = ground_x - center_x
        grid_north = ground_y - center_y

        pixel_east = record["dx"] * record["resolution"]
        pixel_north = record["dy"] * record["resolution"]

        total += ((grid_east - pixel_east) ** 2 + (grid_north - pixel_north) ** 2) ** 0.5

    return total / len(records)


def main():
    root = Path(sys.argv[1])

    for labels_path in sorted((root / "zod").rglob("labels.json")):
        country = labels_path.parent.name
        records = load_records(labels_path)

        # Spread the sample across the whole file, not just its start.
        step = max(1, len(records) // SAMPLE_SIZE)
        sample = records[::step][:SAMPLE_SIZE]

        print(f"\n{labels_path.relative_to(root)}  ({len(records)} records, checked {len(sample)})")

        for epsg, name in CANDIDATES.get(country, {}).items():
            transformer = Transformer.from_crs(4326, epsg, always_xy=True)
            error = mean_error(sample, transformer)
            print(f"  {name:<18} EPSG:{epsg:<6} average error {error:8.3f} m")


if __name__ == "__main__":
    main()
