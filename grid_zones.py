#!/usr/bin/env python3
"""
grid_zones.py

Follow-up to grid_check.py for the Netherlands and Poland, where no single
grid matched a whole file.

Hypothesis: in these countries each patch is drawn in the UTM zone of its
own location. UTM zones are 6-degree-wide strips of longitude:
    zone 31: 0 to 6 E      zone 32: 6 to 12 E
    zone 33: 12 to 18 E    zone 34: 18 to 24 E

For every ZOD record in NL and PL, this finds which candidate grid fits
best, then counts the winners by 1-degree band of longitude. If the
hypothesis holds, the winner switches exactly at 6 E (NL) and 18 E (PL).

Usage:
    python grid_zones.py /gpfs3tmp/pi/swshah/CVGL/OpenCVL
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pyproj import Transformer


CANDIDATES = {
    "NL": {28992: "RD New", 25831: "UTM 31N", 25832: "UTM 32N"},
    "PL": {2180: "PL-1992", 2176: "PL-2000 z5", 2177: "PL-2000 z6",
           2178: "PL-2000 z7", 2179: "PL-2000 z8", 25833: "UTM 33N", 25834: "UTM 34N"},
}


def load_records(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload

    for key in ("labels", "images", "samples", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]

    raise ValueError(f"Unexpected labels.json layout: {path}")


def error_for(record, transformer):
    ground_lat, ground_lon = record["ground_latlon"]
    center_lat, center_lon = record["aerial_latlon"]

    ground_x, ground_y = transformer.transform(ground_lon, ground_lat)
    center_x, center_y = transformer.transform(center_lon, center_lat)

    pixel_east = record["dx"] * record["resolution"]
    pixel_north = record["dy"] * record["resolution"]

    return (((ground_x - center_x) - pixel_east) ** 2
            + ((ground_y - center_y) - pixel_north) ** 2) ** 0.5


def main():
    root = Path(sys.argv[1])

    for country, candidates in CANDIDATES.items():
        transformers = {
            epsg: Transformer.from_crs(4326, epsg, always_xy=True)
            for epsg in candidates
        }

        # winners[longitude band][grid name] = how many records it won
        winners = defaultdict(Counter)
        best_errors = []

        for labels_path in sorted((root / "zod").rglob("labels.json")):
            if labels_path.parent.name != country:
                continue

            for record in load_records(labels_path):
                errors = {epsg: error_for(record, t) for epsg, t in transformers.items()}
                best = min(errors, key=errors.get)

                band = int(record["ground_latlon"][1])
                winners[band][candidates[best]] += 1
                best_errors.append(errors[best])

        if not best_errors:
            print(f"\n{country}: no ZOD records found")
            continue

        print(f"\n{country}: {len(best_errors)} records, "
              f"average best error {sum(best_errors) / len(best_errors):.3f} m, "
              f"worst {max(best_errors):.3f} m")

        for band in sorted(winners):
            counts = ", ".join(f"{name}: {n}" for name, n in winners[band].most_common())
            print(f"  {band}-{band + 1} E   {counts}")


if __name__ == "__main__":
    main()
