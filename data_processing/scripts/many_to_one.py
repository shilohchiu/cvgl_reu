"""Build the many-grounds-to-one-aerial mapping from provider crops.

Each labels entry comes with its own aerial crop. Treating every crop center as
an anchor, the script shares one crop among the nearby grounds it covers,
densest cluster first. The result is written as aerial_to_grounds.json keyed by
the same `aerial_<lat>_<lon>` IDs the sharding script reconstructs, so each
shared crop is sharded once.
"""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer, CRS

MAPPING_FILENAME = "aerial_to_grounds.json"
COUNTS_FILENAME = "aerial_ground_counts.parquet"

# Cap the ground x anchor distance matrix used by the nearest-anchor fallback.
MAX_DISTANCE_MATRIX_ELEMENTS = 20_000_000


def load_labels(path):
    """Read labels.json and return its list of entries."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("labels", [])
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of labels or a dict with a 'labels' list")
    return data


def get_utm_crs_by_zone(utm_zone, is_north=True):
    """WGS84 UTM CRS for a zone and hemisphere."""
    hemisphere = "north" if is_north else "south"
    return CRS.from_string(
        f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
    )


def get_utm_zone(lat, lon):
    """UTM zone and hemisphere for a point."""
    return int((lon + 180) // 6 + 1), lat >= 0


def get_utm_transformer(lat, lon):
    """WGS84 -> UTM transformer for the zone containing lat/lon."""
    utm_crs = get_utm_crs_by_zone(*get_utm_zone(lat, lon))
    return Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)


def recover_patch_center(ground_lat, ground_lon, offset_east_m, offset_north_m):
    """Patch center = ground point - offset (offsets are ground minus center)."""
    transformer = get_utm_transformer(ground_lat, ground_lon)
    gx, gy = transformer.transform(ground_lon, ground_lat)
    center_lon, center_lat = transformer.transform(
        gx - offset_east_m, gy - offset_north_m, direction="INVERSE"
    )
    return center_lat, center_lon


def make_aerial_id(lat, lon):
    """Aerial key encoding the patch center, matching the sharding script."""
    return f"aerial_{lat:.7f}_{lon:.7f}"


def label_entry_offsets(entry):
    """Return (lat, lon, offset_east_m, offset_north_m) for a labels entry.

    Raw Mapillary pose is preferred; OpenSfM is the fallback.
    """
    resolution = entry.get("resolution")
    raw_latlon = entry.get("ground_latlon_mapilraw") or entry.get("ground_latlon")
    raw_dx = entry.get("dx_mapilraw")
    raw_dy = entry.get("dy_mapilraw")
    if raw_latlon and raw_dx is not None and raw_dy is not None and resolution is not None:
        return raw_latlon[0], raw_latlon[1], raw_dx * resolution, raw_dy * resolution

    of_latlon = entry.get("ground_latlon_mapilopensfm")
    of_dx = entry.get("dx_mapilopensfm")
    of_dy = entry.get("dy_mapilopensfm")
    if of_latlon and of_dx is not None and of_dy is not None and resolution is not None:
        return of_latlon[0], of_latlon[1], of_dx * resolution, of_dy * resolution

    if raw_latlon:
        return raw_latlon[0], raw_latlon[1], None, None
    return None


def label_entry_center(entry):
    """Recovered patch center; falls back to ground position without offsets."""
    pose = label_entry_offsets(entry)
    if pose is None:
        return None
    lat, lon, offset_east_m, offset_north_m = pose
    if offset_east_m is None or offset_north_m is None:
        return lat, lon
    return recover_patch_center(lat, lon, offset_east_m, offset_north_m)


def anchor_pairs(labels, radius_m, aerial_dir=None):
    """Share the provider's one-crop-per-ground anchors among nearby grounds.

    Anchors are used densest-cluster first, so each existing crop can serve
    several grounds. `radius_m` caps the ground-to-center distance and, when
    `aerial_dir` is given, is tightened to the crop's own half-extent. A ground
    left outside every sharing radius falls back to its own crop, and a ground
    without a crop of its own goes to the nearest anchor, so every ground with
    a usable pose is matched.
    """
    stats = Counter()
    anchors = {}      # aerial_id -> anchor record
    ground_rows = []  # (ground_id, lat, lon), in labels order
    own_anchor = []   # each ground's own crop, aligned with ground_rows
    seen_grounds = set()

    for entry in labels:
        ground_id = entry.get("mapil_image_id")
        if ground_id is None:
            stats["missing_ground_id"] += 1
            continue
        ground_id = str(ground_id)

        pose = label_entry_offsets(entry)
        center = label_entry_center(entry)
        aerial_id = None
        if center is not None and "aerial_image" in entry:
            aerial_id = make_aerial_id(center[0], center[1])
        if pose is None:
            stats["missing_pose"] += 1

        if pose is not None and ground_id not in seen_grounds:
            seen_grounds.add(ground_id)
            ground_rows.append((ground_id, pose[0], pose[1]))
            own_anchor.append(aerial_id)

        if aerial_id is None:
            stats["missing_patch_center"] += 1
            continue
        if aerial_id in anchors:
            continue

        width = height = None
        if aerial_dir is not None:
            image_path = Path(aerial_dir) / Path(entry["aerial_image"]).name
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except (OSError, ValueError):
                stats["missing_aerial_image"] += 1

        anchors[aerial_id] = {
            "lat": center[0],
            "lon": center[1],
            "resolution": entry.get("resolution"),
            "width": width,
            "height": height,
        }

    if not ground_rows:
        return {}, {}, stats
    if not anchors:
        raise ValueError(
            "no aerial patches could be derived from labels (each needs an "
            "aerial_image and a recoverable patch center); every ground "
            "image would be unmatched"
        )

    lats = [row[1] for row in ground_rows] + [a["lat"] for a in anchors.values()]
    lons = [row[2] for row in ground_rows] + [a["lon"] for a in anchors.values()]
    transformer = get_utm_transformer(statistics.mean(lats), statistics.mean(lons))
    gx, gy = transformer.transform(
        [row[2] for row in ground_rows], [row[1] for row in ground_rows]
    )
    anchor_ids = list(anchors)
    ax, ay = transformer.transform(
        [anchors[a]["lon"] for a in anchor_ids],
        [anchors[a]["lat"] for a in anchor_ids],
    )
    gx, gy, ax, ay = map(np.asarray, (gx, gy, ax, ay))

    # Each anchor keeps only the grounds its own crop can actually show.
    pad_m = 2.0
    radii = []
    capped = 0
    for aerial_id in anchor_ids:
        anchor = anchors[aerial_id]
        resolution = anchor["resolution"]
        if anchor["width"] and anchor["height"] and resolution:
            half_east = anchor["width"] * resolution / 2.0
            half_north = anchor["height"] * resolution / 2.0
            radius = min(radius_m, half_east, half_north) - pad_m
            capped += 1
        else:
            radius = radius_m
        radii.append(max(0.0, radius))
    radii = np.asarray(radii)
    stats["anchors_capped_to_image"] = capped

    # Spatial buckets avoid a full ground x anchor distance matrix.
    cell = max(float(radius_m), 1.0)
    buckets = defaultdict(list)
    for index, (x, y) in enumerate(zip(gx, gy)):
        buckets[(math.floor(x / cell), math.floor(y / cell))].append(index)

    def nearby(anchor_index, radius):
        bucket_x = math.floor(ax[anchor_index] / cell)
        bucket_y = math.floor(ay[anchor_index] / cell)
        members = []
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for ground_index in buckets.get(
                    (bucket_x + offset_x, bucket_y + offset_y), ()
                ):
                    distance = math.hypot(
                        gx[ground_index] - ax[anchor_index],
                        gy[ground_index] - ay[anchor_index],
                    )
                    if distance <= radius:
                        members.append(ground_index)
        return members

    coverage = np.array(
        [len(nearby(i, radii[i])) for i in range(len(anchor_ids))]
    )
    # Densest clusters first; aerial_id breaks ties deterministically.
    order = sorted(
        range(len(anchor_ids)), key=lambda i: (-coverage[i], anchor_ids[i])
    )

    assigned = np.zeros(len(ground_rows), dtype=bool)
    aerial_to_grounds = {}
    centers = {}
    for anchor_index in order:
        if coverage[anchor_index] == 0:
            continue
        members = sorted(
            j for j in nearby(anchor_index, radii[anchor_index])
            if not assigned[j]
        )
        if not members:
            continue
        aerial_id = anchor_ids[anchor_index]
        assigned[members] = True
        aerial_to_grounds[aerial_id] = [ground_rows[j][0] for j in members]
        centers[aerial_id] = (anchors[aerial_id]["lat"], anchors[aerial_id]["lon"])

    # A ground's own crop contains it by construction, so leftovers (offset
    # beyond every sharing radius) keep their own crop instead of dropping out.
    fallback = 0
    for j in np.flatnonzero(~assigned):
        aerial_id = own_anchor[j]
        if aerial_id is None:
            continue
        aerial_to_grounds.setdefault(aerial_id, []).append(ground_rows[j][0])
        centers.setdefault(aerial_id, (anchors[aerial_id]["lat"], anchors[aerial_id]["lon"]))
        assigned[j] = True
        fallback += 1
    stats["matched_to_own_crop"] = fallback

    # Grounds without a provider crop of their own still need an aerial, so
    # send them to the nearest anchor center.
    leftovers = [j for j in np.flatnonzero(~assigned) if own_anchor[j] is None]
    if leftovers:
        step = max(1, min(2048, MAX_DISTANCE_MATRIX_ELEMENTS // len(anchor_ids)))
        for start in range(0, len(leftovers), step):
            chunk = leftovers[start:start + step]
            dx = gx[chunk, None] - ax[None, :]
            dy = gy[chunk, None] - ay[None, :]
            nearest = (dx * dx + dy * dy).argmin(axis=1)
            for offset, anchor_index in enumerate(nearest):
                j = chunk[offset]
                aerial_id = anchor_ids[int(anchor_index)]
                aerial_to_grounds.setdefault(aerial_id, []).append(ground_rows[j][0])
                centers.setdefault(
                    aerial_id, (anchors[aerial_id]["lat"], anchors[aerial_id]["lon"])
                )
                assigned[j] = True
        stats["matched_to_nearest_anchor"] = len(leftovers)

    remaining = int((~assigned).sum())
    if remaining:
        raise ValueError(
            f"{remaining} ground image(s) could not be matched to any aerial"
        )

    stats["unmatched"] = remaining
    stats["anchor_candidates"] = len(anchor_ids)
    return aerial_to_grounds, centers, stats


def write_outputs(aerial_to_grounds, centers, output_dir, min_grounds=1):
    """Write the mapping JSON and counts parquet; drop patches below min_grounds.

    Returns (mapping_path, counts_path, mapping, dropped_grounds).
    """
    mapping = {}
    dropped_grounds = 0
    for aerial_id, ground_ids in aerial_to_grounds.items():
        # Preserve input order for stable diffs.
        ground_ids = list(dict.fromkeys(ground_ids))
        if len(ground_ids) < min_grounds:
            dropped_grounds += len(ground_ids)
            continue
        mapping[aerial_id] = ground_ids

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = output_dir / MAPPING_FILENAME
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
        f.write("\n")

    rows = []
    for aerial_id, ground_ids in mapping.items():
        row = {"aerial_id": aerial_id, "ground_count": len(ground_ids)}
        if aerial_id in centers:
            row["lat"], row["lon"] = centers[aerial_id]
        rows.append(row)
    counts_path = output_dir / COUNTS_FILENAME
    pd.DataFrame(rows, columns=["aerial_id", "ground_count", "lat", "lon"]).to_parquet(
        counts_path, index=False
    )

    return mapping_path, counts_path, mapping, dropped_grounds


def print_stats(mapping, stats):
    """Print a compact many-to-one summary."""
    counts = [len(ground_ids) for ground_ids in mapping.values()]
    non_empty = [count for count in counts if count]
    total_grounds = sum(counts)
    print(f"Satellite images with at least one ground image: {len(non_empty)}")
    if len(non_empty) != len(counts):
        print(f"Satellite images with no ground images: {len(counts) - len(non_empty)}")
    print(f"Ground images associated: {total_grounds}")
    if stats.get("labels_total"):
        print(
            f"Ground images with an aerial: {stats['labels_matched']}/"
            f"{stats['labels_total']}"
        )
        if stats.get("labels_missing"):
            print(f"Ground images without an aerial: {stats['labels_missing']}")
    if non_empty:
        print(
            "Grounds per satellite image: "
            f"min={min(non_empty)} median={statistics.median(non_empty):g} "
            f"mean={statistics.mean(non_empty):.2f} max={max(non_empty)}"
        )
        print(f"Satellite images with a single ground image: {sum(1 for c in non_empty if c == 1)}")

    if stats.get("anchor_candidates"):
        print(
            f"Anchor candidates: {stats['anchor_candidates']} "
            f"({stats.get('anchors_capped_to_image', 0)} capped to their crop extent)"
        )
    if stats.get("matched_to_own_crop"):
        print(f"Ground images matched to their own crop: {stats['matched_to_own_crop']}")
    if stats.get("matched_to_nearest_anchor"):
        print(
            "Ground images matched to the nearest anchor (no own crop): "
            f"{stats['matched_to_nearest_anchor']}"
        )
    for key, label in (
        ("missing_ground_id", "labels without mapil_image_id"),
        ("missing_pose", "labels without a usable pose (cannot be matched)"),
        ("missing_patch_center", "labels without a recoverable patch center"),
        ("missing_aerial_image", "missing aerial image files"),
    ):
        if stats.get(key):
            print(f"Skipped {stats[key]} {label}.")


def build_mapping(args, city_root=None):
    """Build and write the association for one city; returns the mapping path."""
    labels_path = args.labels_path
    aerial_dir = args.aerial_dir
    output_dir = args.output_dir

    if city_root is not None:
        # Country-wide mode ignores CLI paths so they cannot leak across cities.
        city_root = Path(city_root)
        labels_path = str(city_root / "labels.json")
        aerial_dir = str(city_root / "aerial")
        output_dir = str(city_root / "aerial")

    labels = load_labels(labels_path)
    mapping, centers, stats = anchor_pairs(
        labels, args.anchor_radius_m, aerial_dir=aerial_dir
    )

    mapping_path, counts_path, mapping, dropped = write_outputs(
        mapping, centers, output_dir, min_grounds=args.min_grounds,
    )

    label_ids = {
        str(entry["mapil_image_id"]) for entry in labels
        if entry.get("mapil_image_id") is not None
    }
    matched_ids = {gid for ground_ids in mapping.values() for gid in ground_ids}
    stats["labels_total"] = len(label_ids)
    stats["labels_matched"] = len(matched_ids)
    stats["labels_missing"] = len(label_ids - matched_ids)

    print(f"Associated {sum(len(v) for v in mapping.values())} ground images "
          f"with {len(mapping)} satellite images.")
    print_stats(mapping, stats)
    if dropped:
        print(
            f"Dropped {dropped} ground image(s) whose satellite patch had "
            f"fewer than {args.min_grounds} ground image(s)."
        )
    print(f"Mapping: {mapping_path}")
    print(f"Counts:  {counts_path}")
    return mapping_path


def main():
    parser = argparse.ArgumentParser(
        description="Associate many ground-level images with one satellite/aerial image"
    )
    parser.add_argument("--labels_path", help="Path to labels.json")
    parser.add_argument("--aerial_dir",
                        help="Aerial crop directory, used to cap each anchor to its crop extent")
    parser.add_argument("--output_dir", help="Directory for the mapping and counts")
    parser.add_argument("--anchor_radius_m", type=float, default=40.0,
                        help="Maximum ground-to-patch-center distance in meters "
                             "(default: 40); each anchor is further capped by its "
                             "own crop extent")
    parser.add_argument("--min_grounds", type=int, default=1,
                        help="Only keep aerial patches with at least this many grounds")
    parser.add_argument("--country_dir",
                        help="Process every city folder containing labels.json")
    args = parser.parse_args()

    if args.country_dir:
        country_path = Path(args.country_dir)
        cities = [
            d for d in sorted(country_path.iterdir())
            if d.is_dir() and (d / "labels.json").exists()
        ]
        if not cities:
            raise SystemExit(f"No city folders with labels.json under {args.country_dir}")
        print(f"Found {len(cities)} cities in {args.country_dir}: {[c.name for c in cities]}")
        for city_dir in cities:
            print(f"\n--- Building associations for {city_dir.name} ---")
            build_mapping(args, city_root=city_dir)
        return

    if not args.labels_path:
        parser.error("--labels_path is required (or use --country_dir)")
    if not args.aerial_dir:
        parser.error("--aerial_dir is required (or use --country_dir)")
    if not args.output_dir:
        parser.error("--output_dir is required (or use --country_dir)")
    if args.anchor_radius_m <= 0:
        parser.error("--anchor_radius_m must be positive")
    if args.min_grounds < 1:
        parser.error("--min_grounds must be at least 1")

    build_mapping(args)


if __name__ == "__main__":
    main()
