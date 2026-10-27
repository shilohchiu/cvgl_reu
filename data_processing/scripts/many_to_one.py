#!/usr/bin/env python3
"""
Build many-grounds-to-one-satellite associations.

The dataset's many-to-one mapping is a dictionary of the form

    {"aerial_ID_123": ["ground_ID_A", "ground_ID_B", ...], ...}

where each ground-level image belongs to exactly one satellite/aerial patch.
This script builds that mapping and writes it in the format already used by
`burlington_vt/aerial/aerial_to_grounds.json`.

Five association sources are supported:

  labels    Use the `aerial_image` field of labels.json, keying each patch by
            its recovered center (`aerial_<lat>_<lon>`) so the mapping matches
            the sharding script's aerial IDs.
  metadata  Use an existing parquet table with `ground_id` / `aerial_id`
            columns, e.g. `aerial/ground_metadata_linked.parquet`.
  anchors   Keep the provider's own patch centers as anchors and greedily
            assign each ground to an anchor whose crop actually covers it.
            This turns the provider's one-crop-per-ground pairing into a
            many-to-one mapping without needing any new imagery: nearby
            grounds share one existing crop. `--anchor_radius_m` caps the
            ground-to-center distance, and each anchor is tightened to its
            own crop half-extent when the aerial images are available.
  nearest   (Default) Assign every ground point to the nearest aerial patch
            center, optionally bounded by `--max_distance_m`. Use this when the
            source pairing is one-to-one but the satellite tiles are larger than
            the per-ground crops, so many ground views should share one tile.
  grid      Group ground points into a random-phase UTM grid, as in the
            reference pipeline: `grid_step = patch_size - 2*margin`, one
            cell per aerial, and the cell center becomes the aerial center.
            Aerial IDs encode that center as `aerial_<lat>_<lon>`. Use this to
            synthesize many-to-one associations from ground coordinates alone.

Outputs (written to --output_dir):
  aerial_to_grounds.json        the mapping, in first-seen order
  aerial_ground_counts.parquet  one row per aerial with its ground count and,
                                when known, the patch center lat/lon

Examples
--------
# Use the provider's pairing for one city
python3 scripts/many_to_one.py \
    --match labels \
    --labels_path NO/oslo/labels.json \
    --output_dir NO/oslo/aerial

# Many-to-one from the provider's own crops: share each crop with the
# grounds it covers (no new imagery needed)
python3 scripts/many_to_one.py \
    --match anchors \
    --labels_path NO/oslo/labels.json \
    --aerial_dir NO/oslo/aerial \
    --anchor_radius_m 40 \
    --output_dir NO/oslo/aerial

# Group ground images around existing satellite patches by distance
python3 scripts/many_to_one.py \
    --match nearest \
    --labels_path NO/oslo/labels.json \
    --aerial_metadata NO/oslo/aerial/aerial_metadata.parquet \
    --max_distance_m 100 \
    --output_dir NO/oslo/aerial

# Synthesize reference-style grid cells (240 m patch, 24 m margin, seed 42)
python3 scripts/many_to_one.py \
    --match grid \
    --labels_path NO/oslo/labels.json \
    --patch_size 240 --margin 24 --seed 42 \
    --output_dir NO/oslo/aerial

# Rebuild the mapping from an existing linked parquet
python3 scripts/many_to_one.py \
    --match metadata \
    --metadata_path NO/oslo/aerial/ground_metadata_linked.parquet \
    --output_dir NO/oslo/aerial

# Every city under a country directory
python3 scripts/many_to_one.py --country_dir NO/ --match labels
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

# Cap the ground x aerial distance matrix at roughly 20M float64 entries
# (~160 MB) when assigning nearest patches.
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
    """
    Construct the standard WGS84 UTM CRS for a zone and hemisphere.
    """
    hemisphere = "north" if is_north else "south"
    return CRS.from_string(
        f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
    )


def get_utm_zone(lat, lon):
    """
    UTM zone and hemisphere for a point, matching the reference pipeline.
    """
    return int((lon + 180) // 6 + 1), lat >= 0


def get_utm_transformer(lat, lon):
    """
    Return a WGS84 -> UTM transformer for the zone and hemisphere containing
    lat/lon, matching the reference pipeline's zone/hemisphere handling.
    """
    utm_crs = get_utm_crs_by_zone(*get_utm_zone(lat, lon))
    return Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)


def recover_patch_center(ground_lat, ground_lon, offset_east_m, offset_north_m):
    """
    Recover the aerial patch center from a ground point and its offset.

    Offsets are measured as ground minus center, so center = ground - offset.
    """
    transformer = get_utm_transformer(ground_lat, ground_lon)
    gx, gy = transformer.transform(ground_lon, ground_lat)
    center_lon, center_lat = transformer.transform(
        gx - offset_east_m, gy - offset_north_m, direction="INVERSE"
    )
    return center_lat, center_lon


def make_aerial_id(lat, lon):
    """
    Aerial key encoding the patch center, matching the sharding script.
    """
    return f"aerial_{lat:.7f}_{lon:.7f}"


def label_entry_offsets(entry):
    """
    Ground position and pixel-derived offset for a labels entry.

    Mirrors the sharding script: the raw Mapillary pose is used when its
    lat/lon and offsets are present, otherwise the OpenSfM pose. The offset is
    returned in meters, or None when the labels lack the needed fields.
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
    """
    Recovered patch center for a labels entry, matching the sharding script.

    Falls back to the ground position when the labels carry no usable offsets.
    """
    pose = label_entry_offsets(entry)
    if pose is None:
        return None
    lat, lon, offset_east_m, offset_north_m = pose
    if offset_east_m is None or offset_north_m is None:
        return lat, lon
    return recover_patch_center(lat, lon, offset_east_m, offset_north_m)


def add_pair(aerial_to_grounds, ground_to_aerial, aerial_id, ground_id):
    """
    Record one ground -> aerial association.

    A ground image may only belong to one satellite patch. Conflicting
    associations indicate broken input rather than a many-to-one relationship,
    so fail loudly.
    """
    previous = ground_to_aerial.get(ground_id)
    if previous is not None:
        if previous != aerial_id:
            raise ValueError(
                f"ground {ground_id} is associated with both {previous} and "
                f"{aerial_id}; each ground image must map to one aerial patch"
            )
        return False
    ground_to_aerial[ground_id] = aerial_id
    aerial_to_grounds.setdefault(aerial_id, []).append(ground_id)
    return True


def check_image(path):
    """True when an image path exists."""
    return path is not None and Path(path).is_file()


def pairs_from_labels(labels, ground_dir=None, aerial_dir=None):
    """
    Group labels by their provider `aerial_image` and key each patch by its
    recovered center (`aerial_<lat>_<lon>`), matching the sharding script.

    Returns (aerial_to_grounds, centers, stats). `stats` reports skipped rows
    and duplicate entries.
    """
    stats = Counter()
    provider_groups = defaultdict(list)

    for entry in labels:
        if "mapil_image_id" not in entry:
            stats["missing_ground_id"] += 1
            continue
        if "aerial_image" not in entry:
            stats["missing_aerial_field"] += 1
            continue

        if ground_dir is not None:
            ground_path = Path(ground_dir) / Path(entry.get("mapil_image", "")).name
            if not check_image(ground_path):
                stats["missing_ground_image"] += 1
                continue
        if aerial_dir is not None:
            aerial_path = Path(aerial_dir) / Path(entry["aerial_image"]).name
            if not check_image(aerial_path):
                stats["missing_aerial_image"] += 1
                continue

        provider_groups[Path(entry["aerial_image"]).stem].append(entry)

    aerial_to_grounds = {}
    ground_to_aerial = {}
    centers = {}

    for entries in provider_groups.values():
        center = label_entry_center(entries[0])
        if center is None:
            stats["missing_patch_center"] += 1
            continue

        aerial_id = make_aerial_id(center[0], center[1])
        centers.setdefault(aerial_id, center)
        for entry in entries:
            ground_id = str(entry["mapil_image_id"])
            if not add_pair(aerial_to_grounds, ground_to_aerial, aerial_id, ground_id):
                stats["duplicate_pair"] += 1

    return aerial_to_grounds, centers, stats


def pairs_from_metadata(metadata_path, ground_col="ground_id", aerial_col="aerial_id"):
    """
    Build the mapping from an existing ground<->aerial parquet table.

    Returns (aerial_to_grounds, centers, stats); centers are empty because the
    table does not need to carry coordinates.
    """
    frame = pd.read_parquet(metadata_path, columns=[ground_col, aerial_col])
    frame = frame.dropna(subset=[ground_col, aerial_col])
    if frame.empty:
        return {}, {}, Counter()

    conflicts = frame.groupby(ground_col)[aerial_col].nunique()
    conflicting = conflicts[conflicts > 1]
    if not conflicting.empty:
        example = conflicting.index[0]
        raise ValueError(
            f"{len(conflicting)} ground image(s) map to more than one aerial "
            f"patch in {metadata_path} (e.g. ground {example}); run with "
            f"--match nearest to rebuild a consistent assignment"
        )

    aerial_to_grounds = {}
    for aerial_id, group in frame.groupby(aerial_col, sort=False):
        aerial_to_grounds[str(aerial_id)] = [str(g) for g in group[ground_col].drop_duplicates()]

    stats = Counter()
    stats["duplicate_pair"] = len(frame) - frame[[ground_col, aerial_col]].drop_duplicates().shape[0]
    return aerial_to_grounds, {}, stats


def points_from_labels(labels):
    """
    Extract (ground_id, lat, lon) from labels.json, ignoring rows without GPS.
    """
    points = []
    seen = set()
    for entry in labels:
        ground_id = entry.get("mapil_image_id")
        latlon = entry.get("ground_latlon")
        if ground_id is None or latlon is None:
            continue
        ground_id = str(ground_id)
        if ground_id in seen:
            continue
        seen.add(ground_id)
        points.append((ground_id, float(latlon[0]), float(latlon[1])))
    return points


def points_from_parquet(path, id_col="ground_id", lat_col="lat", lon_col="lon"):
    """Extract (id, lat, lon) rows from a parquet table."""
    frame = pd.read_parquet(path, columns=[id_col, lat_col, lon_col])
    frame = frame.dropna(subset=[lat_col, lon_col])
    points = []
    seen = set()
    for row in frame.itertuples(index=False):
        point_id = str(getattr(row, id_col))
        if point_id in seen:
            continue
        seen.add(point_id)
        points.append((point_id, float(getattr(row, lat_col)), float(getattr(row, lon_col))))
    return points


def load_aerial_points(aerial_metadata_path):
    """Load (aerial_id, lat, lon) patch centers from aerial_metadata.parquet."""
    return points_from_parquet(
        aerial_metadata_path, id_col="aerial_id", lat_col="lat", lon_col="lon"
    )


def anchor_pairs(labels, radius_m, aerial_dir=None):
    """
    Build a many-to-one mapping from the provider's own aerial crops.

    The provider ships one crop per ground image, so `labels` mode is
    one-to-one. Here those patch centers become candidate anchors: anchors are
    visited densest-cluster first and kept whenever they still cover an
    unassigned ground, so nearby grounds share one existing crop. No new
    imagery is fetched or synthesized.

    `radius_m` caps the ground-to-anchor-center distance. When `aerial_dir` is
    available, each anchor's cap is also tightened to its own crop half-extent
    (pixels * resolution), so assigned grounds stay inside that image.
    Returns (aerial_to_grounds, centers, stats).
    """
    stats = Counter()
    anchors = {}      # aerial_id -> anchor record
    ground_rows = []  # (ground_id, lat, lon), in labels order
    seen_grounds = set()

    for entry in labels:
        ground_id = entry.get("mapil_image_id")
        if ground_id is None:
            stats["missing_ground_id"] += 1
            continue
        ground_id = str(ground_id)

        pose = label_entry_offsets(entry)
        if pose is not None and ground_id not in seen_grounds:
            seen_grounds.add(ground_id)
            ground_rows.append((ground_id, pose[0], pose[1]))

        center = label_entry_center(entry)
        if center is None or "aerial_image" not in entry:
            stats["missing_patch_center"] += 1
            continue

        aerial_id = make_aerial_id(center[0], center[1])
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

    if not anchors or not ground_rows:
        return {}, {}, stats

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

    stats["unmatched"] = int((~assigned).sum())
    stats["anchor_candidates"] = len(anchor_ids)
    return aerial_to_grounds, centers, stats


def nearest_pairs(ground_points, aerial_points, max_distance_m=None, chunk_size=2048):
    """
    Assign each ground point to its nearest aerial patch center.

    Distances use a city-scale UTM projection (derived from the mean position),
    which is accurate enough within a single city. The distance matrix is
    processed in chunks so large cities do not exhaust memory.

    Returns (aerial_to_grounds, centers, stats) where stats includes
    `unmatched` (grounds with no patch inside --max_distance_m).
    """
    if not aerial_points:
        raise ValueError("no aerial patch centers available for nearest matching")
    if not ground_points:
        return {}, {}, Counter(unmatched=0)

    all_lats = [p[1] for p in ground_points] + [p[1] for p in aerial_points]
    all_lons = [p[2] for p in ground_points] + [p[2] for p in aerial_points]
    transformer = get_utm_transformer(statistics.mean(all_lats), statistics.mean(all_lons))

    ground_ids = [p[0] for p in ground_points]
    gx, gy = transformer.transform(
        np.array([p[2] for p in ground_points]), np.array([p[1] for p in ground_points])
    )
    aerial_ids = [p[0] for p in aerial_points]
    ax, ay = transformer.transform(
        np.array([p[2] for p in aerial_points]), np.array([p[1] for p in aerial_points])
    )
    gx, gy, ax, ay = map(np.asarray, (gx, gy, ax, ay))

    aerial_to_grounds = {}
    ground_to_aerial = {}
    centers = {aid: (lat, lon) for aid, lat, lon in aerial_points}
    distances = {}
    stats = Counter()

    # Keep each chunk's distance matrix bounded.
    step = max(1, min(chunk_size, MAX_DISTANCE_MATRIX_ELEMENTS // len(aerial_ids)))
    for start in range(0, len(ground_ids), step):
        stop = min(start + step, len(ground_ids))
        dx = gx[start:stop, None] - ax[None, :]
        dy = gy[start:stop, None] - ay[None, :]
        dist_sq = dx * dx + dy * dy
        nearest = dist_sq.argmin(axis=1)
        nearest_dist = np.sqrt(dist_sq[np.arange(stop - start), nearest])

        for offset, aerial_index in enumerate(nearest):
            distance = float(nearest_dist[offset])
            if max_distance_m is not None and distance > max_distance_m:
                stats["unmatched"] += 1
                continue
            ground_id = ground_ids[start + offset]
            aerial_id = aerial_ids[int(aerial_index)]
            if add_pair(aerial_to_grounds, ground_to_aerial, aerial_id, ground_id):
                distances[ground_id] = distance

    if distances:
        values = list(distances.values())
        stats["distance_mean_m"] = statistics.mean(values)
        stats["distance_max_m"] = max(values)

    return aerial_to_grounds, centers, stats


def grid_pairs(ground_points, patch_size_m, margin_m, seed):
    """
    Group ground points into a random-phase UTM grid, as in the reference
    pipeline.

    Each grid cell becomes one aerial patch: `grid_step = patch_size - 2*margin`
    keeps every member at least `margin` inside the patch. A different grid
    phase is drawn per UTM zone from `seed`, and the cell center is the aerial
    center. Aerial IDs encode that center as `aerial_<lat>_<lon>`, matching the
    Burlington layout. Returns (aerial_to_grounds, centers, stats).
    """
    grid_step = patch_size_m - 2 * margin_m
    if grid_step <= 0:
        raise ValueError("patch_size must be strictly greater than 2 * margin")

    rng = np.random.RandomState(seed)

    # One grid phase and projection per UTM zone.
    groups = defaultdict(list)
    for ground_id, lat, lon in ground_points:
        groups[get_utm_zone(lat, lon)].append((ground_id, lat, lon))

    aerial_to_grounds = {}
    centers = {}
    stats = Counter()

    for (zone, is_north), points in sorted(groups.items()):
        utm_crs = get_utm_crs_by_zone(zone, is_north)
        to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        to_latlon = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

        grid_offset_x = rng.uniform(0, grid_step)
        grid_offset_y = rng.uniform(0, grid_step)

        cells = defaultdict(list)
        for ground_id, lat, lon in points:
            gx, gy = to_utm.transform(lon, lat)
            cx_idx = math.floor((gx - grid_offset_x) / grid_step)
            cy_idx = math.floor((gy - grid_offset_y) / grid_step)
            cells[(cx_idx, cy_idx)].append(ground_id)

        for (cx_idx, cy_idx), ground_ids in cells.items():
            cell_cx = (cx_idx + 0.5) * grid_step + grid_offset_x
            cell_cy = (cy_idx + 0.5) * grid_step + grid_offset_y
            cell_lon, cell_lat = to_latlon.transform(cell_cx, cell_cy)

            aerial_id = make_aerial_id(cell_lat, cell_lon)
            aerial_to_grounds[aerial_id] = ground_ids
            centers[aerial_id] = (cell_lat, cell_lon)
            stats["cells"] += 1

    return aerial_to_grounds, centers, stats


def write_outputs(aerial_to_grounds, centers, output_dir, min_grounds=1,
                  include_empty=False):
    """
    Write aerial_to_grounds.json and aerial_ground_counts.parquet.

    Returns (mapping_path, counts_path, mapping, dropped_grounds) where
    `mapping` is the filtered mapping that was written and dropped_grounds
    counts grounds dropped because their aerial had fewer than min_grounds
    members.
    """
    mapping = {}
    dropped_grounds = 0
    for aerial_id, ground_ids in aerial_to_grounds.items():
        # add_pair already deduplicated; keep the input order for stable diffs
        # against mappings generated by the original pipeline.
        ground_ids = list(dict.fromkeys(ground_ids))
        if len(ground_ids) < min_grounds:
            dropped_grounds += len(ground_ids)
            continue
        mapping[aerial_id] = ground_ids

    if include_empty:
        for aerial_id in centers:
            mapping.setdefault(aerial_id, [])

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
    if stats.get("unmatched"):
        print(f"Ground images with no satellite patch within range: {stats['unmatched']}")
    if stats.get("distance_mean_m") is not None:
        print(
            "Nearest-patch distance: "
            f"mean={stats['distance_mean_m']:.1f} m max={stats['distance_max_m']:.1f} m"
        )
    for key, label in (
        ("missing_ground_id", "labels without mapil_image_id"),
        ("missing_aerial_field", "labels without aerial_image"),
        ("missing_ground_image", "missing ground image files"),
        ("missing_aerial_image", "missing aerial image files"),
        ("duplicate_pair", "duplicate ground/aerial pairs"),
    ):
        if stats.get(key):
            print(f"Skipped {stats[key]} {label}.")


def build_mapping(args, city_root=None):
    """
    Build and write the association for one city.

    `city_root` is set only in country-wide mode and supplies default input
    paths. Returns the written mapping path, or None when a city was skipped.
    """
    match = args.match
    labels_path = args.labels_path
    metadata_path = args.metadata_path
    aerial_metadata_path = args.aerial_metadata
    ground_dir = args.ground_dir
    aerial_dir = args.aerial_dir
    output_dir = args.output_dir

    if city_root is not None:
        # Country-wide mode always uses each city's own inputs and outputs;
        # path arguments from the command line would otherwise leak across cities.
        city_root = Path(city_root)
        labels_path = str(city_root / "labels.json")
        metadata_path = str(city_root / "aerial" / "ground_metadata_linked.parquet")
        aerial_metadata_path = str(city_root / "aerial" / "aerial_metadata.parquet")
        ground_dir = str(city_root / "ground")
        aerial_dir = str(city_root / "aerial")
        output_dir = str(city_root / "aerial")

    check_images = args.check_images
    ground_dir_arg = ground_dir if check_images else None
    aerial_dir_arg = aerial_dir if check_images else None

    if match == "labels":
        labels = load_labels(labels_path)
        mapping, centers, stats = pairs_from_labels(
            labels, ground_dir=ground_dir_arg, aerial_dir=aerial_dir_arg
        )
    elif match == "anchors":
        labels = load_labels(labels_path)
        mapping, centers, stats = anchor_pairs(
            labels, args.anchor_radius_m, aerial_dir=aerial_dir
        )
    elif match == "metadata":
        mapping, centers, stats = pairs_from_metadata(
            metadata_path, args.ground_id_col, args.aerial_id_col
        )
    elif match == "nearest":
        if labels_path:
            ground_points = points_from_labels(load_labels(labels_path))
        else:
            ground_points = points_from_parquet(
                args.ground_metadata, args.ground_id_col, "lat", "lon"
            )

        aerial_points = load_aerial_points(aerial_metadata_path)
        if check_images:
            missing = 0
            existing = []
            for aerial_id, lat, lon in aerial_points:
                if (Path(aerial_dir) / f"{aerial_id}.png").exists():
                    existing.append((aerial_id, lat, lon))
                else:
                    missing += 1
            aerial_points = existing
            if missing:
                print(f"Skipped {missing} aerial patches without an image file.")

        mapping, centers, stats = nearest_pairs(
            ground_points, aerial_points, args.max_distance_m, args.chunk_size
        )
    elif match == "grid":
        if labels_path:
            ground_points = points_from_labels(load_labels(labels_path))
        else:
            ground_points = points_from_parquet(
                args.ground_metadata, args.ground_id_col, "lat", "lon"
            )

        print(
            f"Grid: patch_size={args.patch_size:g} m, margin={args.margin:g} m, "
            f"step={args.patch_size - 2 * args.margin:g} m, seed={args.seed}"
        )
        mapping, centers, stats = grid_pairs(
            ground_points, args.patch_size, args.margin, args.seed
        )
    else:
        raise ValueError(f"unknown --match value: {match}")

    mapping_path, counts_path, mapping, dropped = write_outputs(
        mapping, centers, output_dir,
        min_grounds=args.min_grounds, include_empty=args.include_empty,
    )

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
    parser.add_argument(
        "--match",
        choices=["labels", "metadata", "nearest", "anchors", "grid"],
        default="nearest",
        help="How to build the association (default: nearest)",
    )
    parser.add_argument("--labels_path", help="Path to labels.json")
    parser.add_argument("--metadata_path", help="Parquet with ground_id/aerial_id columns")
    parser.add_argument("--ground_metadata",
                        help="Parquet with ground_id/lat/lon, for --match nearest without labels")
    parser.add_argument("--aerial_metadata", help="aerial_metadata.parquet with patch centers")
    parser.add_argument("--output_dir", help="Directory for the mapping and counts")
    parser.add_argument("--ground_dir", help="Ground image directory, used with --check_images")
    parser.add_argument("--aerial_dir", help="Aerial image directory, used with --check_images")
    parser.add_argument("--check_images", action="store_true",
                        help="Skip associations whose image files are missing")
    parser.add_argument("--ground_id_col", default="ground_id",
                        help="Ground ID column for parquet inputs (default: ground_id)")
    parser.add_argument("--aerial_id_col", default="aerial_id",
                        help="Aerial ID column for parquet inputs (default: aerial_id)")
    parser.add_argument("--max_distance_m", type=float,
                        help="Nearest matching: maximum ground-to-patch distance in meters")
    parser.add_argument("--anchor_radius_m", type=float, default=40.0,
                        help="Anchors matching: maximum ground-to-patch-center "
                             "distance in meters (default: 40); each anchor is "
                             "further capped by its own crop extent")
    parser.add_argument("--chunk_size", type=int, default=2048,
                        help="Nearest matching: grounds per distance-matrix chunk")
    parser.add_argument("--patch_size", type=float, default=240.0,
                        help="Grid matching: aerial patch physical size in meters (default: 240)")
    parser.add_argument("--margin", type=float, default=24.0,
                        help="Grid matching: margin kept around each ground point in meters (default: 24)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Grid matching: seed for the per-zone grid phase (default: 42)")
    parser.add_argument("--min_grounds", type=int, default=1,
                        help="Only keep aerial patches with at least this many grounds")
    parser.add_argument("--include_empty", action="store_true",
                        help="Include aerial patches that received no ground images")
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
            if args.match == "nearest" and not (
                city_dir / "aerial" / "aerial_metadata.parquet"
            ).exists():
                print("Skipping: no aerial/aerial_metadata.parquet for nearest matching.")
                continue
            if args.match == "metadata" and not (
                city_dir / "aerial" / "ground_metadata_linked.parquet"
            ).exists():
                print("Skipping: no aerial/ground_metadata_linked.parquet for metadata matching.")
                continue
            build_mapping(args, city_root=city_dir)
        return

    if args.match == "labels" and not args.labels_path:
        parser.error("--match labels requires --labels_path")
    if args.match == "anchors":
        if not args.labels_path:
            parser.error("--match anchors requires --labels_path")
        if args.anchor_radius_m <= 0:
            parser.error("--anchor_radius_m must be positive")
    if args.match == "metadata" and not args.metadata_path:
        parser.error("--match metadata requires --metadata_path")
    if args.match == "nearest":
        if not args.aerial_metadata:
            parser.error("--match nearest requires --aerial_metadata")
        if not args.labels_path and not args.ground_metadata:
            parser.error("--match nearest requires --labels_path or --ground_metadata")
    if args.match == "grid":
        if not args.labels_path and not args.ground_metadata:
            parser.error("--match grid requires --labels_path or --ground_metadata")
        if args.patch_size <= 2 * args.margin:
            parser.error("--patch_size must be strictly greater than 2 * --margin")
    if not args.output_dir:
        parser.error("--output_dir is required (or use --country_dir)")
    if args.min_grounds < 1:
        parser.error("--min_grounds must be at least 1")
    if args.check_images:
        if args.match == "labels" and not (args.ground_dir and args.aerial_dir):
            parser.error("--check_images with --match labels requires --ground_dir and --aerial_dir")
        if args.match == "nearest" and not args.aerial_dir:
            parser.error("--check_images with --match nearest requires --aerial_dir")
        if args.match == "grid":
            parser.error("--check_images is not supported with --match grid")

    build_mapping(args)


if __name__ == "__main__":
    main()
