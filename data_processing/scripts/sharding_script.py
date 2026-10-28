import json
import argparse
import pandas as pd
import webdataset as wds
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
from pyproj import Transformer, CRS

# Disable count-based splitting so shards are bounded by size only, as before.
UNLIMITED_COUNT = 2**31 - 1

AERIAL_METADATA_COLUMNS = [
    "aerial_id", "shard", "lat", "lon", "ground_count", "width", "height",
]

OFFSET_COLUMNS = ["offset_east_m", "offset_north_m"]

# Linked ground index: ground record + aerial association, shard, and offsets.
LINKED_METADATA_COLUMNS = [
    "ground_id", "shard", "is_panorama", "has_label", "mapil_sequence_id",
    "quality_score", "captured_at", "lat", "lon", "heading", "aerial_id",
    "osm_id", "text_caption", "has_aerial",
] + OFFSET_COLUMNS

METADATA_COLUMNS = LINKED_METADATA_COLUMNS


def get_utm_crs_by_zone(utm_zone, is_north=True):
    """WGS84 UTM CRS for a zone and hemisphere."""
    hemisphere = "north" if is_north else "south"
    return CRS.from_string(
        f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
    )


def get_utm_zone(lat, lon):
    """UTM zone and hemisphere for a point."""
    return int((lon + 180) // 6 + 1), lat >= 0


def get_zone_transformers(lat, lon, cache):
    """(to_utm, to_latlon) for the zone containing lat/lon, cached per zone."""
    key = get_utm_zone(lat, lon)
    if key not in cache:
        utm_crs = get_utm_crs_by_zone(*key)
        cache[key] = (
            Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True),
            Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True),
        )
    return cache[key]


def compute_utm_offsets(transformer, aerial_lat, aerial_lon, ground_lat, ground_lon):
    """Ground displacement from the patch center in meters, rounded to cm."""
    ax, ay = transformer.transform(aerial_lon, aerial_lat)
    gx, gy = transformer.transform(ground_lon, ground_lat)
    return round(gx - ax, 2), round(gy - ay, 2)


def select_heading(entry):
    """(heading, source) for the ground camera.

    The raw heading is a placeholder before the meridian correction, so those
    frames take the OpenSfM heading.
    """
    raw = entry.get("heading_mapilraw")
    opensfm = entry.get("heading_mapilopensfm")
    if opensfm is not None and (
        entry.get("heading_mapilraw_before_meridian_is_zero") or raw is None
    ):
        return opensfm, "mapilopensfm"
    if raw is not None:
        return raw, "mapilraw"
    return None, "none"


def select_pose(entry):
    """Ground pose for this entry: raw Mapillary pose, OpenSfM if missing.

    Matches the OpenCVL construction, which builds each crop around the raw
    GNSS pose (camera within +/-20 m of the patch center).
    """
    raw_latlon = entry.get("ground_latlon_mapilraw") or entry.get("ground_latlon")
    heading, heading_source = select_heading(entry)
    pose = {
        "source": "mapilraw",
        "lat": raw_latlon[0] if raw_latlon else None,
        "lon": raw_latlon[1] if raw_latlon else None,
        "heading": heading,
        "heading_source": heading_source,
        "dx": entry.get("dx_mapilraw"),
        "dy": entry.get("dy_mapilraw"),
    }
    if pose["lat"] is not None and pose["lon"] is not None \
            and pose["dx"] is not None and pose["dy"] is not None:
        return pose

    of_latlon = entry.get("ground_latlon_mapilopensfm")
    of_dx = entry.get("dx_mapilopensfm")
    of_dy = entry.get("dy_mapilopensfm")
    if of_latlon and of_dx is not None and of_dy is not None:
        heading, heading_source = select_heading(entry)
        return {
            "source": "mapilopensfm",
            "lat": of_latlon[0],
            "lon": of_latlon[1],
            "heading": heading,
            "heading_source": heading_source,
            "dx": of_dx,
            "dy": of_dy,
        }

    return pose


def get_label_offsets(entry):
    """Ground-to-center displacement in meters from the labels' pixel offsets.

    Only used to recover patch centers; sharded offsets are recomputed in UTM.
    """
    pose = select_pose(entry)
    resolution = entry.get('resolution')
    if pose['dx'] is None or pose['dy'] is None or resolution is None:
        return None
    return pose['dx'] * resolution, pose['dy'] * resolution


def recover_patch_center(ground_lat, ground_lon, offset_east_m, offset_north_m,
                         transformers):
    """Patch center = ground point - offset (offsets are ground minus center)."""
    to_utm, to_latlon = transformers
    gx, gy = to_utm.transform(ground_lon, ground_lat)
    center_lon, center_lat = to_latlon.transform(
        gx - offset_east_m, gy - offset_north_m
    )
    return center_lat, center_lon


def make_aerial_id(lat, lon):
    return f"aerial_{lat:.7f}_{lon:.7f}"


def make_shard_writer(path, max_shard_size, prefix):
    """webdataset ShardWriter that splits by byte size only."""
    return wds.ShardWriter(
        str(Path(path) / f"{prefix}-%06d.tar"),
        maxsize=max_shard_size,
        maxcount=UNLIMITED_COUNT,
    )


def remove_if_empty(writer, written):
    """Drop the eagerly opened first shard when nothing was written."""
    if written == 0 and writer.fname and Path(writer.fname).exists():
        Path(writer.fname).unlink()


def assert_unique_ground_ids(labels):
    """Fail before writing if sample keys are not unique."""
    counts = Counter(str(entry['mapil_image_id']) for entry in labels)
    duplicates = [gid for gid, n in counts.items() if n > 1]
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(
            f"{len(duplicates)} duplicate mapil_image_id value(s) in labels "
            f"(e.g. {preview}); sample keys must be unique."
        )


def clear_stale_shards(directory, prefix, overwrite):
    """Remove leftover shards from a previous run, which would be mixed in."""
    stale = sorted(Path(directory).glob(f"{prefix}-*.tar"))
    if not stale:
        return
    if not overwrite:
        raise FileExistsError(
            f"{len(stale)} existing {prefix}-*.tar shard(s) in {directory}. "
            f"Remove them or pass --overwrite to replace them."
        )
    for path in stale:
        path.unlink()
    print(f"Removed {len(stale)} stale {prefix}-*.tar shard(s) from {directory}.")


def shard_dataset(city, labels_path, ground_images_dir, aerial_images_dir,
                  output_base_dir, max_shard_size_gb=5, overwrite=False,
                  aerial_mapping_path=None):
    """Recompute UTM offsets and shard by size using many_to_one.py's mapping.

    The mapping is authoritative: only patches/grounds it names are written.
    Keys must match provider patches (`aerial_<lat>_<lon>`).
    """
    MAX_SHARD_SIZE = int(max_shard_size_gb * 1024 * 1024 * 1024)
    print(f"Loading labels from {labels_path}...")
    with open(labels_path, 'r') as f:
        data = json.load(f)

    labels = data.get('labels', [])
    print(f"Found {len(labels)} labels.")
    assert_unique_ground_ids(labels)

    # Setup output directories
    city_path = Path(output_base_dir) / city
    aerial_out_path = Path(aerial_images_dir)
    ground_dir = Path(ground_images_dir)
    aerial_dir = Path(aerial_images_dir)

    mapping_path = (
        Path(aerial_mapping_path) if aerial_mapping_path
        else aerial_out_path / "aerial_to_grounds.json"
    )
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Aerial mapping not found: {mapping_path}. Run many_to_one.py for "
            f"this city first."
        )
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    print(f"Using aerial mapping from {mapping_path}...")

    transformer_cache = {}
    aerial_sizes = {}
    aerial_filenames = {}

    # Provider patches: group labels by the `aerial_image` filename stem, then
    # key each patch by its recovered center.
    provider_to_grounds = defaultdict(list)
    for entry in labels:
        aerial_filename = Path(entry['aerial_image']).name
        provider_to_grounds[Path(aerial_filename).stem].append(entry)

    provider = {}
    for grounds in provider_to_grounds.values():
        anchor = grounds[0]
        pose = select_pose(anchor)
        offsets = get_label_offsets(anchor)
        if offsets is None:
            center = (pose['lat'], pose['lon'])
        else:
            transformers = get_zone_transformers(
                pose['lat'], pose['lon'], transformer_cache
            )
            center = recover_patch_center(
                pose['lat'], pose['lon'], offsets[0], offsets[1], transformers
            )

        img_filename = Path(anchor['aerial_image']).name
        img_path = aerial_dir / img_filename
        try:
            with Image.open(img_path) as im:
                size = im.size
        except Exception:
            size = (None, None)

        aid = make_aerial_id(center[0], center[1])
        if aid in provider:
            raise ValueError(f"aerial center collision for {aid}")
        provider[aid] = {"center": center, "filename": img_filename, "size": size}

    unknown = sorted(set(mapping) - set(provider))
    if unknown:
        raise ValueError(
            f"{len(unknown)} mapping aerial key(s) do not match provider patches "
            f"(e.g. {unknown[:3]}). Regenerate the mapping with `many_to_one.py`."
        )

    ground_entries = {str(entry['mapil_image_id']): entry for entry in labels}
    aerial_to_grounds = {}
    aerial_centers = {}
    seen_grounds = {}
    for aid, ground_ids in mapping.items():
        aid = str(aid)
        entries = []
        for gid in ground_ids:
            gid = str(gid)
            entry = ground_entries.get(gid)
            if entry is None:
                raise ValueError(f"mapping ground {gid} is not present in {labels_path}")
            if gid in seen_grounds:
                raise ValueError(
                    f"ground {gid} appears in both {seen_grounds[gid]} and {aid}"
                )
            seen_grounds[gid] = aid
            entries.append(entry)

        info = provider[aid]
        aerial_to_grounds[aid] = entries
        aerial_centers[aid] = info["center"]
        aerial_filenames[aid] = info["filename"]
        aerial_sizes[aid] = info["size"]

    print(
        f"Mapping: {len(aerial_to_grounds)} aerials for "
        f"{sum(len(v) for v in aerial_to_grounds.values())} ground images."
    )

    city_path.mkdir(parents=True, exist_ok=True)
    aerial_out_path.mkdir(parents=True, exist_ok=True)

    clear_stale_shards(aerial_out_path, "aerial", overwrite)
    clear_stale_shards(city_path, "dataset", overwrite)

    # 2. Sharding Aerials (Unique patches only)
    unique_aerial_ids = list(aerial_to_grounds.keys())

    missing_aerials = 0
    written_aerials = 0
    aerial_metadata_rows = []
    written_aerial_ids = set()

    with make_shard_writer(aerial_out_path, MAX_SHARD_SIZE, "aerial") as writer:
        for aid in unique_aerial_ids:
            a_lat, a_lon = aerial_centers[aid]
            ground_ids = [
                str(g['mapil_image_id']) for g in aerial_to_grounds[aid]
            ]

            img_path = aerial_dir / aerial_filenames[aid]
            if not img_path.exists():
                missing_aerials += 1
                continue
            img_bytes = img_path.read_bytes()
            width, height = aerial_sizes.get(aid, (None, None))
            resolution = aerial_to_grounds[aid][0].get("resolution")
            patch_size = (
                round(width * resolution, 2) if width and resolution else None
            )

            writer.write({
                "__key__": str(aid),
                "png": img_bytes,
                "json": {
                    "aerial_id": aid,
                    "lat": a_lat,
                    "lon": a_lon,
                    "patch_size_m": patch_size,
                    "margin_m": 0.0,
                    "width_px": width,
                    "height_px": height,
                    "ground_count": len(ground_ids),
                    "ground_ids": ground_ids,
                },
            })
            written_aerials += 1
            written_aerial_ids.add(str(aid))

            aerial_metadata_rows.append({
                "aerial_id": str(aid),
                "shard": writer.fname,
                "lat": a_lat,
                "lon": a_lon,
                "ground_count": 0,  # filled in after the ground pass
                "width": width,
                "height": height,
            })

            if written_aerials % 1000 == 0:
                print(f"  aerial patches written: {written_aerials}/{len(unique_aerial_ids)}")

    remove_if_empty(writer, written_aerials)
    print(f"Aerial shards written to {aerial_out_path} ({written_aerials} patches, {missing_aerials} missing).")

    # 3. Sharding Grounds, with offsets recomputed in UTM against the patch center
    metadata_rows = []
    linked_rows = []
    missing_grounds = 0
    written_grounds = 0
    opensfm_fallback_count = 0
    heading_from_opensfm = 0
    outside_patch = 0

    with make_shard_writer(city_path, MAX_SHARD_SIZE, "dataset") as writer:
        for aid, grounds in aerial_to_grounds.items():
            a_lat, a_lon = aerial_centers[aid]

            # A shared patch must actually cover every ground assigned to it.
            resolution = grounds[0].get("resolution") if grounds else None
            width, height = aerial_sizes.get(aid, (None, None))
            if width and height and resolution:
                half_east = width * resolution / 2.0
                half_north = height * resolution / 2.0
            else:
                half_east = half_north = None

            for g_entry in grounds:
                gid = str(g_entry['mapil_image_id'])
                img_filename = Path(g_entry['mapil_image']).name
                img_path = ground_dir / img_filename

                if not img_path.exists():
                    missing_grounds += 1
                    continue

                pose = select_pose(g_entry)
                if pose['source'] == 'mapilopensfm':
                    opensfm_fallback_count += 1
                if pose.get('heading_source') == 'mapilopensfm':
                    heading_from_opensfm += 1
                g_lat, g_lon = pose['lat'], pose['lon']
                # Recompute offsets in UTM against the patch center.
                transformer = get_zone_transformers(g_lat, g_lon, transformer_cache)[0]
                dx, dy = compute_utm_offsets(transformer, a_lat, a_lon, g_lat, g_lon)
                if half_east is not None and (
                    abs(dx) > half_east or abs(dy) > half_north
                ):
                    outside_patch += 1

                meta = g_entry.copy()
                meta['offset_east_m'] = dx
                meta['offset_north_m'] = dy

                # writer.fname is the shard this sample actually landed in.
                writer.write({
                    "__key__": gid,
                    "png": img_path.read_bytes(),
                    "json": meta,
                })
                written_grounds += 1

                if written_grounds % 1000 == 0:
                    print(f"  ground images written: {written_grounds}")

                row = {
                    "ground_id": gid,
                    "shard": writer.fname,
                    "is_panorama": bool(meta.get('is_panorama', False)),
                    "has_label": True,
                    "mapil_sequence_id": meta.get('mapil_sequence_id'),
                    "quality_score": meta.get('quality_score'),
                    "captured_at": meta.get('captured_at'),
                    "lat": pose['lat'],
                    "lon": pose['lon'],
                    "heading": pose['heading'],
                    "aerial_id": aid,
                    "osm_id": meta.get('osm_id'),
                    "text_caption": meta.get('text_caption'),
                    "has_aerial": aid in written_aerial_ids,
                    "offset_east_m": meta['offset_east_m'],
                    "offset_north_m": meta['offset_north_m'],
                }
                # Both parquet paths carry the same linked index.
                metadata_rows.append({k: row[k] for k in METADATA_COLUMNS})
                linked_rows.append(row)

    remove_if_empty(writer, written_grounds)
    print(f"Ground shards written to {city_path} ({written_grounds} images, {missing_grounds} missing).")
    print(f"Pose selection: {written_grounds - opensfm_fallback_count} raw, {opensfm_fallback_count} OpenSfM fallback.")
    print(f"Heading selection: {written_grounds - heading_from_opensfm} raw, {heading_from_opensfm} OpenSfM (raw heading was unset).")

    counts = [len(grounds) for grounds in aerial_to_grounds.values()]
    if counts:
        multi = sum(1 for count in counts if count > 1)
        print(
            f"Association: {len(counts)} aerials for {written_grounds} grounds "
            f"(mean {written_grounds / len(counts):.2f} per aerial, "
            f"{multi} shared by more than one)."
        )
    if outside_patch:
        print(
            f"WARNING: {outside_patch} ground offset(s) fall outside their aerial "
            f"patch image; use a smaller --anchor_radius_m for the mapping."
        )

    if metadata_rows:
        df = pd.DataFrame(metadata_rows, columns=METADATA_COLUMNS)
        metadata_path = city_path / "metadata.parquet"
        df.to_parquet(metadata_path)
        print(f"Metadata saved to {metadata_path}")

    if aerial_metadata_rows:
        for row in aerial_metadata_rows:
            # Association count, as in the reference pipeline's aerial metadata.
            row['ground_count'] = len(aerial_to_grounds.get(row['aerial_id'], []))
        aerial_df = pd.DataFrame(aerial_metadata_rows, columns=AERIAL_METADATA_COLUMNS)
        aerial_metadata_path = aerial_out_path / "aerial_metadata.parquet"
        aerial_df.to_parquet(aerial_metadata_path)
        print(f"Aerial metadata saved to {aerial_metadata_path}")

    if linked_rows:
        linked_df = pd.DataFrame(linked_rows, columns=LINKED_METADATA_COLUMNS)
        linked_path = aerial_out_path / "ground_metadata_linked.parquet"
        linked_df.to_parquet(linked_path)
        print(f"Linked metadata saved to {linked_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Many-to-One Sharding with UTM Recomputation")
    parser.add_argument("--city", type=str, help="City name for output directory")
    parser.add_argument("--labels_path", type=str, help="Path to labels.json")
    parser.add_argument("--ground_dir", type=str, help="Directory containing ground images")
    parser.add_argument("--aerial_dir", type=str, help="Directory containing aerial images")
    parser.add_argument("--output_dir", type=str, help="Base directory for output")
    parser.add_argument("--aerial_mapping", type=str,
                        help="Path to aerial_to_grounds.json (default: <aerial_dir>/aerial_to_grounds.json)")
    parser.add_argument("--max_shard_size", type=int, default=5, help="Max shard size in GB")
    parser.add_argument("--country_dir", type=str, help="Path to the country directory containing city folders")
    parser.add_argument("--overwrite", action="store_true",
                        help="Remove existing aerial-*.tar / dataset-*.tar shards before writing")

    args = parser.parse_args()

    if args.country_dir:
        # Process all cities in the country directory
        country_path = Path(args.country_dir)
        # Find all directories that have a labels.json inside them
        cities = [d.name for d in country_path.iterdir() if d.is_dir() and (d / "labels.json").exists()]
        print(f"Found {len(cities)} cities in {args.country_dir}: {cities}")

        for city in cities:
            print(f"\n--- Processing City: {city} ---")
            shard_dataset(
                city=city,
                labels_path=str(country_path / city / "labels.json"),
                ground_images_dir=str(country_path / city / "ground"),
                aerial_images_dir=str(country_path / city / "aerial"),
                output_base_dir=str(country_path),
                max_shard_size_gb=args.max_shard_size,
                overwrite=args.overwrite,
            )
    else:
        # Standard single-city process
        if not all([args.city, args.labels_path, args.ground_dir, args.aerial_dir, args.output_dir]):
            parser.error("Either --country_dir OR all city-specific arguments must be provided.")

        shard_dataset(
            city=args.city,
            labels_path=args.labels_path,
            ground_images_dir=args.ground_dir,
            aerial_images_dir=args.aerial_dir,
            output_base_dir=args.output_dir,
            max_shard_size_gb=args.max_shard_size,
            overwrite=args.overwrite,
            aerial_mapping_path=args.aerial_mapping,
        )
