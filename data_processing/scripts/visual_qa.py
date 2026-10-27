import io
import json
import argparse
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def load_font(size):
    """
    Load a scalable font for the label, falling back to the bitmap default.
    """
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def is_missing(value):
    """
    True for None / NaN / NA, without choking on non-scalar values.
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def pick(row, meta, key, default=None):
    """
    Prefer the parquet field, falling back to the per-image shard metadata.
    """
    value = row.get(key)
    if is_missing(value):
        value = meta.get(key, default)
    if is_missing(value):
        return default
    return value


def arrow_head(tip, direction, length, half_width):
    """
    Triangle points for an arrowhead at `tip`, pointing along `direction`.
    """
    ux, uy = direction
    base_x, base_y = tip[0] - length * ux, tip[1] - length * uy
    perp_x, perp_y = -uy, ux
    return [
        tip,
        (base_x + half_width * perp_x, base_y + half_width * perp_y),
        (base_x - half_width * perp_x, base_y - half_width * perp_y),
    ]


def find_image(directory, filename, stem):
    """
    Locate an image by its recorded filename, then by its ID stem.
    """
    directory = Path(directory)
    if filename and not is_missing(filename):
        candidate = directory / Path(filename).name
        if candidate.exists():
            return candidate
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def resolve_shard_path(raw_shard, city_root):
    """
    Resolve the shard path recorded at sharding time.

    ShardWriter stores the path it was given, which may be relative to whatever
    working directory the sharding run used. Try the current directory, the city
    root, and finally the bare filename next to the city's other shards.
    """
    if is_missing(raw_shard):
        return None
    path = Path(raw_shard)
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path,
        Path(city_root) / path,
        Path(city_root) / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class AerialPatches:
    """
    Aerial patch images and metadata, read from the aerial shards.

    The reference pipeline renders previews from the aerial belonging to the
    assigned patch, so the image and the offsets share one center. Reading the
    shard (keyed by `aerial_id`) reproduces that; the source aerial directory
    may contain a different crop per ground.
    """

    def __init__(self, aerial_dir, city_root=None):
        self.aerial_dir = Path(aerial_dir)
        self.city_root = Path(city_root) if city_root else self.aerial_dir.parent
        self._shards = {}
        self._tars = {}
        self._meta_cache = {}
        if not self.aerial_dir.exists():
            return
        metadata_path = self.aerial_dir / "aerial_metadata.parquet"
        if metadata_path.exists():
            frame = pd.read_parquet(metadata_path, columns=["aerial_id", "shard"])
            for aerial_id, shard in zip(frame["aerial_id"], frame["shard"]):
                self._shards[str(aerial_id)] = str(shard)
        else:
            self._index_shards()

    def _index_shards(self):
        """
        Fall back to scanning the aerial shards when no metadata table exists.
        """
        for shard in sorted(self.aerial_dir.glob("aerial-*.tar")):
            try:
                with tarfile.open(shard, "r") as tar:
                    for name in tar.getnames():
                        if name.endswith(".json"):
                            self._shards[name[:-5]] = str(shard)
            except (tarfile.TarError, OSError):
                continue

    def _resolve(self, raw):
        """
        Resolve a shard path recorded at sharding time.
        """
        path = Path(raw)
        if path.is_absolute():
            candidates = [path]
        else:
            candidates = [
                Path.cwd() / path,
                self.aerial_dir / path,
                self.aerial_dir / path.name,
                self.city_root / path,
            ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _tar(self, aerial_id):
        raw = self._shards.get(aerial_id)
        if raw is None:
            self._index_shards()
            raw = self._shards.get(aerial_id)
        if raw is None:
            return None
        path = self._resolve(raw)
        if path is None:
            return None
        if path not in self._tars:
            self._tars[path] = tarfile.open(path, "r")
        return self._tars[path]

    def get(self, aerial_id):
        """
        Return {'image_bytes', 'meta'} for an aerial patch, or None.
        """
        aerial_id = str(aerial_id)
        tar = self._tar(aerial_id)
        if tar is None:
            return None
        try:
            image_file = tar.extractfile(f"{aerial_id}.png")
        except KeyError:
            image_file = None
        if image_file is None:
            return None
        meta = self._meta_cache.get(aerial_id)
        if meta is None:
            try:
                meta = json.load(tar.extractfile(f"{aerial_id}.json"))
            except (KeyError, TypeError, json.JSONDecodeError):
                meta = {}
            self._meta_cache[aerial_id] = meta
        # Image bytes are not cached: a full run touches every patch, and the
        # shard read is cheap.
        return {"image_bytes": image_file.read(), "meta": meta}

    def close(self):
        for tar in self._tars.values():
            try:
                tar.close()
            except OSError:
                pass
        self._tars.clear()


def read_ground_json(tar, ground_id):
    """
    Read the ground's per-image metadata JSON from its shard, if available.
    """
    if tar is None:
        return {}
    try:
        extracted = tar.extractfile(f"{ground_id}.json")
    except KeyError:
        return {}
    if extracted is None:
        return {}
    try:
        return json.load(extracted)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def load_index(city_root, linked_metadata_path=None):
    """
    Build the per-ground index from the sharding metadata parquets.

    metadata.parquet already carries the full linked ground record (including
    `aerial_id`, `shard`, and `has_aerial`); ground_metadata_linked.parquet is
    read as a fallback and merged only for any link columns it adds.
    """
    metadata_path = Path(city_root) / "metadata.parquet"
    linked_path = (
        Path(linked_metadata_path) if linked_metadata_path
        else Path(city_root) / "aerial" / "ground_metadata_linked.parquet"
    )

    if metadata_path.exists():
        df = pd.read_parquet(metadata_path)
    elif linked_path.exists():
        df = pd.read_parquet(linked_path)
    else:
        raise FileNotFoundError(
            f"No metadata.parquet or ground_metadata_linked.parquet in {city_root}"
        )

    if linked_path.exists():
        linked = pd.read_parquet(linked_path)
        link_columns = [
            column for column in ("shard", "aerial_id", "has_aerial")
            if column in linked.columns and column not in df.columns
        ]
        if "ground_id" in linked.columns and link_columns:
            df = df.merge(
                linked[["ground_id"] + link_columns].drop_duplicates("ground_id"),
                on="ground_id",
                how="left",
            )

    if "has_aerial" in df.columns:
        df = df[df["has_aerial"].fillna(False)]

    # Without the linked parquet there is no shard column. A single-shard city
    # (typical in tests) can still resolve its per-image JSON.
    if "shard" not in df.columns:
        shards = sorted(Path(city_root).glob("dataset-*.tar"))
        if len(shards) == 1:
            df = df.copy()
            df["shard"] = str(shards[0])

    return df


def create_qa_pair(row, tar, ground_dir, aerial_dir, output_dir,
                   aerial_patches=None):
    """
    Generates a visual QA pair: aerial patch with ground dot and heading arrow,
    using the recomputed physical UTM offsets from the metadata.

    The aerial is the patch written for `aerial_id` (as in the reference
    pipeline), so the image, its GSD, and the offsets all share one center.
    """
    ground_id = str(row["ground_id"])
    meta = read_ground_json(tar, ground_id)

    aerial_id = pick(row, meta, "aerial_id")
    if aerial_id is None and meta.get("aerial_image"):
        aerial_id = Path(meta["aerial_image"]).stem
    if aerial_id is None:
        print(f"Skipping {ground_id}: no aerial link in metadata.")
        return False
    aerial_id = str(aerial_id)

    ground_img_path = find_image(ground_dir, meta.get("mapil_image"), ground_id)
    if ground_img_path is None:
        print(f"Skipping {ground_id}: ground image missing.")
        return False

    patch = aerial_patches.get(aerial_id) if aerial_patches is not None else None
    if patch is not None:
        patch_meta = patch["meta"]
        img = Image.open(io.BytesIO(patch["image_bytes"])).convert("RGB")
        # The patch's own GSD, matching the image being drawn on.
        if patch_meta.get("patch_size_m") and img.width:
            resolution = patch_meta["patch_size_m"] / img.width
        else:
            resolution = pick(row, meta, "resolution")
        margin_m = float(patch_meta.get("margin_m") or 0.0)
    else:
        aerial_img_path = find_image(aerial_dir, meta.get("aerial_image"), aerial_id)
        if aerial_img_path is None:
            print(f"Skipping {ground_id}: aerial image missing.")
            return False
        img = Image.open(aerial_img_path).convert("RGB")
        resolution = pick(row, meta, "resolution")
        margin_m = 0.0

    if resolution is None:
        print(f"Skipping {ground_id}: no aerial resolution in metadata.")
        return False

    draw = ImageDraw.Draw(img)
    width_px, height_px = img.size

    # Margin kept inside the patch, as drawn by the reference pipeline.
    if margin_m > 0 and patch is not None and patch["meta"].get("patch_size_m"):
        patch_size_m = patch["meta"]["patch_size_m"]
        margin_x = int((margin_m / patch_size_m) * width_px)
        margin_y = int((margin_m / patch_size_m) * height_px)
        if margin_x or margin_y:
            draw.rectangle(
                [margin_x, margin_y, width_px - margin_x, height_px - margin_y],
                outline=(0, 255, 0),
                width=2,
            )

    # Ground position and heading, labelled in the aerial's top-left corner.
    lat = pick(row, meta, "lat")
    lon = pick(row, meta, "lon")
    if lat is None or lon is None:
        ground_latlon = meta.get("ground_latlon") or (None, None)
        lat = lat if lat is not None else ground_latlon[0]
        lon = lon if lon is not None else ground_latlon[1]

    heading = float(pick(row, meta, "heading", pick(row, meta, "heading_mapilraw", 0)))

    if lat is not None and lon is not None:
        font_size = max(14, min(48, int(height_px * 0.025)))
        font = load_font(font_size)
        label = f"N: {float(lat):.6f}°  E: {float(lon):.6f}°  Head: {heading:.1f}°"
        draw.text(
            (12, 12), label, font=font, fill=(255, 255, 0),
            stroke_width=max(1, font_size // 12), stroke_fill=(0, 0, 0),
        )

    # Center of image
    cx, cy = width_px / 2, height_px / 2

    # Recomputed physical UTM offsets (meters)
    dx_m = float(pick(row, meta, "offset_east_m", 0))
    dy_m = float(pick(row, meta, "offset_north_m", 0))

    # Convert meters to pixels
    px_x = cx + (dx_m / resolution)
    px_y = cy - (dy_m / resolution)

    # Draw heading arrow
    theta = np.radians(heading)
    ux, uy = float(np.sin(theta)), float(-np.cos(theta))
    arrow_len = max(40, int(height_px * 0.06))
    arrow_width = max(3, arrow_len // 10)
    tip = (px_x + arrow_len * ux, px_y + arrow_len * uy)

    head_len = arrow_len * 0.35
    head_half = arrow_len * 0.18

    # Black underlay keeps the arrow legible over bright imagery.
    draw.line([px_x, px_y, tip[0], tip[1]], fill=(0, 0, 0), width=arrow_width + 4)
    draw.polygon(arrow_head(tip, (ux, uy), head_len * 1.2, head_half * 1.5), fill=(0, 0, 0))
    draw.line([px_x, px_y, tip[0], tip[1]], fill="yellow", width=arrow_width)
    draw.polygon(arrow_head(tip, (ux, uy), head_len, head_half), fill="yellow")

    # Draw ground dot on top of the arrow
    r = 5
    draw.ellipse([px_x - r, px_y - r, px_x + r, px_y + r], fill="red", outline="white")

    # Create side-by-side comparison
    ground_img = Image.open(ground_img_path).convert("RGB")
    gh, gw = ground_img.size
    scale = height_px / gh
    ground_img = ground_img.resize((int(gw * scale), height_px))

    canvas = Image.new("RGB", (width_px + ground_img.width, height_px))
    canvas.paste(ground_img, (0, 0))
    canvas.paste(img, (ground_img.width, 0))

    pair_path = Path(output_dir) / f"pair_{ground_id}.jpg"
    canvas.save(pair_path)
    print(f"Generated QA pair for {ground_id}")
    return True


def process_city(city, city_root, ground_dir, aerial_dir, output_dir,
                 linked_metadata_path=None, image_id=None, limit=None, seed=1):
    """
    Generate QA pairs for one city from its sharding metadata.

    When sampling (`limit`), records are shuffled with a fixed `seed` first so
    previews span the dataset and the same records are chosen on every run.
    """
    city_root = Path(city_root)
    df = load_index(city_root, linked_metadata_path)

    if image_id:
        df = df[df["ground_id"].astype(str) == str(image_id)]
    if limit:
        if seed is not None:
            df = df.sample(frac=1.0, random_state=seed)
        df = df.head(limit)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(df)} metadata records for verification in {city}...")

    generated = 0
    aerial_patches = AerialPatches(aerial_dir, city_root)
    if "shard" in df.columns:
        groups = df.groupby("shard", dropna=False)
    else:
        groups = [(None, df)]

    try:
        for raw_shard, rows in groups:
            shard_path = resolve_shard_path(raw_shard, city_root)
            tar = tarfile.open(shard_path, "r") if shard_path else None
            try:
                for _, row in rows.iterrows():
                    if create_qa_pair(
                        row, tar, ground_dir, aerial_dir, output_dir,
                        aerial_patches=aerial_patches,
                    ):
                        generated += 1
            finally:
                if tar is not None:
                    tar.close()
    finally:
        aerial_patches.close()

    print(f"Generated {generated} QA pairs for {city}.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual QA pairs from sharded dataset metadata"
    )
    parser.add_argument("--city", type=str, help="City name for output path")
    parser.add_argument("--metadata_path", type=str,
                        help="Path to the city's metadata.parquet")
    parser.add_argument("--linked_metadata_path", type=str,
                        help="Optional path to aerial/ground_metadata_linked.parquet")
    parser.add_argument("--ground_dir", type=str,
                        help="Directory containing ground images")
    parser.add_argument("--aerial_dir", type=str,
                        help="Directory containing aerial images")
    parser.add_argument("--image_id", type=str,
                        help="Optional: process only this specific ground ID")
    parser.add_argument("--limit", type=int, default=5,
                        help="Maximum number of samples to generate per city (default: 5)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Fixed seed for the sampling shuffle; the same "
                             "records are selected on every run (default: 1)")
    parser.add_argument("--country_dir", type=str,
                        help="Path to the country directory containing city folders")

    args = parser.parse_args()

    if args.country_dir:
        country_path = Path(args.country_dir)
        cities = [
            d.name for d in sorted(country_path.iterdir())
            if d.is_dir() and (d / "metadata.parquet").exists()
        ]
        print(f"Found {len(cities)} cities in {args.country_dir}: {cities}")

        for city in cities:
            print(f"\n--- Processing City: {city} ---")
            city_root = country_path / city
            process_city(
                city=city,
                city_root=city_root,
                ground_dir=city_root / "ground",
                aerial_dir=city_root / "aerial",
                output_dir=city_root / "aerial" / "samples_preview",
                image_id=args.image_id,
                limit=args.limit,
                seed=args.seed,
            )
    else:
        if not args.metadata_path:
            parser.error("Either --country_dir or --metadata_path must be provided.")

        metadata_path = Path(args.metadata_path)
        city_root = metadata_path.parent
        city = args.city or city_root.name

        process_city(
            city=city,
            city_root=city_root,
            ground_dir=Path(args.ground_dir) if args.ground_dir else city_root / "ground",
            aerial_dir=Path(args.aerial_dir) if args.aerial_dir else city_root / "aerial",
            output_dir=city_root / "aerial" / "samples_preview",
            linked_metadata_path=args.linked_metadata_path,
            image_id=args.image_id,
            limit=args.limit,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
