# Dataset architecture

## High-level structure

The dataset is organized to support efficient matching between ground-level imagery and aerial patches.

```text
<country>/<city>/
├── dataset-xxxxxx.tar                   # ground image shards
├── metadata.parquet                     # ground-level index
└── aerial/
    ├── aerial-xxxxxx.tar                # aerial patch shards
    ├── aerial_metadata.parquet          # one row per aerial patch
    ├── ground_metadata_linked.parquet   # ground index linked to aerials
    └── samples_preview/        # Visual alignment examples
```


---

## Data formats

### Sharded `.tar` files

To avoid the "millions of small files" problem on disk, the dataset is sharded into large `.tar` archives.

#### Ground shards (`dataset-XXXXXX.tar`)

Each archive contains a batch of ground-level images and their associated metadata:

- **`[image_id].png`** — the ground-level photograph.
- **`[image_id].json`** — per-image metadata, including GPS coordinates, heading, and pose fields. Ground JSON written by `scripts/sharding_script.py` also carries `offset_east_m` / `offset_north_m` relative to the aerial patch center. The original Burlington ground JSON carries pose and source metadata (`mapil_sequence_id`, `quality_score`, `captured_at`, …) but not the offsets; Burlington's offsets live in `ground_metadata_linked.parquet`.

#### Aerial shards (`aerial-XXXXXX.tar`)

These contain the overhead patches associated with the ground images. Patches are keyed `aerial_<lat>_<lon>` from their center coordinates (Burlington layout).

- **`[aerial_id].png`** — the provider's aerial image patch.
- **`[aerial_id].json`** — metadata for the aerial patch: `aerial_id`, patch-center `lat`/`lon`, `patch_size_m`, `margin_m`, `width_px`/`height_px`, `ground_count`, and the associated `ground_ids`.

---

## Metadata specifications

### `metadata.parquet`

The primary index for the ground-level dataset. It allows fast querying and filtering without extracting the `.tar` files. One row per sharded ground image, carrying the full ground↔aerial linked index — the same rows and columns as `aerial/ground_metadata_linked.parquet`, matching the reference Burlington layout.

| Column | Type | Description |
| :--- | :--- | :--- |
| **`ground_id`** | String | Unique identifier for the ground-level image. |
| **`shard`** | String | Tar file the ground image was written to. |
| **`is_panorama` / `has_label`** | Bool | Panorama and label flags. |
| **`mapil_sequence_id` / `quality_score` / `captured_at`** | — | Source Mapillary fields, when present. |
| **`lat` / `lon`** | Float | GPS coordinates of the image capture point. |
| **`heading`** | Float | Camera orientation (compass heading). |
| **`aerial_id`** | String | Aerial patch the ground image is linked to. |
| **`osm_id` / `text_caption`** | — | Source OpenStreetMap fields, when present. |
| **`has_aerial`** | Boolean | `True` when the linked aerial patch was written. |
| **`offset_east_m` / `offset_north_m`** | Float | Ground position relative to the aerial patch center, in meters. |

The same table is written to `aerial/ground_metadata_linked.parquet`, so downstream consumers can read either path.

### `aerial_metadata.parquet`

One row per unique aerial patch. Located at `aerial/aerial_metadata.parquet`.

| Column | Type | Description |
| :--- | :--- | :--- |
| `aerial_id` | String | Patch identifier (the `aerial_image` filename stem). Matches the `__key__` in the aerial tar. |
| `shard` | String | Tar file the patch was written to (e.g. `aerial/aerial-000000.tar`). |
| `lat` | Float | Patch center latitude. Recovered as the ground position minus the ground-to-center displacement from the labels (`dx`/`dy * resolution` of the selected pose), not taken from the anchor ground point. |
| `lon` | Float | Patch center longitude, recovered the same way. |
| `ground_count` | Int | Number of ground images associated with this patch. |
| `width` | Int | Patch width in pixels, or `None` if the image could not be read. |
| `height` | Int | Patch height in pixels, or `None` if the image could not be read. |

Relationship: `aerial_id` joins to `ground_metadata_linked.parquet.aerial_id` and to the ground JSON records in `dataset-*.tar`.

### `ground_metadata_linked.parquet`

One row per sharded ground image, joined to its aerial patch. Located at `aerial/ground_metadata_linked.parquet`. The filename is `ground_metadata_linked.parquet` (not `ground_aerial_metadata_linked.parquet`).

| Column | Type | Description |
| :--- | :--- | :--- |
| *(ground index)* | — | The full 13-column ground index: `ground_id`, `shard`, `is_panorama`, `has_label`, `mapil_sequence_id`, `quality_score`, `captured_at`, `lat`, `lon`, `heading`, `aerial_id`, `osm_id`, `text_caption`. |
| `has_aerial` | Boolean | `True` when the linked aerial patch image existed and was written to the aerial shards; `False` when the ground image was written but its patch was missing. |
| `offset_east_m` | Float | Ground position relative to the patch center, eastward, in meters. Recomputed in UTM as the ground point minus the recovered patch center. |
| `offset_north_m` | Float | Ground position relative to the patch center, northward, in meters. Recomputed in UTM as the ground point minus the recovered patch center. |

Relationship: `ground_id` matches `metadata.parquet.ground_id` and the `__key__` in `dataset-*.tar`. `aerial_id` matches `aerial_metadata.parquet.aerial_id`.

Offsets are measured as **ground minus patch center**, so the patch center is `ground - offset`. Positive `offset_north_m` is north; positive `offset_east_m` is east. The calculation matches the reference pipeline: both the ground point and the recovered patch center are projected into UTM and the displacement is rounded to centimeters, with the zone taken from the longitude and the hemisphere from the latitude. The ground pose is the raw Mapillary pose (`ground_latlon_mapilraw`, `heading_mapilraw`, `dx_mapilraw`/`dy_mapilraw`), matching the OpenCVL construction: each 100 m aerial crop is built around the raw GNSS pose, with the camera within ±20 m of the patch center along each axis. The OpenSfM fields are used only as a fallback when the raw fields are missing, except for heading: frames flagged `heading_mapilraw_before_meridian_is_zero` (the raw heading was unset before meridian correction) take the OpenSfM heading. A pose's lat/lon and pixel offsets are used together, and only that pose's offsets recover each patch's center rather than populate the offset columns.

### `aerial_to_grounds.json`

Maps a single aerial patch ID to the list of all ground-level images that fall within that patch's footprint. `scripts/many_to_one.py` is the sole writer, and `scripts/sharding_script.py` consumes this file as its authoritative association. `labels` mode uses the provider pairing; `nearest` mode assigns grounds to the nearest existing patch centers; `anchors` mode keeps the provider's patch centers as anchors and greedily shares each crop among the grounds it covers, producing many-to-one from one-crop-per-ground input without new imagery. These three key patches by their recovered center (`aerial_<lat>_<lon>`), so the mapping matches the aerial shards. `grid` mode synthesizes cells from a random-phase UTM grid instead and is not consumable by the sharding script, which has no imagery for those centers.

**Example format:**

```json
{
  "aerial_ID_123": ["ground_ID_A", "ground_ID_B", "ground_ID_C"],
  "aerial_ID_456": ["ground_ID_D", "ground_ID_E"]
}
```

---

## Logical workflow

1. **Lookup** — use `metadata.parquet` (or the identical `aerial/ground_metadata_linked.parquet`) for the ground `ground_id`, coordinates, heading, offsets, `shard`, and `aerial_id`.
2. **Retrieve** — extract the `.png` and `.json` from the specified `.tar` file.
3. **Align** — use the `aerial_id` from `ground_metadata_linked.parquet` to find the matching aerial patch in the `aerial/` directory (or use `aerial_to_grounds.json` to go the other way).
4. **Project** — apply `offset_east_m` / `offset_north_m` to locate the ground-view point within the aerial patch.
