# Dataset architecture

Output-format reference. Directory layout, build commands, and script flags are in [`README.md`](README.md).

## Shard contents

Ground shards (`dataset-XXXXXX.tar`) hold `<id>.png` with a matching `<id>.json` carrying pose and source fields (`mapil_sequence_id`, `quality_score`, `captured_at`, …) plus `offset_east_m` / `offset_north_m` when written by `sharding_script.py`. The original Burlington ground JSON has no offsets; its offsets live in `ground_metadata_linked.parquet`.

Aerial shards (`aerial-XXXXXX.tar`) hold `<aerial_id>.png` with a `<aerial_id>.json` containing `aerial_id`, patch-center `lat`/`lon`, `patch_size_m`, `margin_m`, `width_px`/`height_px`, `ground_count`, and `ground_ids`. Patches are keyed `aerial_<lat>_<lon>` from their recovered center.

## Ground index

`metadata.parquet` and `aerial/ground_metadata_linked.parquet` are identical: one row per sharded ground image, carrying both the ground record and its aerial link. They allow querying without extracting the shards.

| Column | Type | Description |
| :--- | :--- | :--- |
| `ground_id` | String | Unique ground image identifier. |
| `shard` | String | Tar file containing the image. |
| `is_panorama` / `has_label` | Bool | Panorama and label flags. |
| `mapil_sequence_id` / `quality_score` / `captured_at` | — | Source Mapillary fields, when present. |
| `lat` / `lon` | Float | GPS coordinates of the capture point. |
| `heading` | Float | Compass heading. |
| `aerial_id` | String | Linked aerial patch. |
| `osm_id` / `text_caption` | — | Source OpenStreetMap fields, when present. |
| `has_aerial` | Bool | Whether the linked patch was written; `False` if the ground exists but its patch was missing. |
| `offset_east_m` / `offset_north_m` | Float | Ground position relative to the patch center, in meters. |

## Aerial index

`aerial/aerial_metadata.parquet` has one row per unique patch.

| Column | Type | Description |
| :--- | :--- | :--- |
| `aerial_id` | String | Patch identifier; matches the `__key__` in the aerial tar. |
| `shard` | String | Tar file containing the patch. |
| `lat` / `lon` | Float | Patch center, recovered as the ground position minus the ground-to-center displacement from the labels (`dx`/`dy * resolution`). |
| `ground_count` | Int | Number of associated ground images. |
| `width` / `height` | Int | Patch dimensions in pixels, or `None` if the image could not be read. |

 
