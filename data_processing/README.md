# Dataset Processing Scripts

This directory contains scripts for sharding & visual QA.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

---

## `sharding_script.py`

- consumes `aerial/aerial_to_grounds.json`, produced first by `many_to_one.py`; the mapping is authoritative
  - keys must be provider patch IDs (`aerial_<lat>_<lon>`), so generate it with `--match labels`, `nearest`, or `anchors`; `grid` keys are synthetic and are rejected
  - `--match anchors` is the many-to-one path when the provider ships one crop per ground: it shares each crop among the grounds that fall inside it (see below)
- validates that every ground assigned to a shared aerial patch lies inside that patch's image, and warns when it does not
- recomputes physical UTM offsets against each aerial patch center (zone and hemisphere from the ground point)
- aerials are keyed `aerial_<lat>_<lon>` of the recovered patch center (Burlington layout)
- shards results into `.tar` files
- produces `.parquet` files

### Usage

Pass in country directory and expects country setup to be in this form:
```txt
<country>/
├── <city1>
├── <city2>
├── ...
└── <cityn>
```
where each `<cityi>` dir is of form: 

```text
<cityi>/
├── <aerial> # folder of aerial images
├── <ground> # folder of ground-view images
└── labels.json
```

### How-to-run Examples
#### Country-wide Processing

Process all cities within a country directory automatically:
```bash
python3 scripts/sharding_script.py --country_dir NO/
```

Or you can run them individually like this:
```bash
python3 scripts/sharding_script.py \
  --city <city_name> \
  --labels_path <path_to_labels_json> \
  --ground_dir <path_to_ground_images> \
  --aerial_dir <path_to_aerial_images> \
  --output_dir <path_to_city_root_dir> \
  --max_shard_size <size_in_gb>
```

### Arguments
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--city` | Name of the city (used for output folder naming). | **Required** |
| `--labels_path` | Path to the `labels.json` file mapping images. | **Required** TODO: set default to <city>/labels |
| `--ground_dir` | Directory containing ground-level `.png` images. | **Required** TODO: set default to <city>/ground |
| `--aerial_dir` | Directory containing the provider's aerial patch images. | **Required** TODO: set default to <city>/aerial |
| `--output_dir` | Base directory where the city folder and shards will be created. | **Required** TODO: default is the same as the city|
| `--aerial_mapping` | Path to `aerial_to_grounds.json`. | `<aerial_dir>/aerial_to_grounds.json` |
| `--max_shard_size` | Maximum size of each `.tar` file in Gigabytes. | `5` |

### Output Structure
The script produces a directory structure identical to the Burlington VT format:
```text
<country>/<city>/
├── dataset-xxxxxx.tar
├── metadata.parquet
└── aerial/
    ├── aerial-xxxxxx.tar
    ├── aerial_metadata.parquet
    └── ground_metadata_linked.parquet
```

`aerial/aerial_to_grounds.json` is not written by this script; run `scripts/many_to_one.py` to produce it.

```text
<country>/<city>/
├── labels.json 
└── ground/ # ground view images
└── aerial/ # aerial view images
```

<!-- ### Example (Oslo)
```bash
python3 scripts/sharding_script.py \
  --city oslo \
  --labels_path NO/oslo/labels.json \
  --ground_dir NO/oslo/ground \
  --aerial_dir NO/oslo/aerial \
  --output_dir NO/oslo
``` -->

---

## `scripts/visual_qa.py`

This script generates visual QA pairs by utilizing physical UTM offsets to verify the alignment between aerial and ground views. It reads each city's `metadata.parquet` (which already carries the aerial link; `aerial/ground_metadata_linked.parquet` is accepted as an optional equivalent), the ground's per-image JSON from its shard, and the aerial patch image plus metadata from the aerial shard, keyed by `aerial_id`. Using the assigned patch — rather than the ground's own provider crop — keeps the dot, the GSD, and the offsets on the same image center, matching the reference pipeline. It does not read `labels.json`.

### Usage

```bash
python3 scripts/visual_qa.py \
  --metadata_path <path_to_metadata.parquet> \
  [--linked_metadata_path <path_to_ground_metadata_linked.parquet>] \
  [--ground_dir <path_to_ground_images>] \
  [--aerial_dir <path_to_aerial_images>]
```

### Arguments
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--country_dir` | Process every city in this directory that has a `metadata.parquet`. | `None` |
| `--metadata_path` | Path to a city's `metadata.parquet` (single-city mode). | **Required** unless `--country_dir` |
| `--linked_metadata_path` | Path to `aerial/ground_metadata_linked.parquet`. | `<metadata dir>/aerial/ground_metadata_linked.parquet` |
| `--city` | Name of the city for output paths. | Metadata directory name |
| `--ground_dir` | Directory containing ground-level images. | `<metadata dir>/ground` |
| `--aerial_dir` | Directory containing aerial images. | `<metadata dir>/aerial` |
| `--image_id` | Process only a specific ground ID. | `None` |
| `--limit` | Maximum number of samples to generate per city. | `5` |
| `--seed` | Fixed seed for the sampling shuffle; the same records are chosen every run. | `1` |

### Running
#### Example (Country-wide processing)
Generate visual QA pairs for all cities within a country directory (run sharding first):
```bash
python3 scripts/visual_qa.py --country_dir NO/
```

#### Per city example
```bash
python3 scripts/visual_qa.py --metadata_path NO/oslo/metadata.parquet
```

Outputs one `pair_<ground_id>.jpg` per record — the ground view on the left and the annotated aerial patch on the right, with latitude (N), longitude (E), and heading labelled in the aerial's top-left — in `<city>/aerial/samples_preview/`. Sampling shuffles the metadata rows with a fixed seed first, so `--limit` selects the same spread-out records on every run.

# Output Dataset

## High-Level Structure

The dataset is organized to support efficient many-to-one matching between ground-level imagery and aerial patches.

```text
<city>/
├── dataset-xxxxxx.tar                   # Ground-level image shards
├── metadata.parquet                     # Ground↔aerial linked index
└── aerial/
    ├── aerial-xxxxxx.tar                # Aerial image shards
    ├── aerial_metadata.parquet          # Index for aerial patches
    ├── ground_metadata_linked.parquet   # Same linked index as metadata.parquet
    ├── aerial_to_grounds.json           # Mapping: Aerial ID → [Ground IDs]
    └── samples_preview/                 # Visual alignment examples
```

---

## Data Formats

### 1. Sharded `.tar` Files
To avoid the "millions of small files" problem on disk, the dataset is sharded into large `.tar` archives.

#### Ground Shards (`dataset-XXXXXX.tar`)
Each archive contains a batch of ground-level images and their associated metadata:
- **`[image_id].png`**: The ground-level photograph.
- **`[image_id].json`**: A JSON file containing specific metadata for that image, including its GPS coordinates and recomputed UTM offsets relative to its aerial anchor.

#### Aerial Shards (`aerial-XXXXXX.tar`)
These contain the overhead patches associated with the ground images:
- **`[aerial_id].png`**: The aerial image patch.
- **`[aerial_id].json`**: Metadata for the aerial patch, including its center coordinates.

---

## 📊 Metadata Specifications

### `metadata.parquet`
This is the primary index for the ground-level dataset. It allows for fast querying and filtering without extracting the `.tar` files. It is the full ground↔aerial linked index, matching the reference Burlington `aerial/ground_metadata_linked.parquet`: one row per ground image, carrying both the ground record and its aerial association.

| Column | Type | Description |
| :--- | :--- | :--- |
| `ground_id` | String | Unique identifier for the ground-level image. |
| `shard` | String | Tar file the ground image was written to. |
| `is_panorama` / `has_label` | Bool | Panorama flag and label-presence flag. |
| `mapil_sequence_id` / `quality_score` / `captured_at` | — | Source Mapillary fields, when present in the labels. |
| `lat` / `lon` | Float | GPS coordinates of the image capture point. |
| `heading` | Float | Camera orientation (compass heading). |
| `aerial_id` | String | Aerial patch this ground image is linked to. |
| `osm_id` / `text_caption` | — | Source OpenStreetMap fields, when present. |
| `has_aerial` | Boolean | `True` when the linked aerial patch image was written. |
| `offset_east_m` / `offset_north_m` | Float | Ground position relative to the aerial patch center, in meters. Recomputed in UTM as the ground point minus the patch center, rounded to centimeters. |

The ground pose is the raw Mapillary pose (matching the OpenCVL construction, where the camera is within ±20 m of the aerial patch center); OpenSfM fields are used only when raw fields are missing. That pose supplies the ground lat/lon, heading, and the `dx`/`dy * resolution` displacement used to recover each patch's center. Heading is the raw heading except for frames flagged `heading_mapilraw_before_meridian_is_zero`, where the raw value was unset and the OpenSfM heading is used instead. The offset columns themselves are recomputed from UTM coordinates using the ground point's zone and hemisphere.

The same rows and columns are also written to `aerial/ground_metadata_linked.parquet`, so either path can be used downstream.

### `aerial_to_grounds.json`
Maps a single aerial patch ID to a list of all ground-level images that fall within that patch's footprint.

**Example Format:**
```json
{
  "aerial_ID_123": ["ground_ID_A", "ground_ID_B", "ground_ID_C"],
  "aerial_ID_456": ["ground_ID_D", "ground_ID_E"]
}
```

---

## ⚙️ Logical Workflow
1. **Lookup:** Use `metadata.parquet` (or the identical `aerial/ground_metadata_linked.parquet`) for the `ground_id`, coordinates, heading, offsets, `shard`, and `aerial_id`.
2. **Retrieve:** Extract the `.png` and `.json` from the specified `.tar` file.
3. **Align:** Use the `aerial_id` from the metadata to find the matching aerial patch in the `aerial/` directory.
4. **Project:** Apply the UTM offsets found in the ground image's `.json` to precisely locate the ground-view point within the aerial patch.