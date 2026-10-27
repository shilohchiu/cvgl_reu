# Dataset Processing

Builds the ground-view ↔ aerial dataset: associate each ground image with an aerial patch, then shard and index it. For the full output schema, see [`architecture.md`](architecture.md).

## Install

```bash
pip install -r requirements.txt
```

## Pipeline

Run in order:

1. **`scripts/many_to_one.py`** — builds `aerial/aerial_to_grounds.json` (aerial patch → ground IDs).
2. **`scripts/sharding_script.py`** — validates and shards images into `.tar`, writes the Parquet indexes.
3. **`scripts/visual_qa.py`** — renders side-by-side ground/aerial alignment previews for QA.

## Input layout

```text
<country>/
└── <city>/
    ├── labels.json
    ├── ground/       # ground-view images
    └── aerial/       # aerial patch images
```

---

## `many_to_one.py`

```bash
python3 scripts/many_to_one.py --match anchors --country_dir NO/
```

`--match` controls how grounds are assigned: `labels` (provider pairing), `nearest` (nearest patch center), `anchors` (share the provider's one-crop-per-ground crops among the grounds they cover — the many-to-one path), `metadata` (reuse an existing linked Parquet), or `grid` (synthetic grid; **not** consumable by sharding). See `--help` for tuning flags (`--max_distance_m`, `--anchor_radius_m`, `--min_grounds`, …).

## `sharding_script.py`

Consumes `aerial/aerial_to_grounds.json` as authoritative and rejects synthetic `grid` keys. Validates that assigned grounds actually fall inside their patch, recomputes UTM offsets from the patch center, and writes size-bounded `.tar` shards plus Parquet indexes.

```bash
# All cities in a country directory
python3 scripts/sharding_script.py --country_dir NO/

# Single city
python3 scripts/sharding_script.py \
  --city oslo \
  --labels_path NO/oslo/labels.json \
  --ground_dir NO/oslo/ground \
  --aerial_dir NO/oslo/aerial \
  --output_dir NO/oslo \
  --max_shard_size 5
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--city` | City name (used for output naming). | **Required** |
| `--labels_path` | Path to `labels.json`. | **Required** |
| `--ground_dir` | Ground-view images. | **Required** |
| `--aerial_dir` | Aerial patch images. | **Required** |
| `--output_dir` | Base directory for the city and its shards. | **Required** |
| `--aerial_mapping` | Path to `aerial_to_grounds.json`. | `<aerial_dir>/aerial_to_grounds.json` |
| `--max_shard_size` | Max size per `.tar`, in GB. | `5` |

## `visual_qa.py`

Reads `metadata.parquet`, the per-image JSON from the ground shard, and the linked aerial patch, then writes annotated `pair_<ground_id>.jpg` previews (ground left, aerial right, with lat/lon/heading) to `<city>/aerial/samples_preview/`. Uses the assigned patch — not the ground's own provider crop — so the offsets and dot stay consistent. Sampling uses a fixed seed.

```bash
# All cities (run sharding first)
python3 scripts/visual_qa.py --country_dir NO/

# Single city
python3 scripts/visual_qa.py --metadata_path NO/oslo/metadata.parquet
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--country_dir` | Process every city with a `metadata.parquet`. | `None` |
| `--metadata_path` | A city's `metadata.parquet` (unless `--country_dir`). | **Required** |
| `--linked_metadata_path` | `aerial/ground_metadata_linked.parquet`. | `<metadata dir>/aerial/ground_metadata_linked.parquet` |
| `--city` | City name for output paths. | Metadata directory name |
| `--ground_dir` / `--aerial_dir` | Image directories. | `<metadata dir>/ground`, `<metadata dir>/aerial` |
| `--image_id` | Only this ground ID. | `None` |
| `--limit` | Samples per city. | `5` |
| `--seed` | Shuffle seed. | `1` |

---

## Output layout

```text
<country>/<city>/
├── dataset-xxxxxx.tar                  # ground image shards
├── metadata.parquet                    # ground ↔ aerial linked index
└── aerial/
    ├── aerial-xxxxxx.tar               # aerial patch shards
    ├── aerial_metadata.parquet         # one row per patch
    ├── ground_metadata_linked.parquet  # same rows as metadata.parquet
    ├── aerial_to_grounds.json
    └── samples_preview/                # visual QA outputs
```

Ground shards hold `<id>.png` + `<id>.json`; aerial shards hold each patch image plus its center coordinates, size, and associated ground IDs. `metadata.parquet` is the primary lookup table — see [`architecture.md`](architecture.md) for every column and the full alignment workflow.
