# cvgl_reu

Tooling for building dataset & vis. verification tool.

## Repo layout

```text
.
├── data_processing/     # Dataset build pipeline
│   ├── scripts/
│   │   ├── many_to_one.py      # Build aerial → [ground] associations
│   │   ├── sharding_script.py  # Shard images, write Parquet
│   │   └── visual_qa.py        # Generate visual alignment QA pairs
│   ├── architecture.md         # Output format specification
│   └── README.md               # Usage and CLI reference
└── visualizer/          # Streamlit visualization tool
```

## Quick start

Requires Python 3.8+.

### Data processing
```bash
cd data_processing
pip install -r requirements.txt

python3 scripts/many_to_one.py --help
python3 scripts/sharding_script.py --country_dir <country>/
python3 scripts/visual_qa.py --country_dir <country>/
```

### Visualization tool
``` bash
cd visualizer
pip install -r requirements.txt
streamlit run app.py
```

See [`data_processing/README.md`](data_processing/README.md) for full CLI arguments and [`data_processing/architecture.md`](data_processing/architecture.md) for the output schema.

## Input layout

Each city is expected as:

```text
<country>/<city>/
├── labels.json      # provider pairings and pose metadata
├── ground/          # ground-level images
└── aerial/          # aerial patch images
```

## Output layout

```text
<country>/<city>/
├── dataset-xxxxxx.tar                  # ground image shards
├── metadata.parquet                    # ground ↔ aerial linked index
└── aerial/
    ├── aerial-xxxxxx.tar               # aerial patch shards
    ├── aerial_metadata.parquet
    ├── ground_metadata_linked.parquet
    ├── aerial_to_grounds.json
    └── samples_preview/                # visual QA examples
```

## Data

Imagery is not included in this repository; large datasets and PDFs are tracked outside version control (see `.gitignore`). Point the scripts at your own country/city directories.
