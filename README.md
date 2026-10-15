# Ground-view Image Visualizer

Internal visualization tool built with Streamlit for image dataset. This tool allows researchers to visually inspect images, view associated metadata, and verify quality scores to assist in dataset building.

*Note that dataset is not included in repo as it is very large and should be added to folder separately.*

**Right now, this tool only shows the sample images as placeholders until we are able to serve up the actual images with the necessary metadata alongside**

## 🎯 Purpose

The Image Quality Visualizer is designed to bridge the gap between raw data storage (Parquet) and human verification. Instead of manually browsing folders, users can:
- Quickly cycle through thousands of images.
- View metadata alongside each image. (**TODO**)
- Log quality scores (0.0 [poor] - 1.0 [excellent]) to CSV

## ⚙️ Dependencies

The project uses the following primary libraries:
- **Streamlit**: For the web-based user interface.
- **Pandas**: For data manipulation and Parquet handling.
- **PyArrow**: The engine used by Pandas to read Parquet files.
- **Pillow (PIL)**: For image processing and rendering.

Full dependencies are listed in `requirements.txt`.

## 🚀 How to Run

### 1. Prerequisites
Requires Python 3.8+.

#### Set up virtual environment (not necessary, but nice to have)

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies
Install the required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Prepare Data
Ensure your data is available in the root directory:
- A file named `data.parquet` (or update the `PARQUET_FILE` variable in `app.py`).
- If your images are stored as paths, ensure those paths are accessible from the machine running the app.

### 4. Launch the Application
Run the Streamlit server:

```bash
streamlit run app.py
```

The app will automatically open in your default web browser (usually at `http://localhost:8501`).

## 🛠️ Configuration

You can customize the behavior of the tool by editing the configuration section at the top of `app.py`:

- `PARQUET_FILE`: Path to your input data.
- `IMAGE_COLUMN`: The column name in your Parquet file containing the images (bytes or file paths).
- `METADATA_COLS`: A list of column names you wish to display in the metadata sidebar.
- `OUTPUT_CSV`: The filename where quality scores will be saved.
