import streamlit as st
import pandas as pd
from PIL import Image
import io
import os
import tarfile
import json
from datetime import datetime

def load_data(file_path):
    """Loads the parquet file into a pandas DataFrame."""
    if not os.path.exists(file_path):
        st.error(f"File {file_path} not found. Please upload a parquet file.")
        return None
    return pd.read_parquet(file_path)

def load_image_from_tar(tar_path, member_name):
    """
    Extracts a single image file from a tar archive without extracting the whole archive.
    """
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # Ensure the member name ends with .jpg as per dataset convention
            if not member_name.endswith('.jpg'):
                member_name += '.jpg'

            member = tar.getmember(member_name)
            f = tar.extractfile(member)
            if f:
                return f.read()
    except Exception as e:
        st.error(f"Error extracting {member_name} from tar: {e}")
    return None

def resolve_ground_images(aerial_id, mapping_path):
    """
    Looks up the ground-view image IDs associated with a specific aerial image.
    """
    try:
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            return mapping.get(aerial_id, [])
    except Exception as e:
        st.error(f"Error loading mapping file: {e}")
        return []

def process_image(img_data):
    """Processes image data (bytes or path) into a PIL Image object."""
    try:
        if isinstance(img_data, bytes):
            return Image.open(io.BytesIO(img_data))
        elif isinstance(img_data, str) and os.path.exists(img_data):
            return Image.open(img_data)
    except Exception as e:
        st.error(f"Error loading image: {e}")
    return None

def save_score(output_csv, image_id, score, notes, view_type):
    """
    Saves the quality score and notes to a CSV file.
    Schema: image_id, score, notes, view_type, timestamp
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_df = pd.DataFrame([{
        "image_id": image_id,
        "score": score,
        "notes": notes,
        "view_type": view_type,
        "timestamp": timestamp
    }])

    file_exists = os.path.isfile(output_csv)
    result_df.to_csv(output_csv, mode='a', index=False, header=not file_exists)
