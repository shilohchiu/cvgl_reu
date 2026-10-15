import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components

# --- CONFIGURATION ---
SAMPLES_DIR = "burlington_vt/aerial/samples_preview"
OUTPUT_CSV = "quality_scores.csv"

from utils import save_score, process_image

def main():
    st.set_page_config(page_title="Image Quality Visualizer - Samples", layout="wide")

    # Inject JavaScript for Keyboard Navigation
    components.html(
        """
        <script>
        window.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                const buttons = Array.from(document.querySelectorAll('button'));
                const prevBtn = buttons.find(b => b.innerText.includes('Previous'));
                if (prevBtn) prevBtn.click();
            } else if (e.key === 'ArrowRight') {
                const buttons = Array.from(document.querySelectorAll('button'));
                const nextBtn = buttons.find(b => b.innerText.includes('Next'));
                if (nextBtn) nextBtn.click();
            }
        });
        </script>
        """,
        height=0,
    )

    # Load sample images
    try:
        sample_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith('.jpg')])
    except FileNotFoundError:
        sample_files = []

    if sample_files:
        # --- SESSION STATE ---
        if 'img_idx' not in st.session_state:
            st.session_state.img_idx = 0

        # --- UI LAYOUT ---
        st.title("Ground-view Image Visualizer")

        # Sidebar for navigation and stats
        with st.sidebar:
            st.header("Navigation")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Previous"):
                    st.session_state.img_idx = max(0, st.session_state.img_idx - 1)
            with col2:
                if st.button("Next ➡️"):
                    st.session_state.img_idx = min(len(sample_files) - 1, st.session_state.img_idx + 1)

            st.write(f"**Current Sample:** {st.session_state.img_idx + 1} / {len(sample_files)}")
            st.divider()
            st.write("### Progress")
            try:
                scores_df = pd.read_csv(OUTPUT_CSV)
                st.success(f"Scored: {len(scores_df)} samples")
            except FileNotFoundError:
                st.info("No scores saved yet.")

        # Main Display Area
        current_file = sample_files[st.session_state.img_idx]
        img_path = os.path.join(SAMPLES_DIR, current_file)
        main_col, meta_col = st.columns([3, 1])

        with main_col:
            st.subheader("Sample Preview")
            image = process_image(img_path)
            if image:
                st.image(image, use_container_width=True, caption=current_file)
            else:
                st.warning(f"Could not load image: {current_file}")

        with meta_col:
            st.subheader("Metadata")
            st.write(f"**Filename:** {current_file}")
            st.write("**Type:** Comparison Sample")

            st.divider()
            st.subheader("Quality Scoring")

            # Float scale: 0.0 (Poor) to 1.0 (Excellent)
            score = st.slider(
                "Rate Image Quality",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.01,
                help="0.0: Poor, 1.0: Excellent"
            )

            notes = st.text_area("Notes", placeholder="Optional comments...")

            if st.button("✅ Save Score", use_container_width=True):
                # Save using filename as ID
                save_score(OUTPUT_CSV, current_file, score, notes, "sample")
                st.toast("Score saved!", icon="💾")
                st.session_state.img_idx = min(len(sample_files) - 1, st.session_state.img_idx + 1)
                st.rerun()
    else:
        st.info(f"No sample images found in {SAMPLES_DIR}")

if __name__ == "__main__":
    main()
