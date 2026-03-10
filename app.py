import streamlit as st
import tempfile
import zipfile
from pathlib import Path
import time
import shutil
import pandas as pd
from insightface_master import run_pipeline

st.set_page_config(page_title="Golden Ratio Tool", layout="wide")

st.title("Golden Ratio Tool")
st.caption("Upload a dataset ZIP, configure options in the sidebar, and click Run Analysis.")

# --------------------------
# Sidebar Options
# --------------------------

with st.sidebar:
    st.header("⚙️ Pipeline Options")

    st.subheader("Output Settings")
    per_person_only = st.checkbox(
        "Per-person only",
        value=False,
        help="Write only consolidated & summary CSVs (skip per-image master CSV)"
    )
    golden_score_scale = st.number_input(
        "Golden score scale",
        value=100.0,
        step=1.0,
        help="Scale factor for phi_closeness median → golden_ratio_score_0_100"
    )

    st.subheader("Landmark Settings")
    landmarks = st.selectbox(
        "Landmarks",
        options=["auto", "68", "106"],
        index=0,
        help="Preferred landmark model: 68-point if available, else 106-point. 'auto' chooses automatically."
    )

    st.subheader("Detection Settings")
    det_sizes = st.text_input(
        "Detection sizes",
        value="480,640",
        help="Comma-separated list of detection sizes (e.g. '384,480,640,1024') or 'auto'. Larger sizes catch smaller faces."
    )
    det_policy = st.selectbox(
        "Detection policy",
        options=["first-hit", "best"],
        index=0,
        help="'first-hit': stop on first good scale (faster). 'best': evaluate all scales and pick the best result."
    )
    det_reuse = st.checkbox(
        "Reuse detector",
        value=False,
        help="Reuse detector instances per size across images (faster startup for large datasets)"
    )
    min_face_area_pct = st.slider(
        "Min face area %",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help="Require face bounding box area / image area ≥ this value (0–1). Filters out very small/distant faces."
    )
    min_det_score = st.slider(
        "Min detection score",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.01,
        help="Minimum detection confidence score to keep a face (0–1). Lower = more permissive."
    )
    ctx_id = st.selectbox(
        "Compute device",
        options=[-1, 0, 1, 2],
        index=1,
        format_func=lambda x: "CPU" if x == -1 else f"GPU {x}",
        help="GPU ID to use for inference. Select CPU to force CPU-only mode."
    )

    st.subheader("Frontality Filter")
    frontality_filter = st.checkbox(
        "Enable frontality filter",
        value=False,
        help="Drop non-frontal faces based on yaw/pitch/roll angle thresholds"
    )
    if frontality_filter:
        yaw_max = st.slider("Max yaw (°)", 0.0, 90.0, 50.0, help="Maximum absolute yaw angle to keep (degrees). Lower = stricter frontal requirement.")
        pitch_max = st.slider("Max pitch (°)", 0.0, 90.0, 40.0, help="Maximum absolute pitch angle to keep (degrees). Lower = stricter frontal requirement.")
        roll_max = st.slider("Max roll (°)", 0.0, 90.0, 45.0, help="Maximum absolute roll angle to keep (degrees). Lower = stricter frontal requirement.")
        keep_rejected = st.checkbox(
            "Keep rejected rows",
            value=False,
            help="Instead of dropping filtered faces, keep them in CSV with a 'filter_reason' column"
        )
    else:
        yaw_max, pitch_max, roll_max, keep_rejected = 50.0, 40.0, 45.0, False

    st.subheader("Debug / Extra")
    save_overlays = st.checkbox(
        "Save annotated overlays",
        value=False,
        help="Save annotated face overlay images (JPG) under output_dir/overlays"
    )
    save_landmarks = st.checkbox(
        "Save landmarks JSON",
        value=False,
        help="Dump per-image landmark/keypoint data as JSON under output_dir/landmarks"
    )
    print_found = st.checkbox(
        "Print found people",
        value=False,
        help="Print discovered people and image counts to terminal; also logs per-image detection size"
    )
    anime_weights = st.text_input(
        "Anime weights path",
        value="",
        help="(Optional) Path to a YOLO anime-face weights .pt file to enable cartoon/anime face fallback detection"
    )

# --------------------------
# Instructions
# --------------------------

with st.expander("ℹ️ Instructions & ZIP Structure"):
    st.markdown("""
### Expected ZIP Structure
```
my_dataset.zip
└── faces/
    ├── person1/
    │   ├── img1.jpg
    │   └── img2.png
    └── person2/
        └── img1.jpg
```
**Requirements:**
- Each subfolder = one person
- All person folders must be inside `faces/`
- Supported formats: `.jpg`, `.jpeg`, `.png`

**Tips:**
- Avoid special characters in folder names
""")

# --------------------------
# Session State
# --------------------------

if "results" not in st.session_state:
    st.session_state.results = None

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

# --------------------------
# Upload
# --------------------------

uploaded_zip = st.file_uploader("Upload Dataset ZIP", type=["zip"])

if uploaded_zip and uploaded_zip.name != st.session_state.last_uploaded_name:
    st.session_state.results = None
    st.session_state.last_uploaded_name = uploaded_zip.name

# --------------------------
# Run Button
# --------------------------

if uploaded_zip:

    if st.button("▶ Run Facial Analysis", use_container_width=True, type="primary"):

        temp_root = Path(tempfile.mkdtemp())
        input_dir = temp_root / "input"
        output_dir = temp_root / "output"

        input_dir.mkdir()
        output_dir.mkdir()

        zip_path = temp_root / "dataset.zip"

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(input_dir)
        except zipfile.BadZipFile:
            st.error("Invalid ZIP file.")
            shutil.rmtree(temp_root)
            st.stop()

        root_for_pipeline = input_dir / "faces"

        if not root_for_pipeline.exists():
            st.error("ZIP must contain a 'faces/' directory at the top level.")
            shutil.rmtree(temp_root)
            st.stop()

        extra_options = {
            "det_sizes": det_sizes,
            "ctx_id": ctx_id,
            "save_overlays": save_overlays,
            "save_landmarks": save_landmarks,
            "print_found": print_found,
            "det_policy": det_policy,
            "det_reuse": det_reuse,
            "min_face_area_pct": min_face_area_pct,
            "min_det_score": min_det_score,
            "landmarks": landmarks,
            "per_person_only": per_person_only,
            "golden_score_scale": golden_score_scale,
            "frontality_filter": frontality_filter,
            "yaw_max": yaw_max,
            "pitch_max": pitch_max,
            "roll_max": roll_max,
            "keep_rejected": keep_rejected,
        }
        if anime_weights.strip():
            extra_options["anime_weights"] = anime_weights.strip()

        try:
            with st.spinner("Running pipeline..."):
                start = time.time()
                run_pipeline(
                    root_path=str(root_for_pipeline),
                    outdir=str(output_dir),
                    extra_options=extra_options
                )
                elapsed = round(time.time() - start, 2)

            st.session_state.results = {
                "output_dir": str(output_dir),
                "time": elapsed
            }

            st.success(f"✅ Processing complete in {elapsed}s")

        except Exception as e:
            st.error(f"Processing failed: {e}")
            shutil.rmtree(temp_root)
            st.stop()

# --------------------------
# Results Display
# --------------------------

if st.session_state.results:

    output_dir = Path(st.session_state.results["output_dir"])

    consolidated_path = output_dir / "if_consolidated.csv"
    summary_path = output_dir / "if_summary_by_person.csv"
    master_path = output_dir / "if_master.csv"

    st.divider()

    # --- Summary Table ---
    if consolidated_path.exists():
        df_consolidated = pd.read_csv(consolidated_path)

        st.subheader("Summary by Person")

        summary_cols = ["person", "n_images", "n_used", "n_errors", "n_rejected", "shape_guess_mode", "golden_ratio_score_0_100"]
        available_cols = [c for c in summary_cols if c in df_consolidated.columns]
        missing_cols = [c for c in summary_cols if c not in df_consolidated.columns]

        if missing_cols:
            st.caption(f"Note: columns not found in output: `{'`, `'.join(missing_cols)}`")

        if available_cols:
            df_display = df_consolidated[available_cols].copy()

            # Style the dataframe
            styled = df_display.style.format({
                "golden_ratio_score_0_100": "{:.2f}",
                "n_images": "{:.0f}",
                "n_used": "{:.0f}",
                "n_errors": "{:.0f}",
                "n_rejected": "{:.0f}",
            }, na_rep="—")

            if "golden_ratio_score_0_100" in df_display.columns:
                styled = styled.background_gradient(
                    subset=["golden_ratio_score_0_100"],
                    cmap="RdYlGn",
                    vmin=0, vmax=100
                )

            st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Downloads ---
    st.subheader("⬇️ Download CSVs")
    col1, col2, col3 = st.columns(3)

    if consolidated_path.exists():
        with open(consolidated_path, "rb") as f:
            col1.download_button("📄 Consolidated CSV", f, file_name="if_consolidated.csv", use_container_width=True)

    if summary_path.exists():
        with open(summary_path, "rb") as f:
            col2.download_button("📄 Summary CSV", f, file_name="if_summary_by_person.csv", use_container_width=True)

    if master_path.exists():
        with open(master_path, "rb") as f:
            col3.download_button("📄 Master CSV", f, file_name="if_master.csv", use_container_width=True)

    # --- Full CSV Viewer ---
    st.divider()
    st.subheader("🔍 Data Explorer")

    tabs_config = []
    if consolidated_path.exists():
        tabs_config.append(("Consolidated", consolidated_path))
    if summary_path.exists():
        tabs_config.append(("Summary by Person", summary_path))
    if master_path.exists():
        tabs_config.append(("Master (per image)", master_path))

    if tabs_config:
        tab_objects = st.tabs([label for label, _ in tabs_config])

        for tab, (label, path) in zip(tab_objects, tabs_config):
            with tab:
                try:
                    df = pd.read_csv(path)
                    st.caption(f"{len(df)} rows × {len(df.columns)} columns")

                    # Search/filter
                    if "person" in df.columns:
                        search = st.text_input(
                            "Filter by person name",
                            key=f"search_{label}",
                            placeholder="Type to filter..."
                        )
                        if search:
                            df = df[df["person"].str.contains(search, case=False, na=False)]

                    st.dataframe(df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Could not load {label}: {e}")
    else:
        st.info("No output CSV files found.")