#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for Editorial Pose Transformation Pipeline.

Run:
  set REPLICATE_API_TOKEN=r8_...
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from pose_transform_gemini import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_UPSCALE_WORKERS,
    PIPELINE_MODES,
    EditorialPoseTransformer,
    get_replicate_api_key,
    output_folder_for_mode,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]

MODE_LABELS = {
    "upscale": "1 · Upscale (4K → *_1.jpg)",
    "pose": "2 · Pose change (2K → *_2.jpg)",
    "upscale_pose_bg": "3 · Upscale + Pose/BG + Weird Pose (4K)",
    "both": "4 · Both — Pose/BG + Weird Pose → output_both/",
    "upscale_pose": "5 · Upscale + Pose (4K → *_1 + *_2)",
}

MODE_HELP = {
    "upscale": "Nano Banana 2 upscale only — no LLM. Writes stem_1.jpg.",
    "pose": "Same person/outfit/bg, invent weird pose. Writes stem_2.jpg.",
    "upscale_pose_bg": "Donor folder for BG+pose → _1, then weird pose on _1 → _2 @ 4K.",
    "both": "Same as mode 3, outputs go to output_both/.",
    "upscale_pose": "Upscale → _1, then weird pose on _1 → _2 (shared folder).",
}


def _needs_donors(mode: str) -> bool:
    return mode in ("upscale_pose_bg", "both")


def _default_workers(mode: str) -> int:
    if mode in ("upscale", "upscale_pose_bg", "both", "upscale_pose"):
        return DEFAULT_UPSCALE_WORKERS
    return DEFAULT_MAX_WORKERS


def _safe_filename(name: str) -> str:
    stem = Path(name).stem
    suffix = Path(name).suffix.lower() or ".jpg"
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return f"{cleaned or 'image'}{suffix}"


def _save_uploads(files, dest: Path) -> List[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for uploaded in files:
        path = dest / _safe_filename(uploaded.name)
        # Avoid overwrite collisions within the same batch
        if path.exists():
            path = dest / f"{path.stem}_{uuid.uuid4().hex[:6]}{path.suffix}"
        path.write_bytes(uploaded.getvalue())
        saved.append(path)
    return saved


def _collect_outputs(result_item: Dict) -> List[Path]:
    paths: List[Path] = []
    for key in ("output_image_1", "output_image_2", "output_image"):
        value = result_item.get(key)
        if value and Path(value).is_file() and Path(value) not in paths:
            paths.append(Path(value))
    return paths


def _zip_folder(folder: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(folder).as_posix())
    buffer.seek(0)
    return buffer.getvalue()


def _new_run_dirs(mode: str) -> Tuple[Path, Path, Path]:
    run_id = datetime_run_id()
    root = RUNS_DIR / run_id
    input_dir = root / "input"
    donor_dir = root / "donors"
    output_dir = root / output_folder_for_mode(mode)
    input_dir.mkdir(parents=True, exist_ok=True)
    donor_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, donor_dir, output_dir


def datetime_run_id() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def _show_result_gallery(items: List[Dict]) -> None:
    if not items:
        return

    for item in items:
        name = Path(item.get("input_image", "image")).name
        ok = item.get("success")
        skipped = item.get("skipped")
        status = "skipped" if skipped else ("ok" if ok else "failed")
        st.markdown(f"**{name}** — `{status}`")

        if item.get("shot_description"):
            st.caption(item["shot_description"])
        if item.get("error"):
            st.error(item["error"])

        outputs = _collect_outputs(item)
        if outputs:
            cols = st.columns(min(len(outputs), 3))
            for col, path in zip(cols, outputs):
                with col:
                    st.image(str(path), caption=path.name, use_container_width=True)
                    st.download_button(
                        label=f"Download {path.name}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="image/jpeg",
                        key=f"dl_{path}",
                    )
        st.divider()


def run_pipeline(
    *,
    mode: str,
    input_paths: List[Path],
    donor_paths: Optional[List[Path]],
    embed_metadata: bool,
    max_workers: int,
) -> Dict:
    input_dir, donor_dir, output_dir = _new_run_dirs(mode)

    for src in input_paths:
        shutil.copy2(src, input_dir / src.name)

    donor_folder: Optional[str] = None
    if _needs_donors(mode):
        if not donor_paths or len(donor_paths) < 2:
            raise ValueError("Modes 3/4 need at least 2 donor images (BG + pose).")
        for src in donor_paths:
            shutil.copy2(src, donor_dir / src.name)
        donor_folder = str(donor_dir)

    transformer = EditorialPoseTransformer(
        mode=mode,
        max_workers=max_workers,
        embed_metadata=embed_metadata,
    )

    if len(input_paths) == 1 and not _needs_donors(mode):
        # Single-image fast path (no donor pairing)
        item = transformer.process_image(str(input_dir / input_paths[0].name), str(output_dir))
        if mode in ("pose", "upscale_pose"):
            transformer.save_descriptions_json(str(output_dir), [item])
        results = {
            "total": 1,
            "success": 1 if item.get("success") else 0,
            "failed": 0 if item.get("success") else 1,
            "items": [item],
            "output_folder": str(output_dir),
        }
        return results

    if len(input_paths) == 1 and _needs_donors(mode):
        # Single subject + donors: still use run() so pairing logic is shared
        results = transformer.run(
            str(input_dir),
            str(output_dir),
            donor_folder=donor_folder,
        )
        results["output_folder"] = str(output_dir)
        return results

    results = transformer.run(
        str(input_dir),
        str(output_dir),
        donor_folder=donor_folder,
    )
    results["output_folder"] = str(output_dir)
    return results


def main() -> None:
    st.set_page_config(
        page_title="Pose Transform",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Editorial Pose Transform")
    st.caption("Gemini 3 Flash + Nano Banana 2 via Replicate — single image or batch.")

    api_key = get_replicate_api_key()
    with st.sidebar:
        st.header("Settings")
        if api_key:
            st.success("Replicate key loaded from env")
            st.caption("REPLICATE_API_TOKEN / REPLICATE_API_KEY")
        else:
            st.error("Missing Replicate API key")
            st.caption("Set REPLICATE_API_TOKEN or REPLICATE_API_KEY, then restart Streamlit.")

        mode = st.selectbox(
            "Pipeline mode",
            options=list(PIPELINE_MODES),
            format_func=lambda m: MODE_LABELS.get(m, m),
            index=list(PIPELINE_MODES).index("pose"),
            help=MODE_HELP.get("pose", ""),
        )
        st.info(MODE_HELP.get(mode, ""))

        embed_metadata = st.checkbox(
            "Embed camera EXIF (Nikon D7500-style)",
            value=False,
        )
        max_workers = st.slider(
            "Max parallel workers",
            min_value=1,
            max_value=40,
            value=min(8, _default_workers(mode)),
            help="Lower this if you hit Replicate rate limits.",
        )

        run_style = st.radio(
            "Run style",
            options=("single", "batch"),
            format_func=lambda x: "Single image" if x == "single" else "Batch (many images)",
            horizontal=True,
        )

    if not api_key:
        st.warning("Configure REPLICATE_API_TOKEN in the environment before running.")
        return

    needs_donors = _needs_donors(mode)

    if run_style == "single":
        st.subheader("Single image")
        subject = st.file_uploader(
            "Subject image",
            type=IMAGE_TYPES,
            accept_multiple_files=False,
            key="single_subject",
        )
        donor_files = None
        if needs_donors:
            st.markdown("**Donor images** (BG + pose — at least 2)")
            donor_files = st.file_uploader(
                "Donors",
                type=IMAGE_TYPES,
                accept_multiple_files=True,
                key="single_donors",
            )

        if subject:
            st.image(subject, caption=subject.name, width=280)

        run = st.button("Run single", type="primary", disabled=subject is None)
        if run and subject is not None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                subject_paths = _save_uploads([subject], tmp_path / "in")
                donors: Optional[List[Path]] = None
                if needs_donors:
                    if not donor_files or len(donor_files) < 2:
                        st.error("Upload at least 2 donor images for this mode.")
                        return
                    donors = _save_uploads(donor_files, tmp_path / "donors")

                with st.spinner("Running pipeline… this can take a few minutes"):
                    try:
                        results = run_pipeline(
                            mode=mode,
                            input_paths=subject_paths,
                            donor_paths=donors,
                            embed_metadata=embed_metadata,
                            max_workers=1,
                        )
                    except Exception as exc:
                        st.exception(exc)
                        return

                st.session_state["last_results"] = results

    else:
        st.subheader("Batch")
        subjects = st.file_uploader(
            "Subject images",
            type=IMAGE_TYPES,
            accept_multiple_files=True,
            key="batch_subjects",
        )
        donor_files = None
        if needs_donors:
            st.markdown("**Donor images** (shared pool — at least 2; ideally 2× subject count)")
            donor_files = st.file_uploader(
                "Donors",
                type=IMAGE_TYPES,
                accept_multiple_files=True,
                key="batch_donors",
            )

        if subjects:
            st.caption(f"{len(subjects)} subject image(s) selected")
            preview_cols = st.columns(min(4, len(subjects)))
            for col, uploaded in zip(preview_cols, subjects[:4]):
                with col:
                    st.image(uploaded, caption=uploaded.name, use_container_width=True)

        run = st.button(
            "Run batch",
            type="primary",
            disabled=not subjects,
        )
        if run and subjects:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                subject_paths = _save_uploads(subjects, tmp_path / "in")
                donors: Optional[List[Path]] = None
                if needs_donors:
                    if not donor_files or len(donor_files) < 2:
                        st.error("Upload at least 2 donor images for this mode.")
                        return
                    donors = _save_uploads(donor_files, tmp_path / "donors")

                progress = st.progress(0, text="Starting batch…")
                with st.spinner(f"Processing {len(subject_paths)} images…"):
                    try:
                        results = run_pipeline(
                            mode=mode,
                            input_paths=subject_paths,
                            donor_paths=donors,
                            embed_metadata=embed_metadata,
                            max_workers=max_workers,
                        )
                        progress.progress(1.0, text="Batch complete")
                    except Exception as exc:
                        progress.empty()
                        st.exception(exc)
                        return

                st.session_state["last_results"] = results

    results = st.session_state.get("last_results")
    if not results:
        return

    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", results.get("total", 0))
    c2.metric("Success", results.get("success", 0))
    c3.metric("Failed", results.get("failed", 0))
    out_folder = results.get("output_folder")
    if out_folder:
        c4.caption(f"Saved under\n`{Path(out_folder).name}`")

    items = results.get("items") or []
    _show_result_gallery(items)

    if out_folder and Path(out_folder).is_dir():
        zip_bytes = _zip_folder(Path(out_folder))
        st.download_button(
            label="Download all outputs (ZIP)",
            data=zip_bytes,
            file_name=f"{Path(out_folder).name}.zip",
            mime="application/zip",
            key="zip_all",
        )


if __name__ == "__main__":
    main()
