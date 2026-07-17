#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for Editorial Pose Transformation Pipeline.

Run:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from pose_transform_gemini import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_UPSCALE_WORKERS,
    PIPELINE_MODES,
    EditorialPoseTransformer,
    build_donor_pairs,
    output_folder_for_mode,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
IMAGE_EXTENSIONS = {f".{ext}" for ext in IMAGE_TYPES}

# Streamlit Cloud upload cap is 200 MB per file; batches run sequentially to avoid timeouts.
MAX_ZIP_UPLOAD_BYTES = 200 * 1024 * 1024
RECOMMENDED_MAX_BATCH_IMAGES = 30
HARD_MAX_BATCH_IMAGES = 100

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


def _is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _find_images_recursive(root: Path) -> List[Path]:
    """Resolve all images under root, including nested subfolders."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if _is_image_path(p))


def _safe_zip_member_path(name: str) -> Optional[str]:
    """Normalize ZIP entry paths and block path traversal."""
    normalized = Path(name.replace("\\", "/"))
    parts = [part for part in normalized.parts if part not in ("", ".")]
    if not parts or ".." in parts:
        return None
    if parts[0] == "__MACOSX":
        return None
    if any(part.startswith("._") for part in parts):
        return None
    return str(Path(*parts))


def _extract_images_from_zip(zip_bytes: bytes, dest: Path) -> List[Path]:
    """
    Extract image files from a ZIP, preserving subfolder layout.

    Works with archives like:
      input/alice.jpg
      input/set01/bob.png
      photos/2024/march/charlie.webp
    """
    dest.mkdir(parents=True, exist_ok=True)
    extracted = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            member = _safe_zip_member_path(info.filename)
            if not member:
                continue

            member_path = Path(member)
            if member_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            target = dest / member_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            extracted += 1

    images = _find_images_recursive(dest)
    if extracted == 0:
        raise ValueError("ZIP contains no supported image files (.jpg, .jpeg, .png, .bmp, .tiff, .webp).")
    if not images:
        raise ValueError("ZIP extracted files, but no readable images were found.")
    return images


def _validate_batch_size(count: int) -> None:
    if count <= 0:
        raise ValueError("No images found to process.")
    if count > HARD_MAX_BATCH_IMAGES:
        raise ValueError(
            f"Batch has {count} images — hard limit is {HARD_MAX_BATCH_IMAGES}. "
            "Split the ZIP into smaller batches."
        )


def _relative_display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


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
        name = item.get("input_display") or Path(item.get("input_image", "image")).name
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


def run_pipeline_sequential(
    *,
    mode: str,
    input_paths: List[Path],
    donor_paths: Optional[List[Path]],
    embed_metadata: bool,
    replicate_api_key: str,
    progress_callback=None,
) -> Dict:
    """Process images one by one (batch ZIP mode)."""
    _, _, output_dir = _new_run_dirs(mode)
    os.makedirs(output_dir, exist_ok=True)

    donor_pair_map: Dict[str, Dict[str, Optional[str]]] = {}
    if _needs_donors(mode):
        if not donor_paths or len(donor_paths) < 2:
            raise ValueError("Modes 3/4 need at least 2 donor images (BG + pose).")
        donor_pair_map = build_donor_pairs(
            [str(path) for path in input_paths],
            [str(path) for path in donor_paths],
        )

    transformer = EditorialPoseTransformer(
        replicate_api_key=replicate_api_key,
        mode=mode,
        max_workers=1,
        embed_metadata=embed_metadata,
    )

    items: List[Dict] = []
    total = len(input_paths)
    for index, path in enumerate(input_paths, start=1):
        if progress_callback:
            progress_callback(index, total, path)
        pair = donor_pair_map.get(str(path), {})
        item = transformer.process_image(
            str(path),
            str(output_dir),
            bg_source=pair.get("bg_source"),
            pose_source=pair.get("pose_source"),
        )
        item["input_display"] = path.name
        items.append(item)

    if mode in ("pose", "upscale_pose_bg", "both", "upscale_pose"):
        transformer.save_descriptions_json(str(output_dir), items)

    success = sum(1 for item in items if item.get("success"))
    return {
        "total": total,
        "success": success,
        "failed": total - success,
        "items": items,
        "output_folder": str(output_dir),
    }


def run_pipeline(
    *,
    mode: str,
    input_paths: List[Path],
    donor_paths: Optional[List[Path]],
    embed_metadata: bool,
    max_workers: int,
    replicate_api_key: str,
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
        replicate_api_key=replicate_api_key,
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

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "Replicate API key",
            type="password",
            placeholder="r8_...",
            help="Paste your Replicate API token. It is used for this session only.",
        ).strip()
        if api_key:
            st.success("API key entered")
        else:
            st.warning("Enter your Replicate API key to run")

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

        run_style = st.radio(
            "Run style",
            options=("single", "batch"),
            format_func=lambda x: "Single image" if x == "single" else "Batch (ZIP)",
            horizontal=True,
        )

        max_workers = st.slider(
            "Max parallel workers",
            min_value=1,
            max_value=40,
            value=min(8, _default_workers(mode)),
            help="Used for single-image runs only. Batch ZIP always runs one image at a time.",
            disabled=run_style == "batch",
        )

    if not api_key:
        st.info("Enter your Replicate API key in the sidebar to get started.")
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
                            replicate_api_key=api_key,
                        )
                    except Exception as exc:
                        st.exception(exc)
                        return

                st.session_state["last_results"] = results

    else:
        st.subheader("Batch (ZIP)")
        st.caption(
            "Upload a ZIP of subject images. Nested folders are supported "
            "(e.g. `input/set01/photo.jpg`). Images are processed **one by one**."
        )
        with st.expander("Batch limits & what to expect"):
            st.markdown(
                f"""
**Upload**
- Max ZIP size: **200 MB** (Streamlit upload limit)
- Recommended: **≤ {RECOMMENDED_MAX_BATCH_IMAGES} images** per batch on Streamlit Cloud
- Hard cap in this app: **{HARD_MAX_BATCH_IMAGES} images** per ZIP

**Timing (approx., one-by-one)**
- Mode 1 (upscale): ~1–2 min / image
- Mode 2 (pose): ~2–4 min / image
- Modes 3–5 (two outputs): ~4–8 min / image

**Practical max without breaking**
- **Streamlit Cloud free**: stay around **10–20 images** (session can time out on long runs)
- **Streamlit Cloud paid / local**: **30–50 images** is usually safe if you keep the tab open
- **100 images** is the app hard limit; beyond that, split into multiple ZIPs

**Replicate**
- Pipeline allows up to **40 parallel** API calls, but batch ZIP runs sequentially to stay stable
- Large batches mainly hit **time limits**, not Replicate rate limits
                """
            )

        subject_zip = st.file_uploader(
            "Subject images ZIP",
            type=["zip"],
            accept_multiple_files=False,
            key="batch_subject_zip",
        )
        donor_zip = None
        if needs_donors:
            st.markdown("**Donor images ZIP** (BG + pose pool — at least 2 images; nested folders OK)")
            donor_zip = st.file_uploader(
                "Donor images ZIP",
                type=["zip"],
                accept_multiple_files=False,
                key="batch_donor_zip",
            )

        preview_paths: List[Path] = []
        if subject_zip is not None:
            zip_size = len(subject_zip.getvalue())
            st.caption(f"ZIP size: {zip_size / (1024 * 1024):.1f} MB")
            if zip_size > MAX_ZIP_UPLOAD_BYTES:
                st.error("ZIP exceeds the 200 MB upload limit.")
            else:
                with tempfile.TemporaryDirectory() as preview_tmp:
                    try:
                        preview_paths = _extract_images_from_zip(
                            subject_zip.getvalue(),
                            Path(preview_tmp) / "preview",
                        )
                        st.success(f"Found **{len(preview_paths)}** image(s) in ZIP (including subfolders).")
                        if len(preview_paths) > RECOMMENDED_MAX_BATCH_IMAGES:
                            st.warning(
                                f"{len(preview_paths)} images is above the recommended "
                                f"{RECOMMENDED_MAX_BATCH_IMAGES}. The run may time out on Streamlit Cloud."
                            )
                        preview_lines = [
                            _relative_display_path(path, Path(preview_tmp) / "preview")
                            for path in preview_paths[:12]
                        ]
                        st.code("\n".join(preview_lines) + ("\n..." if len(preview_paths) > 12 else ""))
                    except Exception as exc:
                        st.error(str(exc))

        run = st.button(
            "Run batch",
            type="primary",
            disabled=subject_zip is None,
        )
        if run and subject_zip is not None:
            zip_bytes = subject_zip.getvalue()
            if len(zip_bytes) > MAX_ZIP_UPLOAD_BYTES:
                st.error("ZIP exceeds the 200 MB upload limit.")
                return

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                input_root = tmp_path / "in"
                try:
                    subject_paths = _extract_images_from_zip(zip_bytes, input_root)
                    _validate_batch_size(len(subject_paths))
                except Exception as exc:
                    st.error(str(exc))
                    return

                donors: Optional[List[Path]] = None
                if needs_donors:
                    if donor_zip is None:
                        st.error("Upload a donor ZIP with at least 2 images for this mode.")
                        return
                    donor_bytes = donor_zip.getvalue()
                    if len(donor_bytes) > MAX_ZIP_UPLOAD_BYTES:
                        st.error("Donor ZIP exceeds the 200 MB upload limit.")
                        return
                    try:
                        donors = _extract_images_from_zip(donor_bytes, tmp_path / "donors")
                    except Exception as exc:
                        st.error(f"Donor ZIP: {exc}")
                        return
                    if len(donors) < 2:
                        st.error("Donor ZIP must contain at least 2 images.")
                        return

                progress = st.progress(0, text="Starting batch…")
                status = st.empty()

                def _on_progress(current: int, total: int, path: Path) -> None:
                    rel = _relative_display_path(path, input_root)
                    progress.progress(current / total, text=f"Processing {current}/{total}: {rel}")
                    status.caption(f"Current: `{rel}`")

                with st.spinner(f"Processing {len(subject_paths)} images one by one…"):
                    try:
                        results = run_pipeline_sequential(
                            mode=mode,
                            input_paths=subject_paths,
                            donor_paths=donors,
                            embed_metadata=embed_metadata,
                            replicate_api_key=api_key,
                            progress_callback=_on_progress,
                        )
                        progress.progress(1.0, text="Batch complete")
                        status.caption("Done.")
                    except Exception as exc:
                        progress.empty()
                        status.empty()
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
