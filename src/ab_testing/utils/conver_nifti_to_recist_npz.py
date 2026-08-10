#!/usr/bin/env python3
"""Self-contained single-file NIfTI -> RECIST NPZ converter (resumable).

Converts nifti cancer datasets to per-case npz. Each
dataset is windowed by its anatomical region (location-based windowing), then
per-lesion RECIST longest-diameter lines are computed. Runs are resumable:
cases whose npz already exists and opens cleanly (with the expected keys) are
skipped, so an interrupted run can simply be re-launched. Use --overwrite to
force every case to be rewritten.

Each npz is written to a temp file and atomically renamed into place, so a
killed run never leaves a truncated npz that would be mistaken for done.

Lesions whose RECIST longest diameter is shorter than MIN_RECIST_MM (10 mm, in
physical units from the image spacing) are excluded from BOTH 'gts' and
'recist' -- they are not sub-threshold segmentation targets without a prompt,
they are gone. On by default; disable with --no-exclude-small-lesions. Surviving
lesions keep their original cc3d ids, so ids may be non-contiguous.

--tumor-slices-only (off by default) crops every array along Z to the tumor
ROI: the contiguous span of axial slices holding label, optionally padded by
--tumor-slice-margin. This is the big size lever. The span is contiguous rather
than the exact set of labeled slices, so the result stays a real volume with a
meaningful z spacing; 'origin' is shifted to the new first slice so the npz is
still geometrically correct, and 'z_crop' records [z_start, z_stop) in the
original volume.

Whole-body datasets (Dataset011_WholeBody, Dataset5310) are intentionally NOT
included -- they are multi-region per case and need per-lesion bucketing.

Per-case NPZ contract:
  imgs      float16   (Z, Y, X)   CT windowed to [0, 255.0]
  gts       uint16  (Z, Y, X)   cc3d 26-connectivity instance labels
  recist    uint16  (Z, Y, X)   per-lesion longest-diameter line, value == lesion id
  spacing   float64 (3,)        sitk GetSpacing (x, y, z)
  direction float64 (9,)        sitk GetDirection
  origin    float64 (3,)        sitk GetOrigin (of slice z_crop[0] when cropped)
  reader    str                 'sitk' | 'nibabel_orthofix'
  z_crop    int64   (2,)        [z_start, z_stop) kept; only with --tumor-slices-only
(--no-recist skips the RECIST computation and omits the 'recist' key.)

Usage:
  python conver_nifti_to_recist_npz.py --all
  python conver_nifti_to_recist_npz.py --dataset Dataset003_LungCancer
  python conver_nifti_to_recist_npz.py --all --workers 8
  python conver_nifti_to_recist_npz.py --all --no-recist          # imgs/gts only
  python conver_nifti_to_recist_npz.py --all --overwrite          # ignore existing npz
  python conver_nifti_to_recist_npz.py --all --min-label-slices 0 # keep every case
  python conver_nifti_to_recist_npz.py --all --prune-filtered     # delete npz now filtered out
  python conver_nifti_to_recist_npz.py --all --no-exclude-small-lesions  # keep <10mm lesions
  python conver_nifti_to_recist_npz.py --all --tumor-slices-only  # crop Z to the tumor ROI
  python conver_nifti_to_recist_npz.py --all --tumor-slices-only --tumor-slice-margin 5

ab_testing notes:
    - img == scan
    - lbl == mask

"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cc3d
import cv2
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.spatial.distance import pdist, squareform

# We parallelize across cases with ProcessPoolExecutor; disable OpenCV's own
# internal thread pool so N workers don't oversubscribe cores (and to silence
# "Can't spawn new thread" (EAGAIN) on thread-capped login/compute nodes).
cv2.setNumThreads(0)

# ─── Constants ───────────────────────────────────────────────────────────────

MIN_LABEL_SLICES = 5            # exclude the case if the label spans fewer axial slices
MIN_RECIST_MM = 10.0            # drop lesions whose RECIST longest diameter is shorter (mm)
RECIST_LINE_THICKNESS = 2       # cv2.line thickness
MAX_CONTOUR_PTS = 500           # subsample cap before the O(n^2) pairwise distance
IMG_SUFFIX = "_0000.nii.gz"
ROOT_DEFAULT = "train_per_cancer_type"

# The 4 anatomical window buckets (level / width in HU).
WINDOW_BUCKET: dict[str, dict[str, int]] = {
    "lung":        {"level": -600, "width": 1500},
    "soft_tissue": {"level": 40, "width": 400},   # includes heart
    "bone":        {"level": 400, "width": 1800},
    "brain":       {"level": 40, "width": 80},
}

# dataset -> window bucket (location-based). Whole-body 011 / 5310 excluded.
DATASET_WINDOW: dict[str, str] = {
    "Dataset002_EsophagusCancer": "soft_tissue",
    "Dataset003_LungCancer": "lung",
    "Dataset004_LiverCancer": "soft_tissue",
    "Dataset005_AdrenalCancer": "soft_tissue",
    "Dataset006_PancreaticCancer": "soft_tissue",
    "Dataset007_KidneyCancer": "soft_tissue",
    "Dataset008_LymphNodes": "soft_tissue",
    "Dataset009_ColonCancer": "soft_tissue",
    "Dataset010_EndometrialCancer": "soft_tissue"
}


# ─── CT windowing ─────────────────────────────────────────────────────────────

def ct_window(
    data: np.ndarray, 
    level: int, 
    width: int
) -> np.ndarray:
    """Apply windowing to CT HU data and then standardize to [0, 255.0] float16.
    Args:
        data: (Z, Y, X) float32 CT array in Hounsfield units
        level: window level (center)
        width: window width 
    """
    lo = level - width / 2
    hi = level + width / 2
    data = np.clip(data, lo, hi)
    return ((data - lo) / (hi - lo) * 255.0).astype(np.float16)


# ─── NIfTI reading (sitk, with a nibabel fallback for non-orthonormal sforms) ──

def _orthogonalize(direction: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(direction)
    return u @ vt


def _meta_from_affine(affine: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(affine[:3, :3], dtype=np.float64)
    spacing = np.linalg.norm(matrix, axis=0)
    if np.any(spacing == 0):
        raise RuntimeError("NIfTI affine has a zero-length spatial axis")
    direction = _orthogonalize(matrix / spacing)
    origin = np.asarray(affine[:3, 3], dtype=np.float64)
    return spacing, direction.reshape(-1), origin


def read_case(
    img_path: str | Path, 
    lbl_path: str | Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Read scan (CT) and segmentation mask from nifti and return as (Z,Y,X) numpy arrays, where Z is slices. Fall back to nibabel for sforms ITK rejects.
    Args:
        img_path: path to the image file
        lbl_path: path to the label file
    Returns:
        ct: (Z,Y,X) float32 CT array
        lbl: (Z,Y,X) uint8 label array
        spacing: (3,) float64 image spacing from sitk.GetSpacing (x,y,z)
        direction: (9,) float64 image direction from sitk.GetDirection
        origin: (3,) float64 image origin from sitk.GetOrigin
        reader: str, which reader was used to load the image and label, either 'sitk' or 'nibabel_orthofix'
    """
    try:
        img = sitk.ReadImage(str(img_path))
        ct = sitk.GetArrayFromImage(img).astype(np.float32)
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(str(lbl_path)))
        return (ct, lbl, np.array(img.GetSpacing()), np.array(img.GetDirection()),
                np.array(img.GetOrigin()), "sitk")
    except RuntimeError as exc:
        if "orthonormal direction cosines" not in str(exc):
            raise
        img_nib = nib.load(str(img_path))
        lbl_nib = nib.load(str(lbl_path))
        ct = np.asanyarray(img_nib.dataobj).astype(np.float32).transpose(2, 1, 0)
        lbl = np.asanyarray(lbl_nib.dataobj).transpose(2, 1, 0)
        spacing, direction, origin = _meta_from_affine(np.asarray(img_nib.affine))
        return ct, lbl, spacing, direction, origin, "nibabel_orthofix"


# ─── Label extent filter ───────────────────────────────────────────────────────

def count_label_slices(lbl_path: str | Path) -> int:
    """Number of axial slices containing any foreground label.

    Read with nibabel and counted on the *last* array axis, which is the axis
    read_case transposes to front as Z -- so this matches the pipeline's notion
    of a slice without needing the CT (the expensive read) at all.
    """
    lbl = np.asanyarray(nib.load(str(lbl_path)).dataobj)
    return int(np.count_nonzero(np.any(lbl > 0, axis=(0, 1))))


# ─── RECIST longest-diameter line ─────────────────────────────────────────────

def compute_recist_line(mask_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Farthest pair of external-contour pixels on a 2D lesion slice (clinical LD)."""
    contours, _ = cv2.findContours(
        mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    pts = np.vstack(contours).squeeze()
    if pts.ndim != 2 or len(pts) < 2:
        return None
    if len(pts) > MAX_CONTOUR_PTS:
        pts = pts[np.linspace(0, len(pts) - 1, MAX_CONTOUR_PTS, dtype=int)]
    dist = squareform(pdist(pts))
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    return pts[i].astype(int), pts[j].astype(int)


def recist_length_mm(
    p1: np.ndarray, 
    p2: np.ndarray, 
    spacing: np.ndarray
) -> float:
    """Physical length of an in-plane RECIST line.

    Contour points are cv2 (col, row) == (X, Y) on an axial slice; `spacing` is
    sitk order (x, y, z), so component 0 scales the column delta and component 1
    the row delta. The line is in-plane, so z spacing never enters.
    """
    dx = (float(p1[0]) - float(p2[0])) * float(spacing[0])
    dy = (float(p1[1]) - float(p2[1])) * float(spacing[1])
    return float(np.hypot(dx, dy))


def generate_recist(
    instance: np.ndarray, 
    spacing: np.ndarray, 
    min_recist_mm: float = 0.0,
) -> tuple[np.ndarray, list[int]]:
    """Draw each lesion's longest-diameter line on its largest-area axial slice.

    Returns (recist, short_ids). short_ids are lesions whose longest diameter is
    below min_recist_mm (empty when the filter is off, i.e. min_recist_mm <= 0);
    they get no line, and the caller drops them from `gts` too. There is no voxel
    -count floor: min_recist_mm already excludes small lesions, and in physical
    units rather than voxels.

    Args:
        instance: segmentation mask, (Z, Y, X) uint16 cc3d 26-connectivity, can have multiple lesions with different label values
        spacing: mask spacing (3,) float64 (x, y, z)
        min_recist_mm: minimum diameter threshold for removal of small lesions from gts
    
    Returns:
        recist: array of generated longest-diameter line recist annotations per-lesion (Z, Y, X) uint16, value == lesion id
        short_ids: list of lesion ids whose longest diameter is below min_recist_mm
    """
    recist = np.zeros_like(instance, dtype=np.uint16)
    short_ids: list[int] = []
    for lid in np.unique(instance):
        if lid == 0:
            continue
        # isolate individual lesion mask and find the axial slice with the largest area (most voxels)
        mask = (instance == lid).astype(np.uint8)
        key_slice = int(np.argmax(np.sum(mask, axis=(1, 2))))
        # Compute RECIST measurement endpoints
        result = compute_recist_line(mask[key_slice])
        if result is None:
            # Degenerate lesion (largest slice is a single pixel), so its
            # diameter is 0 mm -- short by any positive cutoff. Counting it as
            # short keeps the "nothing in gts without a prompt" invariant.
            if min_recist_mm > 0:
                short_ids.append(int(lid))
            continue
        p1, p2 = result
        if min_recist_mm > 0 and recist_length_mm(p1, p2, spacing) < min_recist_mm:
            # Lesion diameter is below the threshold, so it is not included in the RECIST prompt.
            short_ids.append(int(lid))
            continue
        # Draw RECIST line for largest enough lesions
        cv2.line(recist[key_slice], (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                 color=int(lid), thickness=RECIST_LINE_THICKNESS)
    
    return recist, short_ids


# ─── Tumor ROI crop along Z ────────────────────────────────────────────────────

def tumor_z_range(
    instance: np.ndarray, 
    margin: int = 0
) -> tuple[int, int] | None:
    """[z_start, z_stop) contiguous span of axial slices holding label, + margin.

    Contiguous on purpose: the exact set of labeled slices would drop the gaps
    between separate lesions and leave a stack whose z spacing is a lie. None
    when the volume holds no label at all.
    """
    idx = np.flatnonzero(np.any(instance > 0, axis=(1, 2)))
    if idx.size == 0:
        return None
    z0 = max(0, int(idx[0]) - margin)
    z1 = min(int(instance.shape[0]), int(idx[-1]) + 1 + margin)
    return z0, z1


def shift_origin_z(
    origin: np.ndarray, 
    spacing: np.ndarray, 
    direction: np.ndarray,
    z0: int
) -> np.ndarray:
    """Physical origin of the sub-volume that starts at axial index z0.

    sitk maps index->physical as origin + D @ (spacing * index) with D the 3x3
    direction matrix (GetDirection is its row-major flattening), so moving the
    first slice to z0 shifts the origin along D's third column. The nibabel
    fallback in read_case builds direction with the same column convention.
    """
    if z0 == 0:
        return np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    return np.asarray(origin, dtype=np.float64) + d[:, 2] * float(spacing[2]) * float(z0)


# ─── Resume support ────────────────────────────────────────────────────────────

def expected_keys(
    with_recist: bool,
    with_zcrop: bool = False
) -> set[str]:
    """Helper function for is_complete to check if the npz has the expected keys.
    
    Args:
        with_recist: bool, whether the npz should have the 'recist' key
        with_zcrop: bool, whether the npz should have the 'z_crop' key
    
    Returns:
        keys: set of expected keys in the npz file
    """
    keys = {"imgs", "gts", "spacing", "direction", "origin", "reader"}
    if with_recist:
        keys.add("recist")
    if with_zcrop:
        keys.add("z_crop")
    return keys


def is_complete(
    out_path: str | Path, 
    with_recist: bool, 
    with_zcrop: bool = False
) -> bool:
    """True if out_path is an npz that opens cleanly and holds the expected keys.

    Only the zip central directory + member names are read (np.load is lazy), so
    this is cheap. A truncated/corrupt file from a killed run reads as False and
    gets reconverted. 'z_crop' must be present iff cropping is requested, so
    flipping --tumor-slices-only reconverts stale npz without needing
    --overwrite.
    """
    out_path = Path(out_path)
    if not out_path.is_file() or out_path.stat().st_size == 0:
        return False
    try:
        with np.load(str(out_path), allow_pickle=False) as data:
            files = set(data.files)
    except (zipfile.BadZipFile, OSError, ValueError, EOFError):
        return False
    if not with_zcrop and "z_crop" in files:
        return False
    return expected_keys(with_recist, with_zcrop).issubset(files)


def savez_atomic(
    out_path: Path, 
    arrays: dict[str, Any]
) -> None:
    """np.savez_compressed via a temp file + rename, so readers never see a partial npz."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile in the destination dir keeps the rename on one filesystem.
    # Suffix must end in '.npz' -- np.savez_compressed appends '.npz' otherwise.
    fd, tmp = tempfile.mkstemp(dir=str(out_path.parent), prefix=f".{out_path.stem}.", suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ─── One case ──────────────────────────────────────────────────────────────────

def case_to_npz(
    img_path: str | Path, 
    lbl_path: str | Path, 
    out_path: str | Path, 
    *,
    level: int, 
    width: int, 
    with_recist: bool = True,
    min_recist_mm: float = 0.0, 
    tumor_slices_only: bool = False, 
    tumor_slice_margin: int = 0,
) -> dict[str, int]:
    """Window one NIfTI image+label -> npz. Always (over)writes; caller handles skipping.

    -> {'dropped', 'remaining', 'z0', 'z1', 'nz_full'} (z* == 0/nz_full when uncropped)

    Args:
        img_path: str | Path, path to the image/scan file
        lbl_path: str | Path, path to the segmentation label/mask file
        out_path: str | Path, path to save the output npz file
        level: int, window level (center) for image windowing
        width: int, window width for image windowing
        with_recist: bool, whether to compute RECIST longest-diameter lines
        min_recist_mm: float, minimum diameter threshold for inclusion of lesions in gts and recist, default 0.0 (no filtering)
        tumor_slices_only: bool, whether to crop the output arrays along Z to the tumor ROI, default False
        tumor_slice_margin: int, margin to add around the tumor slice crop, default 0

    Returns:
        stats: dict[str, int], statistics about the conversion, including:
            'dropped': number of lesions dropped due to min_recist_mm
            'remaining': number of lesions remaining in gts after filtering
            'z0': starting slice index of the tumor crop (0 if uncropped)
            'z1': ending slice index of the tumor crop (nz_full if uncropped)
            'nz_full': total number of slices in the original volume
    """
    ct, lbl, spacing, direction, origin, reader = read_case(img_path, lbl_path)
    imgs = ct_window(ct, level, width)
    
    instance = cc3d.connected_components((lbl > 0).astype(np.uint8), connectivity=26).astype(np.uint16)

    recist = None
    short_ids: list[int] = []
    if with_recist:
        recist, short_ids = generate_recist(instance, spacing, min_recist_mm=min_recist_mm)
        if short_ids:
            # Drop sub-threshold lesions from the segmentation target as well, so
            # gts never contains a lesion that has no RECIST prompt. Surviving
            # lesions keep their original ids (gaps are expected).
            instance[np.isin(instance, short_ids)] = 0

    nz_full = int(instance.shape[0])
    z0, z1 = 0, nz_full
    if tumor_slices_only:
        # Derived from the post-filter instance map, so lesions dropped above do
        # not hold the crop open. A case with no label left keeps its full extent
        # -- a zero-slice npz would just be a broken file.
        span = tumor_z_range(instance, margin=tumor_slice_margin)
        if span is not None:
            z0, z1 = span
            imgs = imgs[z0:z1]
            instance = instance[z0:z1]
            if recist is not None:
                recist = recist[z0:z1]
            origin = shift_origin_z(origin, spacing, direction, z0)

    arrays: dict[str, Any] = {
        "imgs": imgs, "gts": instance, "spacing": spacing,
        "direction": direction, "origin": origin, "reader": np.array(reader),
    }
    if recist is not None:
        arrays["recist"] = recist
    if tumor_slices_only:
        arrays["z_crop"] = np.array([z0, z1], dtype=np.int64)

    savez_atomic(Path(out_path), arrays)
    return {"dropped": len(short_ids), "remaining": int(len(np.unique(instance)) - 1),
            "z0": z0, "z1": z1, "nz_full": nz_full}


def _convert_one(job: dict[str, Any]) -> tuple[str, str, str | None, dict[str, int]]:
    """-> (status, out_path, detail, stats). status: ok | skip | filtered | error."""
    out = job["out"]
    try:
        # Label-extent filter runs BEFORE the resume check, so a case that no
        # longer qualifies is reported (and optionally pruned) even when its npz
        # was written by an earlier run. Only the label is read here, not the CT.
        min_slices = job["min_slices"]
        if min_slices > 0:
            n_slices = count_label_slices(job["lbl"])
            if n_slices < min_slices:
                existed = Path(out).exists()
                if existed and job["prune_filtered"]:
                    Path(out).unlink(missing_ok=True)
                detail = f"{n_slices} label slice(s) < {min_slices}"
                if existed:
                    detail += " [pruned existing npz]" if job["prune_filtered"] else " [existing npz LEFT in place]"
                return ("filtered", out, detail, {})

        if not job["overwrite"] and is_complete(out, job["with_recist"], job["tumor_slices_only"]):
            return ("skip", out, None, {})
        stats = case_to_npz(
            job["img"], job["lbl"], out, level=job["level"], width=job["width"],
            with_recist=job["with_recist"], min_recist_mm=job["min_recist_mm"],
            tumor_slices_only=job["tumor_slices_only"],
            tumor_slice_margin=job["tumor_slice_margin"])
        dropped, remaining = stats["dropped"], stats["remaining"]
        detail = None
        if dropped:
            detail = f"dropped {dropped} lesion(s) with RECIST < {job['min_recist_mm']:g} mm"
            if remaining == 0:
                detail += " -- NO lesions left in gts"
        return ("ok", out, detail, stats)
    except Exception as exc:  # noqa: BLE001 -- per-case error, collected not fatal
        return ("error", out, repr(exc), {})


# ─── Batch ─────────────────────────────────────────────────────────────────────

def build_pairs(
    ds: str, 
    root: Path
) -> list[tuple[Path, Path, str]]:
    """(image, label, case_id) triples. Fail loud on structural problems."""
    if ds not in DATASET_WINDOW:
        raise SystemExit(f"[setup] unknown dataset {ds!r} (known: {sorted(DATASET_WINDOW)})")
    images_tr = root / ds / "imagesTr"
    labels_tr = root / ds / "labelsTr"
    if not images_tr.is_dir():
        raise SystemExit(f"[setup] missing imagesTr: {images_tr}")
    if not labels_tr.is_dir():
        raise SystemExit(f"[setup] missing labelsTr: {labels_tr}")
    images = sorted(images_tr.glob(f"*{IMG_SUFFIX}"))
    if not images:
        raise SystemExit(f"[setup] no '*{IMG_SUFFIX}' images under {images_tr}")
    pairs: list[tuple[Path, Path, str]] = []
    for img in images:
        case = img.name[: -len(IMG_SUFFIX)]
        lbl = labels_tr / f"{case}.nii.gz"
        if not lbl.exists():
            raise SystemExit(f"[fail-loud] image without label: {img.name} -> expected {lbl}")
        pairs.append((img, lbl, case))
    return pairs


def run_dataset(
        ds: str, 
        root: Path, 
        out_root: Path, 
        workers: int, 
        with_recist: bool,
        overwrite: bool = False, 
        min_slices: int = MIN_LABEL_SLICES,
        prune_filtered: bool = False, 
        min_recist_mm: float = MIN_RECIST_MM,
        tumor_slices_only: bool = False,
        tumor_slice_margin: int = 0
) -> tuple[int, int, int, int, int, int, int]:
    """Convert one dataset to per-case npz, with optional RECIST and tumor-slice cropping.

    Args:
        ds: str, dataset name (must be in DATASET_WINDOW)
        root: Path, pathway to the image and mask directories (nnU-Net raw root)
        out_root: Path, output directory for the npz files
        workers: int, number of parallel processes to use
        with_recist: bool, whether to compute RECIST longest-diameter lines
        overwrite: bool, whether to reconvert every case (default False, skip existing npz)
        min_slices: int, minimum number of axial slices with label for inclusion of a case
        prune_filtered: bool, whether to delete npz that no longer meets the min_slices
        min_recist_mm: float, minimum diameter threshold for inclusion of lesions in gts and recist, default 0.0 (no filtering)
        tumor_slices_only: bool, whether to crop the output arrays along Z to the tumor ROI, default False
        tumor_slice_margin: int, margin to add around the tumor slice crop, default 0
    
    Returns:
        tuple of counts: (written, skipped, filtered, errors, dropped_lesions, z_kept, z_total)
    """
    bucket = DATASET_WINDOW[ds]
    win = WINDOW_BUCKET[bucket]
    level, width = win["level"], win["width"]
    pairs = build_pairs(ds, root) # list of (image_path, label_path, case_id) --> can get this from med-imagetools index
    out_dir = out_root / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {ds}  bucket={bucket}  window=L{level}/W{width}  cases={len(pairs)}  "
          f"recist={'on' if with_recist else 'off'}  "
          f"mode={'overwrite' if overwrite else 'resume'}  "
          f"min_label_slices={min_slices or 'off'}  "
          f"min_recist_mm={f'{min_recist_mm:g}' if min_recist_mm > 0 else 'off'}  "
          f"tumor_slices_only={f'on(+{tumor_slice_margin})' if tumor_slices_only else 'off'} ===",
          flush=True)

    jobs = [{"img": str(img), "lbl": str(lbl), "out": str(out_dir / f"{case}.npz"),
             "level": level, "width": width, "with_recist": with_recist,
             "overwrite": overwrite, "min_slices": min_slices,
             "prune_filtered": prune_filtered, "min_recist_mm": min_recist_mm,
             "tumor_slices_only": tumor_slices_only,
             "tumor_slice_margin": tumor_slice_margin}
            for img, lbl, case in pairs]
    written = 0
    skipped = 0
    dropped_lesions = 0
    z_kept = 0
    z_total = 0
    small: list[tuple[str, str | None]] = []
    filtered: list[tuple[str, str | None]] = []
    errors: list[tuple[str, str | None]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_convert_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            status, out, detail, stats = fut.result()
            dropped_lesions += stats.get("dropped", 0)
            if stats:
                z_kept += stats["z1"] - stats["z0"]
                z_total += stats["nz_full"]
            if status == "ok":
                written += 1
                if detail:
                    small.append((out, detail))
            elif status == "skip":
                skipped += 1
            elif status == "filtered":
                filtered.append((out, detail))
            else:
                errors.append((out, detail))
            if i % 50 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} done; written={written} skipped={skipped} "
                      f"filtered={len(filtered)} errors={len(errors)} "
                      f"small_lesions_dropped={dropped_lesions}", flush=True)
    for out, detail in small:
        print(f"  SMALL {Path(out).stem}: {detail}", flush=True)
    for out, detail in filtered:
        print(f"  FILTERED {Path(out).stem}: {detail}", flush=True)
    for out, detail in errors[:10]:
        print(f"  ERROR {out}: {detail}", flush=True)
    crop_note = ""
    if tumor_slices_only and z_total:
        crop_note = f" slices_kept={z_kept}/{z_total} ({100.0 * z_kept / z_total:.1f}%)"
    print(f"  summary: {ds} written={written} skipped={skipped} "
          f"filtered={len(filtered)} errors={len(errors)} "
          f"small_lesions_dropped={dropped_lesions}{crop_note}", flush=True)
    return written, skipped, len(filtered), len(errors), dropped_lesions, z_kept, z_total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=sorted(DATASET_WINDOW), help="convert one dataset")
    g.add_argument("--all", action="store_true", help="convert all 14 datasets")
    p.add_argument("--root", type=Path, default=Path(ROOT_DEFAULT), help="nnU-Net raw root")
    p.add_argument("--out-root", type=Path, default='./RECIST-npz',
                   help="output dir (default <root>/npz_version_data)")
    p.add_argument("--workers", type=int, default=24, help="process-pool size")
    p.add_argument("--no-recist", action="store_true",
                   help="skip RECIST line generation and omit the 'recist' key")
    p.add_argument("--overwrite", action="store_true",
                   help="reconvert every case (default: skip cases whose npz is already complete)")
    p.add_argument("--min-label-slices", type=int, default=MIN_LABEL_SLICES,
                   help=f"exclude cases whose label spans fewer axial slices "
                        f"(default {MIN_LABEL_SLICES}; 0 disables the filter)")
    p.add_argument("--prune-filtered", action="store_true",
                   help="delete existing npz for cases the label-slice filter now excludes "
                        "(default: report them and leave the files alone)")
    p.add_argument("--exclude-small-lesions", action=argparse.BooleanOptionalAction, default=True,
                   help=f"drop lesions whose RECIST longest diameter is < --min-recist-mm from "
                        f"both 'gts' and 'recist' (default: on, threshold {MIN_RECIST_MM:g} mm)")
    p.add_argument("--min-recist-mm", type=float, default=MIN_RECIST_MM,
                   help=f"RECIST longest-diameter cutoff in mm for --exclude-small-lesions "
                        f"(default {MIN_RECIST_MM:g})")
    p.add_argument("--tumor-slices-only", action="store_true",
                   help="crop every array along Z to the contiguous span of axial slices "
                        "holding label (default: off, keep the full volume); shifts 'origin' "
                        "and records 'z_crop'")
    p.add_argument("--tumor-slice-margin", type=int, default=0,
                   help="extra axial slices to keep on each side of the tumor span "
                        "(default 0; only with --tumor-slices-only)")
    args = p.parse_args(argv)
    if args.min_label_slices < 0:
        p.error("--min-label-slices must be >= 0")
    if args.min_recist_mm < 0:
        p.error("--min-recist-mm must be >= 0")
    if args.tumor_slice_margin < 0:
        p.error("--tumor-slice-margin must be >= 0")
    if args.tumor_slice_margin and not args.tumor_slices_only:
        p.error("--tumor-slice-margin has no effect without --tumor-slices-only")

    out_root = args.out_root or (args.root / "npz_version_data")
    datasets = sorted(DATASET_WINDOW) if args.all else [args.dataset]
    with_recist = not args.no_recist
    min_recist_mm = args.min_recist_mm if args.exclude_small_lesions else 0.0
    # The cutoff is measured on the RECIST line, so it cannot be applied at all
    # without computing one.
    if min_recist_mm > 0 and not with_recist:
        p.error("--exclude-small-lesions needs the RECIST lines; drop --no-recist "
                "or pass --no-exclude-small-lesions")

    print(f"root={args.root}  out_root={out_root}  datasets={len(datasets)}  "
          f"workers={args.workers}  recist={'on' if with_recist else 'off'}  "
          f"mode={'overwrite' if args.overwrite else 'resume'}  "
          f"min_label_slices={args.min_label_slices or 'off'}  "
          f"min_recist_mm={f'{min_recist_mm:g}' if min_recist_mm > 0 else 'off'}  "
          f"tumor_slices_only={f'on(+{args.tumor_slice_margin})' if args.tumor_slices_only else 'off'}"
          f"{'  prune_filtered=on' if args.prune_filtered else ''}", flush=True)
    total_written = 0
    total_skipped = 0
    total_filtered = 0
    total_err = 0
    total_dropped = 0
    total_z_kept = 0
    total_z_full = 0
    for ds in datasets:
        w, s, f, e, d, zk, zt = run_dataset(
            ds, args.root, out_root, args.workers, with_recist, args.overwrite,
            args.min_label_slices, args.prune_filtered, min_recist_mm,
            args.tumor_slices_only, args.tumor_slice_margin)
        total_written += w
        total_skipped += s
        total_filtered += f
        total_err += e
        total_dropped += d
        total_z_kept += zk
        total_z_full += zt
    crop_note = ""
    if args.tumor_slices_only and total_z_full:
        crop_note = (f" slices_kept={total_z_kept}/{total_z_full} "
                     f"({100.0 * total_z_kept / total_z_full:.1f}%)")
    print(f"\nTOTAL written={total_written} skipped={total_skipped} "
          f"filtered={total_filtered} errors={total_err} "
          f"small_lesions_dropped={total_dropped}{crop_note}", flush=True)
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main())