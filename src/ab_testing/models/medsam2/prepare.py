import logging
from pathlib import Path

import click
from damply import dirs

from ab_testing.utils.conver_nifti_to_recist_npz import (
    MIN_LABEL_SLICES,
    MIN_RECIST_MM,
    run_dataset,
)

logfile = dirs.LOGS / "models" / "prepare_medsam2.log"
logfile.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
	level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=logfile
)

logger = logging.getLogger(__name__)

ANATOMICAL_BUCKETS = [
    "lung",
    "soft_tissue",   # includes heart
    "bone",
    "brain",
]

@click.command(no_args_is_help=True)
@click.argument(
    "dataset",
    type=str,
)
@click.argument(
    "mit_directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=False, path_type=Path, resolve_path=True, exists=True
    ),
)
@click.argument(
    "output_directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, path_type=Path, resolve_path=True
    ),
)
@click.option(
    "--anat-window",
    type=click.Choice(ANATOMICAL_BUCKETS),
    required=True,
    help=f"Anatomical windowing to apply for this dataset. Must be one of {ANATOMICAL_BUCKETS}."
)
@click.option(
    "--min-slices",
    type=int,
    default=MIN_LABEL_SLICES,
    show_default=True,
    help="Minimum labelled slice count required for npz conversion."
)
@click.option(
    "--min-recist-mm",
    type=float,
    default=MIN_RECIST_MM,
    show_default=True,
    help="Minimum RECIST diameter measurement required for npz conversion in millimetres (mm)."
)
@click.option(
    "--workers",
    type=int,
    default=24,
    help="Number of parallel processes to use."
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Whether to overwrite existing files."
)
def prepare_data(
    dataset: str, 
    mit_directory: str | Path, 
    output_directory: str | Path,
    anat_window: str,
    *,
    min_slices: int = MIN_LABEL_SLICES,
    min_recist_mm: float = MIN_RECIST_MM,
    workers: int = 24,
    overwrite: bool = False,
    ) -> int:
    """Process a med-imagetools nifti dataset by pair of image-masks and save into npzs.
    
    Converted npz contains:
        * imgs: The image, windowed based on the anatomical window specified,
        * gts: The corresponding 3D segmentation mask, 
        * spacing: Original image/mask spacing, from SimpleITK
        * direction: Original image/mask direction, from SimpleITK
        * origin: Original image/mask origin, from SimpleTIK 
        * reader: What library was used to load in the image/mask, will be sitk or nibabel_orthofix
        * recist: Reverse-engineered RECIST longest diameter annotation derived from gts
    
    Based on min_slices and min_recist_mm, samples will be pruned and not saved as npz if they fall below the threshold.

    Args:
        dataset: str, dataset name
        mit_directory: str | Path, pathway to the image and mask directory
        output_directory: str | Path, output directory npz files will be saved to
        anat_window: anatomical windowing to apply for this dataset
        min_slices: int, minimum number of axial slices with label for lesion inclusion (default 5)
        min_recist_mm: float, RECIST diameter threshold in millimetres for lesion inclusion (default 10)
        workers: int, number of parallel processes to use (default 24)
        overwrite: bool, whether to reconvert every case (default False, skip existing npz)
    
    Returns:
        0 if successful, 1 if errors occurred.
    """
    written, skipped, filtered, errors, dropped_lesions, _z_kept, _z_total = run_dataset(
        ds = dataset,
        root = Path(mit_directory),
        out_root = Path(output_directory),
        workers = workers,
        with_recist = True,
        overwrite = overwrite,
        min_slices = min_slices,
        min_recist_mm = min_recist_mm,
        prune_filtered = True,
        anat_window = anat_window,
        pair_builder = 'mit',
    )

    logger.info(f"\nTOTAL written={written} skipped={skipped} "
                f"filtered={filtered} errors={errors} "
                f"small_lesions_dropped={dropped_lesions}")

    return 1 if errors else 0

            

            

if __name__ == "__main__":
    prepare_data()
