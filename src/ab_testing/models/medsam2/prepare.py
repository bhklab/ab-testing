import argparse
import logging


from damply import dirs
from pathlib import Path
from typing import Literal

from ab_testing.utils.conver_nifti_to_recist_npz import run_dataset, MIN_LABEL_SLICES, MIN_RECIST_MM


logfile = dirs.LOGS / "models" / "prepare_medsam2.log"
logfile.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
	level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=logfile
)

logger = logging.getLogger(__name__)

ANATOMICAL_BUCKETS = Literal[
    "lung"
    "soft_tissue",   # includes heart
    "bone",
    "brain"
]

def prepare_data(
    dataset: str, 
    imgtools_path: str | Path, 
    output_path: str | Path,
    anat_window: ANATOMICAL_BUCKETS = 'soft-tissue',
    min_slices: int = MIN_LABEL_SLICES,
    min_recist_mm: float = MIN_RECIST_MM,
    workers: int = 24,
    overwrite: bool = False,
    ) -> None:
    """Process a med-imagetools nifti dataset by pair of image-masks and save into a npz:
    * imgs: The image, windowed based on the anatomical window specified,
    * gts: The corresponding 3D segmentation mask, 
    * spacing: Original image/mask spacing, from SimpleITK
    * direction: Original image/mask direction, from SimpleITK
    * origin: Original image/mask origin, from SimpleTIK 
    * reader: What library was used to load in the image/mask, will be sitk or nibabel_orthofix
    * recist: Reverse-engineered RECIST longest diameter annotation derived from gts
    
    Based on min_slices and min_recist_mm, samples will be pruned and not saved as npz if they fall below the threshold.
    """
    written, skipped, filtered, errors, dropped_lesions, _z_kept, _z_total = run_dataset(
        ds = dataset,
        root = Path(imgtools_path),
        out_root = Path(output_path),
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
    prepare_data(
        dataset='NSCLC-Radiomics',
        imgtools_path= dirs.RAWDATA / 'TCIA_NSCLC-Radiomics' / 'images' / 'mit_NSCLC-Radiomics',
        output_path = dirs.PROCDATA / 'TCIA_NSCLC-Radiomics' / 'images' / 'npz_medsam2_data',
        anat_window = 'lung',
        min_slices= 3,
    )



    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("--imgtools_path", type=str, required=True)
    # argparser.add_argument("--output_path", type=str, required=True)
    # args = argparser.parse_args()

    # # dirs.PROCDATA / 'TCIA_NSCLC-Radiomics' / 'images' / 'npz_medsam2_data'

    # prepare_data(args.imgtools_path, args.output_path)