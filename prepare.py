import argparse
import logging
import pandas as pd
import SimpleITK as sitk

from damply import dirs
from imgtools.transforms.functional import window_intensity
from pathlib import Path

from utils.masks import get_max_area_slice
from utils.annotations import get_recist_pts, get_line_from_recist

logfile = dirs.LOGS / "models" / "prepare_medsam2.log"
logfile.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
	level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=logfile
)

logger = logging.getLogger(__name__)


def prepare_data(
    imgtools_path: str | Path, 
    output_path: str | Path,
    window_level: int = 0,
    window_width: int = -1,
    padding: int | list = 0,
    ) -> None:

    imgtools_path = Path(imgtools_path)

    index_csv = pd.read_csv(imgtools_path / f"{imgtools_path.stem}_index-simple.csv")

    # DO THE NPZ CONVERSION HERE

    # group the index by SampleNumber and iterate over the groups
    for sample_number, metadata in index_csv.iloc[0:5].groupby("SampleNumber"):
        logger.info(f"Processing sample {sample_number} with {len(metadata)} images")

        # Get metadata for the scan and mask(s) for this sample
        scan_metadata = metadata[metadata['class'] == 'Scan']
        mask_metadata = metadata[metadata['class'] == 'Mask']

        if mask_metadata.empty:
            logger.warning(f"No mask found for sample {sample_number}, skipping.")
            continue

        # Load and process scan image for this sample
        scan = sitk.ReadImage(imgtools_path / scan_metadata.iloc[0]['filepath'], outputPixelType=sitk.sitkInt16)
        scan_size = scan.GetSize()

        # Apply windowing to the image only
        if window_width != -1:
            scan = window_intensity(scan, window_level, window_width)

        # Load each mask for this sample and save the scan and mask as .npz files
        for mask_count, (_, mask_row) in enumerate(mask_metadata.iterrows()):
            sample_id = f"{mask_row['PatientID']}_{mask_row['SampleNumber']}_{mask_count}"
            mask = sitk.ReadImage(imgtools_path / mask_row['filepath'], outputPixelType=sitk.sitkUInt8)

            #TODO: figure out how to do rerecist calculation
            # Get largest area slice in mask
            max_area_slice, _max_area_slice_idx = get_max_area_slice(mask)

            # Calculate RERECIST points from largest mask slice
            rerecist_pts = get_recist_pts(max_area_slice)
            rerecist_line = get_line_from_recist(rerecist_pts,
                                                 slice_idx = max_area_slice, 
                                                 scan_size = [scan_size[2], scan_size[0], scan_size[1]])



            print(rerecist_line)
            

            

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--imgtools_path", type=str, required=True)
    argparser.add_argument("--output_path", type=str, required=True)
    args = argparser.parse_args()

    prepare_data(args.imgtools_path, args.output_path)