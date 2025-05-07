from pathlib import Path
from unittest import mock

import pytest
from imap_data_access.processing_input import ProcessingInput, AncillaryInput

from imap_processing.cdf.utils import load_cdf
from tools.ancillary.ancillary_dataset_generator import AncillaryConverter, \
    MagAncillaryConverter


@pytest.fixture
def mocks():
    with (
        mock.patch("tools.ancillary.ancillary_dataset_generator.cdf_to_xarray") as read_cdf):
        mocks = {
            "read_cdf": read_cdf,
        }
        yield mocks

@pytest.fixture
def mag_calibration_dataset():
    imap_dir = Path(__file__).parent.parent.parent.parent
    cal_file = (
            imap_dir
            / "imap_processing"
            / "tests"
            / "mag"
            / "validation"
            / "calibration"
            / "imap_mag_l2-calibration-matrices_20251017_v004.cdf"
    )
    calibration_data = load_cdf(cal_file)

    return calibration_data


def test_ancillary_converter(mocks, mag_calibration_dataset):
    # Simple case, no overlap
    input_example = AncillaryInput(
        "imap_mag_l2-calibration-matrices_20251017_20251020_v003.cdf",
        "imap_mag_l2-calibration-matrices_20251020_20251021_v004.cdf"
    )

    mocks['read_cdf'].return_value = mag_calibration_dataset

    output = MagAncillaryConverter(input_example)
    print(output.combined_dataset)
