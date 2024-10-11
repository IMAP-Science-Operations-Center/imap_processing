import numpy as np
import pandas as pd

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.l1b.swe_l1b import swe_l1b


def test_swe_l1b(decom_test_data_derived):
    """Test that calculate engineering unit(EU) matches validation data.

    Parameters
    ----------
    decom_test_data_derived : xarray.dataset
        Dataset with derived values
    """
    science_l1a_ds = swe_science(decom_test_data_derived, "001")

    # read science validation data
    test_data_path = imap_module_directory / "tests/swe/l0_validation_data"
    eu_validation_data = pd.read_csv(
        test_data_path / "idle_export_eu.SWE_SCIENCE_20240510_092742.csv",
        index_col="SHCOARSE",
    )

    second_data = science_l1a_ds.isel(epoch=1)
    validation_data = eu_validation_data.loc[second_data["shcoarse"].values]

    science_eu_field_list = [
        "SPIN_PHASE",
        "SPIN_PERIOD",
        "THRESHOLD_DAC",
    ]

    # Test EU values for science data
    for field in science_eu_field_list:
        np.testing.assert_almost_equal(
            second_data[field.lower()].values, validation_data[field], decimal=5
        )


def test_cdf_creation():
    """Test that CDF file is created and has the correct name."""
    test_data_path = "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    l1a_datasets = swe_l1a(imap_module_directory / test_data_path, "002")

    sci_l1a_filepath = write_cdf(l1a_datasets)

    assert sci_l1a_filepath.name == "imap_swe_l1a_sci_20240510_v002.cdf"

    # reads data from CDF file and passes to l1b
    l1a_cdf_dataset = load_cdf(sci_l1a_filepath)
    l1b_dataset = swe_l1b(l1a_cdf_dataset, "002")

    sci_l1b_filepath = write_cdf(l1b_dataset)

    assert sci_l1b_filepath.name == "imap_swe_l1b_sci_20240510_v002.cdf"
