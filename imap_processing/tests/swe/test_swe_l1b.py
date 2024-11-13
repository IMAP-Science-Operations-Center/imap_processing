import numpy as np
import pandas as pd

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.l1b.swe_l1b import swe_l1b
from imap_processing.swe.l1b.swe_l1b_science import (
    convert_counts_to_rate,
    deadtime_correction,
)


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


def test_cdf_creation(l1b_validation_df):
    """Test that CDF file is created and has the correct name."""
    test_data_path = "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    l1a_datasets = swe_l1a(imap_module_directory / test_data_path, "002")

    l1b_dataset = swe_l1b(l1a_datasets, "002")

    sci_l1b_filepath = write_cdf(l1b_dataset)

    assert sci_l1b_filepath.name == "imap_swe_l1b_sci_20240510_v002.cdf"
    # load the CDF file and compare the values
    l1b_cdf_dataset = load_cdf(sci_l1b_filepath)
    processed_science = l1b_cdf_dataset["science_data"].data
    validation_science = l1b_validation_df.values[:, 1:].reshape(6, 24, 30, 7)
    np.testing.assert_allclose(processed_science, validation_science, rtol=1e-7)


def test_count_rate():
    x = np.array([1, 10, 100, 1000, 10000, 38911, 65535])
    acq_duration = 80000
    deatime_corrected = deadtime_correction(x, acq_duration)
    count_rate = convert_counts_to_rate(deatime_corrected, acq_duration)
    # Ruth provided the expected output for this test
    expected_output = [
        12.50005653,
        125.00562805,
        1250.56278121,
        12556.50455087,
        130890.05519127,
        589631.73670132,
        1161815.68783304,
    ]
    np.testing.assert_allclose(count_rate, expected_output, rtol=1e-7)
