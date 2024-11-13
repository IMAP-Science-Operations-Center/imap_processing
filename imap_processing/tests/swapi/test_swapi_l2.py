import numpy as np

from imap_processing.cdf.utils import write_cdf
from imap_processing.swapi.l1.swapi_l1 import swapi_l1
from imap_processing.swapi.l2.swapi_l2 import TIME_PER_BIN, swapi_l2


def test_swapi_l2_cdf(swapi_l0_test_data_path):
    """Test housekeeping processing and CDF file creation"""
    l0_data_path = swapi_l0_test_data_path / "imap_swapi_l0_raw_20240924_v001.pkts"
    processed_data = swapi_l1(l0_data_path, data_version="v001")
    l1_dataset = processed_data[0]

    l2_dataset = swapi_l2(l1_dataset, data_version="v001")
    l2_cdf = write_cdf(l2_dataset)
    assert l2_cdf.name == "imap_swapi_l2_sci_20240924_v001.cdf"

    # Test uncertainty variables are as expected
    np.testing.assert_array_equal(
        l2_dataset["swp_pcem_rate_err_plus"],
        l1_dataset["swp_pcem_counts_err_plus"] / TIME_PER_BIN,
    )
