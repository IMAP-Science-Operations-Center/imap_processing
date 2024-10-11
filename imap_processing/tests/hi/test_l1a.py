import numpy as np
import pytest

from imap_processing.cdf.utils import write_cdf
from imap_processing.hi.l1a import histogram as hist
from imap_processing.hi.l1a.hi_l1a import hi_l1a
from imap_processing.hi.utils import HIAPID


def test_sci_de_decom(create_de_data):
    """Test science direct event data"""

    # Process using test data
    processed_data = hi_l1a(
        packet_file_path=create_de_data(HIAPID.H45_SCI_DE.value), data_version="001"
    )

    assert processed_data[0].attrs["Logical_source"] == "imap_hi_l1a_45sensor-de"
    assert processed_data[0].attrs["Data_version"] == "001"

    # unique ESA steps should be [1, 2]
    assert np.array_equal(
        np.sort(np.unique(processed_data[0]["esa_step"].values)),
        np.array([1, 2]),
    )
    # unique trigger_id should be [1, 2, 3]
    assert np.array_equal(
        np.sort(np.unique(processed_data[0]["trigger_id"].values)), np.array([1, 2, 3])
    )
    # tof_x should be in this range [0, 1023]
    assert processed_data[0]["tof_1"].min() >= 0
    assert processed_data[0]["tof_1"].max() <= 1023
    assert processed_data[0]["tof_2"].min() >= 0
    assert processed_data[0]["tof_2"].max() <= 1023
    assert processed_data[0]["tof_3"].min() >= 0
    assert processed_data[0]["tof_3"].max() <= 1023

    # Write to CDF
    cdf_filename = "imap_hi_l1a_45sensor-de_20230927_v001.cdf"
    # TODO: Dropping duplicates to ignore ISTP for now. Need to update test data
    processed_data[0] = processed_data[0].sortby("epoch").groupby("epoch").first()
    cdf_filepath = write_cdf(processed_data[0])
    assert cdf_filepath.name == cdf_filename


def test_app_nhk_decom(hi_l0_test_data_path):
    """Test housekeeping data"""

    # Unpack housekeeping data
    bin_data_path = hi_l0_test_data_path / "20231030_H45_APP_NHK.bin"
    processed_data = hi_l1a(packet_file_path=bin_data_path, data_version="001")

    assert np.unique(processed_data[0]["pkt_apid"].values) == HIAPID.H45_APP_NHK.value
    assert processed_data[0].attrs["Logical_source"] == "imap_hi_l1a_45sensor-hk"
    assert processed_data[0].attrs["Data_version"] == "001"
    # TODO: compare with validation data once we have it

    # Write CDF
    cem_raw_cdf_filepath = write_cdf(processed_data[0], istp=False)

    # TODO: ask Vivek about this date mismatch between the file name
    # and the data. May get resolved when we have good sample data.
    assert cem_raw_cdf_filepath.name == "imap_hi_l1a_45sensor-hk_20100313_v001.cdf"


@pytest.mark.skip(
    reason="Need new test data with monotonically increasing epoch values"
)
def test_app_hist_decom(hi_l0_test_data_path):
    """Test histogram (SCI_CNT) data"""
    bin_data_path = hi_l0_test_data_path / "20231030_H45_SCI_CNT.bin"
    processed_data = hi_l1a(packet_file_path=bin_data_path, data_version="001")

    assert processed_data[0].attrs["Logical_source"] == "imap_hi_l1a_45sensor-hist"
    # TODO: compare with validation data once we have it
    # TODO: Dropping duplicates to ignore ISTP for now. Need to update test data
    processed_data[0] = processed_data[0].sortby("epoch").groupby("epoch").first()

    # Write CDF
    cem_raw_cdf_filepath = write_cdf(processed_data[0])

    assert cem_raw_cdf_filepath.name.startswith("imap_hi_l1a_45sensor-hist_")


def test_allocate_histogram_dataset():
    """Test hi.l1a.histogram.allocate_histogram_dataset()"""
    n_packets = 5
    dataset = hist.allocate_histogram_dataset(n_packets)

    assert dataset.attrs["Data_type"] == "L1A_HIST>Level-1A Histogram"
    assert dataset.sizes["epoch"] == n_packets
    assert dataset.sizes["angle"] == 90
    for var_name in (
        "ccsds_met",
        "esa_stepping_num",
        *hist.QUALIFIED_COUNTERS,
        *hist.LONG_COUNTERS,
        *hist.TOTAL_COUNTERS,
    ):
        assert var_name in dataset
