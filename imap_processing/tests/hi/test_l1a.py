import numpy as np

from imap_processing.cdf.utils import write_cdf
from imap_processing.hi.l1a.hi_l1a import hi_l1a
from imap_processing.hi.l1a.histogram import unpack_hist_counter
from imap_processing.hi.utils import HIAPID


def test_sci_de_decom(hi_l0_test_data_path):
    """Test science direct event data"""

    bin_data_path = hi_l0_test_data_path / "H90_sci_de_20241104.bin"
    processed_data = hi_l1a(bin_data_path, data_version="001")

    assert processed_data[0].attrs["Logical_source"] == "imap_hi_l1a_90sensor-de"
    assert processed_data[0].attrs["Data_version"] == "001"

    # TODO: Verify correct unpacking of sample data. Issue: #1186

    # Write to CDF
    cdf_filename = "imap_hi_l1a_90sensor-de_20241105_v001.cdf"
    # TODO: Dropping duplicates to ignore ISTP for now. Should be fixed by #1186
    processed_data[0] = processed_data[0].sortby("epoch").groupby("epoch").first()
    cdf_filepath = write_cdf(processed_data[0])
    assert cdf_filepath.name == cdf_filename


def test_app_nhk_decom(hi_l0_test_data_path):
    """Test housekeeping data"""

    # Unpack housekeeping data
    bin_data_path = hi_l0_test_data_path / "H90_NHK_20241104.bin"
    processed_data = hi_l1a(packet_file_path=bin_data_path, data_version="001")

    assert np.unique(processed_data[0]["pkt_apid"].values) == HIAPID.H90_APP_NHK.value
    assert processed_data[0].attrs["Logical_source"] == "imap_hi_l1a_90sensor-hk"
    assert processed_data[0].attrs["Data_version"] == "001"
    # TODO: compare with validation data once we have it. Issue: #1184

    # Write CDF
    cem_raw_cdf_filepath = write_cdf(processed_data[0], istp=False)

    # TODO: ask Vivek about this date mismatch between the file name
    # and the data. May get resolved when we have good sample data.
    assert cem_raw_cdf_filepath.name == "imap_hi_l1a_90sensor-hk_20241105_v001.cdf"


def test_app_hist_decom(hi_l0_test_data_path):
    """Test histogram (SCI_CNT) data"""
    bin_data_path = hi_l0_test_data_path / "H90_sci_cnt_20241104.bin"
    processed_data = hi_l1a(packet_file_path=bin_data_path, data_version="001")

    assert processed_data[0].attrs["Logical_source"] == "imap_hi_l1a_90sensor-hist"
    # TODO: compare with validation data once we have it. Issue: #1185

    # Write CDF
    cem_raw_cdf_filepath = write_cdf(processed_data[0])

    assert cem_raw_cdf_filepath.name.startswith("imap_hi_l1a_90sensor-hist_")


def test_unpack_hist_counter():
    """Test hi.l1a.histogram.unpack_hist_counter()"""
    # To ensure correct unpacking, use expected values with ones in the upper
    # and lower parts of the 12-bit numbers
    expected = (np.arange(180).reshape((2, 90)) + 2**10).astype(">u2")
    # convert each expected uint16 to a 12-bit bitstring and join
    bin_str = "".join([f"{val:012b}" for val in expected.ravel()])
    # convert the bitstring to a bytes object
    bytes_array = int(bin_str, 2).to_bytes(len(bin_str) // 8, byteorder="big")
    output_array = unpack_hist_counter(bytes_array)
    np.testing.assert_array_equal(output_array, expected)
    assert output_array.dtype == np.uint16
