import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo.l0.lo_apid import LoAPID
from imap_processing.lo.l0.lo_science import decompress, parse_histogram
from imap_processing.utils import packet_file_to_datasets


@pytest.mark.skip()
@pytest.fixture()
def sample_packet_dataset():
    # original file name:
    # Instrument_Emulator_ILO_Emulator_v3.4_HVSCI_Sample_20240627T204953.CCSDS
    test_file = (
        imap_module_directory / "tests/lo/sample_data/imap_lo_l0_raw_20240627_v001.pkts"
    )
    xtce_file = imap_module_directory / "lo/packet_definitions/lo_xtce.xml"

    datasets_by_apid = packet_file_to_datasets(
        packet_file=test_file.resolve(),
        xtce_packet_definition=xtce_file.resolve(),
    )
    hist_data = datasets_by_apid[LoAPID.ILO_SCI_CNT]

    return hist_data


@pytest.mark.skip()
@pytest.fixture()
def attr_mgr():
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1a")
    attr_mgr.add_global_attribute("Data_version", "001")
    return attr_mgr


# The Lo sample data is currently too large to be committed to the repo
# skipping this test until smaller sample data file size is available.
# The validation data is also in a compressed integer form, so only checking
# the data shapes for the tests for now until uncompressed validation data
# becomes available.
@pytest.mark.skip()
def test_parse_histogram(sample_packet_dataset, attr_mgr):
    # Arrange
    # all fields that should have 6 azimuth bins
    az_6_dims = ["tof0_tof1", "tof0_tof2", "tof1_tof2", "silver", "hydrogen", "oxygen"]
    # all fields that should only have an epoch dim
    epoch_only_dims = [
        "shcoarse",
        "sci_cnt",
        "chksum",
        "version",
        "type",
        "sec_hdr_flg",
        "pkt_apid",
        "pkt_len",
        "seq_flgs",
        "src_seq_ctr",
    ]
    # all other fields should have 60 az bins

    # Act
    parsed_dataset = parse_histogram(sample_packet_dataset, attr_mgr)

    # Assert
    # Checking for data shape only until Lo sends validation data to use
    for field in parsed_dataset.data_vars:
        if field in az_6_dims:
            assert parsed_dataset[field].shape == (5, 60, 7)
        elif field in epoch_only_dims:
            assert parsed_dataset[field].shape == (5,)
        else:
            assert parsed_dataset[field].shape == (5, 6, 7)


def test_decompress_8_to_16_bit():
    # Arrange
    # 174 in binary (= 2206 decompressed)
    idx0 = "10101110"
    # 20 in binary (= 20 decompressed)
    idx1 = "00010100"
    bin_str = idx0 + idx1
    bits_per_index = 8
    section_start = 0
    section_length = 16
    expected = [2206, 20]

    # Act
    out = decompress(bin_str, bits_per_index, section_start, section_length)

    # Assert
    assert out == expected


def test_decompress_12_to_16_bit():
    # Arrange
    # 3643 in binary (= 33400 decompressed)
    idx0 = "111000111011"
    # 20 in binary (= 20 decompressed
    idx1 = "000000010100"
    bin_str = idx0 + idx1
    bits_per_index = 12
    section_start = 0
    section_length = 24
    expected = [33400, 20]

    # Act
    out = decompress(bin_str, bits_per_index, section_start, section_length)

    # Assert
    assert out == expected
