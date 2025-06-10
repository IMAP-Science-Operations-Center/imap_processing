import pytest

from imap_processing import imap_module_directory
from imap_processing.utils import packet_file_to_datasets

IALIRT_PACKET_LENGTH = 1464


@pytest.mark.external_test_data
def test_generate_xarray():
    """Checks that xarray data is properly generated."""

    apid = 478
    packet_path = (
        imap_module_directory / "tests" / "ialirt" / "data" / "l0" / "apid_478.bin"
    )
    xtce_ialirt_path = (
        imap_module_directory / "ialirt" / "packet_definitions" / "ialirt.xml"
    )
    xarray_data = packet_file_to_datasets(
        packet_path, xtce_ialirt_path, use_derived_value=True
    )[apid]

    for key in xarray_data.keys():
        assert len(xarray_data[key]) == 44429

    total_packet_length = xarray_data["pkt_len"].values[0] + 7
    # Convert to bytes
    assert total_packet_length * 8 == IALIRT_PACKET_LENGTH
