import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.l1a.hit_l1a import hit_l1a


@pytest.fixture(scope="module")
def packet_filepath():
    """Set path to test data file"""
    return (
        imap_module_directory / "tests/hit/test_data/imap_hit_l0_raw_20100105_v001.pkts"
    )


def test_hit_l1a(packet_filepath):
    """Create L1A datasets from a packet file.

    Parameters
    ----------
    packet_filepath : str
        Path to ccsds file
    """
    processed_datasets = hit_l1a(packet_filepath, "001")
    # TODO: update assertions after science data processing is completed
    assert isinstance(processed_datasets, list)
    assert len(processed_datasets) == 1
    assert isinstance(processed_datasets[0], xr.Dataset)
    assert processed_datasets[0].attrs["Logical_source"] == "imap_hit_l1a_hk"
