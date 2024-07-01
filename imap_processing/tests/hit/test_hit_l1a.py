import pytest
import xarray as xr

from imap_processing import imap_module_directory, utils
from imap_processing.hit.l0.data_classes.housekeeping import Housekeeping
from imap_processing.hit.l1a import hit_l1a


@pytest.fixture(scope="module")
def packet_filepath():
    """Set path to test data file"""

    return imap_module_directory / "tests/hit/test_data/hskp_sample.ccsds"


@pytest.fixture(scope="module")
def unpacked_packets(packet_filepath):
    """Read test data from file

    Parameters
    ----------
    packet_filepath : str
        Path to ccsds file

    Returns
    -------
    sorted_packets : list[space_packet_parser.parser.Packet]
        A sorted list of decommutated packets
    """
    packets = hit_l1a.decom_packets(packet_filepath)
    sorted_packets = utils.sort_by_time(packets, "SHCOARSE")
    return sorted_packets


def test_group_data(unpacked_packets):
    """Group packet data classes by apid

    Creates data classes for decommutated packets and
    groups them by apid

    Parameters
    ----------
    unpacked_packets : list[space_packet_parser.parser.Packet]
        A sorted list of decommutated packets
    """
    grouped_data = hit_l1a.group_data(unpacked_packets)
    assert len(grouped_data.keys()) == 1
    assert all([isinstance(i, Housekeeping) for i in grouped_data[1251]])


def test_create_datasets(unpacked_packets):
    """Create xarray datasets

    Creates a xarray dataset for each apid and stores
    them in a dictionary

    Parameters
    ----------
    unpacked_packets : list[space_packet_parser.parser.Packet]
        A sorted list of decommutated packets
    """
    grouped_data = hit_l1a.group_data(unpacked_packets)
    skip_keys = [
        "shcoarse",
        "ground_sw_version",
        "packet_file_name",
        "ccsds_header",
        "leak_i_raw",
    ]
    datasets_by_apid = hit_l1a.create_datasets(grouped_data, "001", skip_keys=skip_keys)
    assert len(datasets_by_apid.keys()) == 1
    assert isinstance(datasets_by_apid[hit_l1a.HitAPID.HIT_HSKP], xr.Dataset)


def test_hit_l1a(packet_filepath):
    """Create L1A datasets

    Creates xarray datasets from a packet.

    Parameters
    ----------
    packet_filepath : str
        Path to ccsds file
    """
    datasets = hit_l1a.hit_l1a(packet_filepath, "001")
    assert len(datasets) == 1
    assert isinstance(datasets[0], xr.Dataset)
    assert datasets[0].attrs["Logical_source"] == "imap_hit_l1a_hk"


def test_total_datasets(unpacked_packets):
    """Test if total number of datasets is correct"""
    # assert len(unpacked_packets) == total_datasets


def test_dataset_dims_length(unpacked_packets):
    """Test if the time dimension length in the dataset is correct"""
    # grouped_data = hit_l1a.group_data(unpacked_packets)
    # assert grouped_data["hit_l1a.HitAPID.HIT_HSKP"].dims["epoch"] == num_packet_times
