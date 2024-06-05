import pathlib

import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import write_cdf
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
    sorted_packets = sorted(packets, key=lambda x: x.data["SHCOARSE"].derived_value)
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
    datasets = hit_l1a.create_datasets(grouped_data, skip_keys=skip_keys)
    assert len(datasets.keys()) == 1
    assert isinstance(datasets[1251], xr.Dataset)


def test_hit_l1a(packet_filepath):
    """Create L1A CDF files

    Creates a CDF file for each apid dataset and stores all
    cdf file paths in a list

    Parameters
    ----------
    packet_filepath : str
        Path to ccsds file
    """
    packets = hit_l1a.decom_packets(packet_filepath)
    sorted_packets = sorted(packets, key=lambda x: x.data["SHCOARSE"].derived_value)
    grouped_data = hit_l1a.group_data(sorted_packets)
    skip_keys = [
        "shcoarse",
        "ground_sw_version",
        "packet_file_name",
        "ccsds_header",
        "leak_i_raw",
    ]
    datasets = hit_l1a.create_datasets(grouped_data, skip_keys=skip_keys)
    cdf_filepaths = []
    for dataset in datasets.values():
        cdf_file = write_cdf(dataset)
        cdf_filepaths.append(cdf_file)
    assert len(cdf_filepaths) == 1
    assert isinstance(cdf_filepaths[0], pathlib.PurePath)
    assert cdf_filepaths[0].name == "imap_hit_l1a_hk_19700105_v001.cdf"
