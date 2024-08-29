"""Decommutates i-alirt packets and creates L1 data products."""

import logging

import xarray as xr

from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)


def generate_xarray(packet_file: str, xtce: str) -> dict[int, xr.Dataset]:
    """
    Generate xarray from unpacked data.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    xtce : str
        Path to the XTCE packet definition file.

    Returns
    -------
    alirt_dict : dict
        A dictionary of the dataset containing the decoded data fields.
    """
    alirt_dict = packet_file_to_datasets(packet_file, xtce, use_derived_value=False)

    return alirt_dict
