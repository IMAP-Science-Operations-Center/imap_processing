"""Decommutate IDEX CCSDS packets."""

import logging
from pathlib import Path
from typing import Any, Union

from xarray import Dataset

from imap_processing import decom, imap_module_directory
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)


def decom_packets(
    packet_file: Union[str, Path],
) -> tuple[list[Any], dict[int, Dataset], dict[int, Dataset]]:
    """
    Decom IDEX data packets using IDEX packet definition.

    Parameters
    ----------
    packet_file : pathlib.Path | str
        String to data packet path with filename.

    Returns
    -------
    Tuple[list, dict]
        Returns a list of all unpacked science data and a dictionary of datasets
        indexed by their APIDs, one for raw and derived values.

    Notes
    -----
    The function 'packet_file_to_dataset' does not work with IDEX science packets due to
    branching logic within the science xml file. The science data and housekeeping data
    will be decommed separately and both returned from this function.
    """
    xtce_base_path = f"{imap_module_directory}/idex/packet_definitions"
    science_xtce_file = f"{xtce_base_path}/idex_science_packet_definition.xml"
    hk_xtce_file = f"{xtce_base_path}/idex_housekeeping_packet_definition.xml"

    science_decom_packet_list = decom.decom_packets(packet_file, science_xtce_file)
    raw_datasets_by_apid = packet_file_to_datasets(
        packet_file, hk_xtce_file, use_derived_value=False
    )
    derived_datasets_by_apid = packet_file_to_datasets(
        packet_file, hk_xtce_file, use_derived_value=True
    )

    return (
        list(science_decom_packet_list),
        raw_datasets_by_apid,
        derived_datasets_by_apid,
    )
