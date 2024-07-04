"""Decommutates i-alert packets and creates L1 data products."""

import collections
import logging
from typing import Optional

import xarray as xr

from imap_processing.decom import decom_packets

logger = logging.getLogger(__name__)


def generate_xarray(
    packet_file: str, xtce: str, time_keys: Optional[dict] = None
) -> xr.Dataset:
    """
    Generate xarray from unpacked data.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    xtce : str
        Path to the XTCE packet definition file.
    time_keys : dict
        Keys used for creating xarray dataset.

    Returns
    -------
    dataset : xarray.Dataset
        A dataset containing the decoded data fields with 'time' as the coordinating
        dimension.

    Examples
    --------
    # This is an example of what the xarray dataset might look like
    # after being processed by this function.

    <xarray.Dataset>
    Dimensions:       (SC_SCLK_SEC: 5)
    Coordinates:
      * SC_SCLK_SEC   (SC_SCLK_SEC) int64 322168 322169 322170 322171 322172
    Data variables:
        SC_MAG_STATUS (SC_SCLK_SEC) int64 0 1 0 1 0
        SC_HIT_STATUS (SC_SCLK_SEC) int64 1 0 1 0 1

    This example shows a dataset with 'SC_SCLK_SEC' as the coordinate
    and two data variables 'SC_MAG_STATUS' and 'SC_HIT_STATUS'.
    """
    packets = decom_packets(packet_file, xtce)

    logger.info(f"Decommutated {len(packets)} packets from {packet_file}.")

    if time_keys is None:
        time_keys = {
            "SC": "SC_SCLK_SEC",
            "HIT": "HIT_SC_TICK",
            "MAG": "MAG_ACQ",
            "COD_LO": "COD_LO_ACQ",
            "COD_HI": "COD_HI_ACQ",
            "SWE": "SWE_ACQ_SEC",
            "SWAPI": "SWAPI_ACQ",
        }

    instruments = list(time_keys.keys())

    # Initialize storage dictionary using defaultdict
    data_storage: dict = {inst: collections.defaultdict(list) for inst in instruments}

    for packet in packets:
        for key, value in packet.data.items():
            key_matched = False
            for inst in instruments:
                if key.startswith(inst):
                    # Directly append to the list
                    data_storage[inst][key].append(value.derived_value)
                    key_matched = True
                    break

            if not key_matched:
                # If after checking all instruments, none match, raise an error.
                raise ValueError(f"Unexpected key '{key}' found in packet data.")

    logger.info("Generating datasets for each instrument.")

    # Generate xarray dataset for each instrument and spacecraft
    datasets = {}
    for inst in instruments:
        dataset_dict = {
            key: (time_keys[inst], data_storage[inst][key])
            for key in data_storage[inst]
            if key != time_keys[inst]
        }
        datasets[inst] = xr.Dataset(
            dataset_dict, coords={time_keys[inst]: data_storage[inst][time_keys[inst]]}
        )

    return datasets
