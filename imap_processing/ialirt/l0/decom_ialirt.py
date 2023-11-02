import collections
import logging

import xarray as xr

from imap_processing.decom import decom_packets

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def generate_xarray(packet_file: str, xtce: str):
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
    xr.Dataset
        A dataset containing the decoded data fields with 'time' as the coordinating
        dimension.
    """
    try:
        packets = decom_packets(packet_file, xtce)
    except Exception as e:
        logger.error(f"Error during packet decomposition: {str(e)}")
        return

    logger.info(f"Decommutated {len(packets)} packets from {packet_file}.")

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
    data_storage = {inst: collections.defaultdict(list) for inst in instruments}

    for packet in packets:
        for key, value in packet.data.items():
            key_matched = False
            for inst in instruments:
                if key.startswith(inst):
                    # Directly append to the list without checking if the key exists
                    data_storage[inst][key].append(value.derived_value)
                    key_matched = True
                    break

        if not key_matched:
            # If after checking all instruments, none match, then log a warning
            logger.warning(f"Unexpected key '{key}' found in packet data.")

    logger.debug("Generating datasets for each instrument.")

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
