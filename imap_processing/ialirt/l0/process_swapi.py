"""Functions to support I-ALiRT SWAPI processing."""

import logging

import xarray as xr
from xarray import DataArray

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ialirt.utils.grouping import find_groups

# from imap_processing.swapi.l1.swapi_l1 import process_sweep_data
from imap_processing.swapi.swapi_utils import SWAPIAPID
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)


TIME_PER_BIN = 0.167  # seconds


def process_swapi_ialirt(file_path: str) -> dict[str, DataArray]:
    """
    Process L0 data, extract I-ALiRT variables and calculate coincidence count rate.

    Parameters
    ----------
    file_path : str
        L0 SWAPI file.

    Returns
    -------
    swapi_data : dict
        Dictionary containing all data variables for SWAPI I-ALiRT product.
    """
    cdf_manager = ImapCdfAttributes()

    # Packet definition
    xtce_definition = (
        f"{imap_module_directory}/ialirt/packet_definitions/ialirt_swapi.xml"
    )
    unpacked_data = packet_file_to_datasets(file_path, xtce_definition)
    sci_dataset = unpacked_data[SWAPIAPID.SWP_IALIRT]

    grouped_dataset = find_groups(sci_dataset, (0, 11), "seq_number", "swapi_acq")

    total_packets = len(grouped_dataset["seq_number"].data)

    # It takes 12 sequence data to make one full sweep
    total_sequence = 12
    total_full_sweeps = total_packets // total_sequence

    # Separate out the epoch time associated with each sweep, taken from the 1st packet
    epoch_values = grouped_dataset["epoch"].data.reshape(total_full_sweeps, 12)[:, 0]

    filtered_epoch = xr.DataArray(
        epoch_values,
        name="epoch",
        dims=["epoch"],
        attrs=cdf_manager.get_variable_attributes("epoch", check_schema=False),
    )

    # raw_coin_count = process_sweep_data(grouped_dataset, "coin_cnt")
    # raw_coin_rate = raw_coin_count / TIME_PER_BIN

    swapi_data = {
        "epoch_time": filtered_epoch,
        # more variables to go here
    }

    return swapi_data
