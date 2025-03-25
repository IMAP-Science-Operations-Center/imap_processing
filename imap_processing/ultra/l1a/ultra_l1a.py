"""Generate ULTRA L1a CDFs."""

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ultra.l0.decom_ultra import (
    process_ultra_events,
    process_ultra_rates,
    process_ultra_tof,
)
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
    ULTRA_RATES,
    ULTRA_TOF,
)
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)


def get_event_id(decom_ultra_dict: dict) -> dict:
    """
    Get unique event IDs using data from events packets.

    Parameters
    ----------
    decom_ultra_dict : dict
        Events data.

    Returns
    -------
    decom_events : dict
        Ultra events data with calculated unique event IDs as 64-bit integers.
    """
    decom_events: dict = decom_ultra_dict[ULTRA_EVENTS.apid[0]]

    event_ids = []
    packet_counters = {}

    for met in decom_events["SHCOARSE"]:
        # Initialize the counter for a new packet (MET value)
        if met not in packet_counters:
            packet_counters[met] = 0
        else:
            packet_counters[met] += 1

        # Left shift SHCOARSE (u32) by 31 bits, to make room for our event counters
        # (31 rather than 32 to keep it positive in the int64 representation)
        # Append the current number of events in this packet to the right-most bits
        # This makes each event a unique value including the MET and event number
        # in the packet
        # NOTE: CDF does not allow for uint64 values,
        # so we use int64 representation here
        event_id = (np.int64(met) << np.int64(31)) | np.int64(packet_counters[met])
        event_ids.append(event_id)

    decom_events["EVENTID"] = event_ids

    return decom_events


def ultra_l1a(
    packet_file: str, data_version: str, apid: Optional[int] = None
) -> list[xr.Dataset]:
    """
    Will process ULTRA L0 data into L1A CDF files at output_filepath.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    data_version : str
        Version of the data product being created.
    apid : Optional[int]
        Optional apid.

    Returns
    -------
    output_datasets : list[xarray.Dataset]
        List of xarray.Dataset.
    """
    xtce = str(
        f"{imap_module_directory}/ultra/packet_definitions/" f"ULTRA_SCI_COMBINED.xml"
    )

    datasets_by_apid = packet_file_to_datasets(packet_file, xtce)

    output_datasets = []

    # This is used for two purposes currently:
    #    For testing purposes to only generate a dataset for a single apid.
    #    Each test dataset is only for a single apid while the rest of the apids
    #    contain zeros. Ideally we would have
    #    test data for all apids and remove this parameter.
    if apid is not None:
        apids = [apid]
    else:
        apids = list(datasets_by_apid.keys())

    for apid in apids:
        if apid in ULTRA_AUX.apid:
            decom_ultra_dataset = datasets_by_apid[apid]
            gattr_key = ULTRA_AUX.logical_source[ULTRA_AUX.apid.index(apid)]
        elif apid in ULTRA_TOF.apid:
            decom_ultra_dataset = process_ultra_tof(
                datasets_by_apid[apid], defaultdict(list)
            )
            gattr_key = ULTRA_TOF.logical_source[ULTRA_TOF.apid.index(apid)]
        elif apid in ULTRA_RATES.apid:
            decom_ultra_dataset = process_ultra_rates(
                datasets_by_apid[apid], defaultdict(list)
            )
            gattr_key = ULTRA_RATES.logical_source[ULTRA_RATES.apid.index(apid)]
        elif apid in ULTRA_EVENTS.apid:
            decom_ultra_dataset = process_ultra_events(
                datasets_by_apid[apid], defaultdict(list)
            )
            gattr_key = ULTRA_EVENTS.logical_source[ULTRA_EVENTS.apid.index(apid)]

        # Update dataset global attributes
        attr_mgr = ImapCdfAttributes()
        attr_mgr.add_instrument_global_attrs("ultra")
        attr_mgr.add_global_attribute("Data_version", data_version)
        decom_ultra_dataset.attrs.update(attr_mgr.get_global_attributes(gattr_key))

        for key in decom_ultra_dataset.data_vars:
            attr_mgr.add_instrument_variable_attrs("ultra", "l1a")
            attrs = attr_mgr.get_variable_attributes(key.lower())
            decom_ultra_dataset.data_vars[key].attrs.update(attrs)

        output_datasets.append(decom_ultra_dataset)

    return output_datasets
