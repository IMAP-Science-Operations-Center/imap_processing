"""Generate ULTRA L1a CDFs."""

# TODO: Evaluate naming conventions for fields and variables
# TODO: Improved short and long descriptions for each variable
# TODO: Improved var_notes for each variable
import logging
from typing import Optional

import numpy as np
import xarray as xr

from imap_processing import decom, imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.ultra.l0.decom_ultra import process_ultra_apids
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
    ULTRA_RATES,
    ULTRA_TOF,
)
from imap_processing.utils import group_by_apid

logger = logging.getLogger(__name__)


def initiate_data_arrays(decom_ultra: dict, apid: int) -> xr.Dataset:
    """
    Initiate xarray data arrays.

    Parameters
    ----------
    decom_ultra : dict
        Parsed data.
    apid : int
        Packet APID.

    Returns
    -------
    dataset : xarray.Dataset
        Data in xarray format.
    """
    if apid in ULTRA_EVENTS.apid:
        index = ULTRA_EVENTS.apid.index(apid)
        logical_source = ULTRA_EVENTS.logical_source[index]
        addition_to_logical_desc = ULTRA_EVENTS.addition_to_logical_desc
        raw_time = decom_ultra["SHCOARSE"]
    elif apid in ULTRA_TOF.apid:
        index = ULTRA_TOF.apid.index(apid)
        logical_source = ULTRA_TOF.logical_source[index]
        addition_to_logical_desc = ULTRA_TOF.addition_to_logical_desc
        raw_time = np.unique(decom_ultra["SHCOARSE"])
    elif apid in ULTRA_AUX.apid:
        index = ULTRA_AUX.apid.index(apid)
        logical_source = ULTRA_AUX.logical_source[index]
        addition_to_logical_desc = ULTRA_AUX.addition_to_logical_desc
        raw_time = decom_ultra["SHCOARSE"]
    elif apid in ULTRA_RATES.apid:
        index = ULTRA_RATES.apid.index(apid)
        logical_source = ULTRA_RATES.logical_source[index]
        addition_to_logical_desc = ULTRA_RATES.addition_to_logical_desc
        raw_time = decom_ultra["SHCOARSE"]
    else:
        raise ValueError(f"APID {apid} not recognized.")

    # Load the CDF attributes
    cdf_manager = ImapCdfAttributes()
    cdf_manager.add_instrument_global_attrs("ultra")
    cdf_manager.add_instrument_variable_attrs("ultra", "l1a")

    epoch_time = xr.DataArray(
        met_to_ttj2000ns(raw_time),
        name="epoch",
        dims=["epoch"],
        attrs=cdf_manager.get_variable_attributes("epoch"),
    )

    sci_cdf_attrs = cdf_manager.get_global_attributes("imap_ultra_l1a_sci")
    # replace the logical source and logical source description
    sci_cdf_attrs["Logical_source"] = logical_source
    sci_cdf_attrs["Logical_source_desc"] = (
        f"IMAP Mission ULTRA Instrument Level-1A {addition_to_logical_desc} Data"
    )

    if apid not in (ULTRA_TOF.apid[0], ULTRA_TOF.apid[1]):
        dataset = xr.Dataset(
            coords={"epoch": epoch_time},
            attrs=sci_cdf_attrs,
        )
    else:
        row = xr.DataArray(
            # Number of pixel rows
            np.arange(54),
            name="row",
            dims=["row"],
            attrs=cdf_manager.get_variable_attributes("ultra_metadata_attrs"),
        )

        column = xr.DataArray(
            # Number of pixel columns
            np.arange(180),
            name="column",
            dims=["column"],
            attrs=cdf_manager.get_variable_attributes("ultra_metadata_attrs"),
        )

        sid = xr.DataArray(
            # Number of pixel columns
            np.arange(8),
            name="sid",
            dims=["sid"],
            attrs=cdf_manager.get_variable_attributes("ultra_metadata_attrs"),
        )

        dataset = xr.Dataset(
            coords={"epoch": epoch_time, "sid": sid, "row": row, "column": column},
            attrs=sci_cdf_attrs,
        )

    return dataset


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


def create_dataset(decom_ultra_dict: dict) -> xr.Dataset:
    """
    Create xarray for packet.

    Parameters
    ----------
    decom_ultra_dict : dict
        Dictionary of parsed data.

    Returns
    -------
    dataset : xarray.Dataset
        Data in xarray format.
    """
    # Combine events and aux datasets so we can have proper event timestamps
    if ULTRA_EVENTS.apid[0] in decom_ultra_dict.keys():
        apid = ULTRA_EVENTS.apid[0]
        decom_ultra = get_event_id(decom_ultra_dict)
    else:
        apid = next(iter(decom_ultra_dict.keys()))
        decom_ultra = decom_ultra_dict[apid]

    # Load the CDF attributes
    # TODO: call this once and pass the object to the function
    cdf_manager = ImapCdfAttributes()
    cdf_manager.add_instrument_global_attrs("ultra")
    cdf_manager.add_instrument_variable_attrs("ultra", "l1a")

    dataset = initiate_data_arrays(decom_ultra, apid)

    for key, value in decom_ultra.items():
        # EVENT DATA and FASTDATA_00 have been broken down further
        # (see ultra_utils.py) and are therefore not needed.
        # SID is also not needed as it is used as a dimension.
        if key in {"EVENTDATA", "FASTDATA_00", "SID"}:
            continue
        # Everything in the TOF packet has dimensions of (time, sid) except
        # for PACKETDATA which has dimensions of (time, sid, row, column) and
        # SHCOARSE with has dimensions of (time)
        elif apid == ULTRA_TOF.apid[0] and key != "PACKETDATA" and key != "SHCOARSE":
            # TODO: fix this to use the correct attributes
            attrs = cdf_manager.get_variable_attributes("ultra_support_attrs")
            dims = ["epoch", "sid"]
        # AUX enums require string attributes
        elif key in [
            "SPINPERIODVALID",
            "SPINPHASEVALID",
            "SPINPERIODSOURCE",
            "CATBEDHEATERFLAG",
            "HWMODE",
            "IMCENB",
            "LEFTDEFLECTIONCHARGE",
            "RIGHTDEFLECTIONCHARGE",
        ]:
            # TODO: fix this to use the correct attributes
            attrs = cdf_manager.get_variable_attributes("string_base_attrs")
            dims = ["epoch"]
        # TOF packetdata has multiple dimensions
        elif key == "PACKETDATA":
            # TODO: fix this to use the correct attributes
            attrs = cdf_manager.get_variable_attributes("packet_data_attrs")
            dims = ["epoch", "sid", "row", "column"]
        # Use metadata with a single dimension for
        # all other data products
        else:
            # TODO: fix this to use the correct attributes
            attrs = cdf_manager.get_variable_attributes("ultra_support_attrs")
            dims = ["epoch"]

        dataset[key] = xr.DataArray(
            value,
            name=key if key == "epoch" else key.lower(),
            dims=dims,
            attrs=attrs,
        )

    return dataset


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

    packets = decom.decom_packets(packet_file, xtce)
    grouped_data = group_by_apid(packets)

    output_datasets = []

    # This is used for two purposes currently:
    #    For testing purposes to only generate a dataset for a single apid.
    #    Each test dataset is only for a single apid while the rest of the apids
    #    contain zeros. Ideally we would have
    #    test data for all apids and remove this parameter.
    if apid is not None:
        apids = [apid]
    else:
        apids = list(grouped_data.keys())

    for apid in apids:
        decom_ultra_dict = {
            apid: process_ultra_apids(grouped_data[apid], apid),
        }
        dataset = create_dataset(decom_ultra_dict)
        # TODO: move this to use ImapCdfAttributes().add_global_attribute()
        dataset.attrs["Data_version"] = data_version
        output_datasets.append(dataset)

    return output_datasets
