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
from imap_processing.spice.time import met_to_j2000ns
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
        raw_time = decom_ultra["EVENTTIMES"]
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
        met_to_j2000ns(raw_time),
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


def get_event_time(decom_ultra_dict: dict) -> dict:
    """
    Get event times using data from events and aux packets.

    Parameters
    ----------
    decom_ultra_dict : dict
        Events and aux data.

    Returns
    -------
    decom_events : dict
        Ultra events data with calculated events timestamps.

    Notes
    -----
    Equation for event time:
    t = t_(spin start) + t_(spin start sub)/1000 +
    t_(spin duration)/1000 * phase_angle/720
    """
    event_times, durations, spin_starts = ([] for _ in range(3))
    decom_aux = decom_ultra_dict[ULTRA_AUX.apid[0]]
    decom_events: dict = decom_ultra_dict[ULTRA_EVENTS.apid[0]]

    timespinstart_array = np.array(decom_aux["TIMESPINSTART"])
    timespinstartsub_array = np.array(decom_aux["TIMESPINSTARTSUB"]) / 1000

    # spin start according to aux data
    aux_spin_starts = timespinstart_array + timespinstartsub_array

    for time in np.unique(decom_events["SHCOARSE"]):
        # Get the nearest spin start and duration prior to the event
        spin_start = aux_spin_starts[aux_spin_starts <= time][-1]
        duration = np.array(decom_aux["DURATION"])[aux_spin_starts <= time][-1]

        # Find the events
        event_indices = np.where(np.array(decom_events["SHCOARSE"]) == time)

        for event_index in event_indices[0]:
            phase_angle = decom_events["PHASE_ANGLE"][event_index]

            durations.append(duration)
            spin_starts.append(spin_start)

            # If there were no events, the time is set to 'SHCOARSE'
            if decom_events["COUNT"][event_index] == 0:
                event_times.append(decom_events["SHCOARSE"][event_index])
            else:
                event_times.append(spin_start + (duration / 1000) * (phase_angle / 720))

    decom_events["DURATION"] = durations
    decom_events["TIMESPINSTART"] = spin_starts
    decom_events["EVENTTIMES"] = event_times

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
        decom_ultra = get_event_time(decom_ultra_dict)
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
    # 1. For testing purposes to only generate a dataset for a single apid.
    #    Each test dataset is only for a single apid while the rest of the apids
    #    contain zeros. Ideally we would have
    #    test data for all apids and remove this parameter.
    # 2. When we are generating the l1a dataset for the events packet since
    #    right now we need to combine the events and aux packets to get the
    #    correct event timestamps (get_event_time). This part will change
    #    when we begin using the spin table in the database instead of the aux packet.
    if apid is not None:
        apids = [apid]
    else:
        apids = list(grouped_data.keys())

    for apid in apids:
        if apid == ULTRA_EVENTS.apid[0]:
            decom_ultra_dict = {
                apid: process_ultra_apids(grouped_data[apid], apid),
                ULTRA_AUX.apid[0]: process_ultra_apids(
                    grouped_data[ULTRA_AUX.apid[0]], ULTRA_AUX.apid[0]
                ),
            }
        else:
            decom_ultra_dict = {
                apid: process_ultra_apids(grouped_data[apid], apid),
            }
        dataset = create_dataset(decom_ultra_dict)
        # TODO: move this to use ImapCdfAttributes().add_global_attribute()
        dataset.attrs["Data_version"] = data_version
        output_datasets.append(dataset)

    return output_datasets
