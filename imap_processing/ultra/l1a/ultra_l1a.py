"""Contains code to perform ULTRA L1a cdf generation."""
# TODO: Evaluate naming conventions for fields and variables
# TODO: Improved short and long descriptions for each variable
# TODO: Improved var_notes for each variable
import dataclasses
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import calc_start_time, write_cdf
from imap_processing.ultra import ultra_cdf_attrs
from imap_processing.ultra.l0.decom_ultra import (
    ULTRA_AUX,
    ULTRA_EVENTS,
    ULTRA_TOF,
    decom_ultra_apids,
)

logger = logging.getLogger(__name__)


def initiate_data_arrays(decom_ultra: dict, apid: int):
    """Initiate xarray data arrays.

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
    # Converted time
    time_converted = []

    if apid == ULTRA_EVENTS.apid[0]:
        for time in decom_ultra["EVENTTIMES"]:
            time_converted.append(calc_start_time(time))
    elif apid == ULTRA_TOF.apid[0]:
        for time in np.unique(decom_ultra["SHCOARSE"]):
            time_converted.append(calc_start_time(time))
    else:
        for time in decom_ultra["SHCOARSE"]:
            time_converted.append(calc_start_time(time))

    epoch_time = xr.DataArray(
        time_converted,
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    if apid != ULTRA_TOF.apid[0]:
        dataset = xr.Dataset(
            coords={"epoch": epoch_time},
            attrs=ultra_cdf_attrs.ultra_l1a_attrs.output(),
        )
    else:
        row = xr.DataArray(
            # Number of pixel rows
            np.arange(54),
            name="row",
            dims=["row"],
            attrs=dataclasses.replace(
                ultra_cdf_attrs.ultra_metadata_attrs,
                catdesc="row",  # TODO: short and long descriptions
                fieldname="row",
            ).output(),
        )

        column = xr.DataArray(
            # Number of pixel columns
            np.arange(180),
            name="column",
            dims=["column"],
            attrs=dataclasses.replace(
                ultra_cdf_attrs.ultra_metadata_attrs,
                catdesc="column",  # TODO: short and long descriptions
                fieldname="column",
            ).output(),
        )

        sid = xr.DataArray(
            # Number of pixel columns
            np.arange(8),
            name="sid",
            dims=["sid"],
            attrs=dataclasses.replace(
                ultra_cdf_attrs.ultra_metadata_attrs,
                catdesc="sid",  # TODO: short and long descriptions
                fieldname="sid",
            ).output(),
        )

        dataset = xr.Dataset(
            coords={"epoch": epoch_time, "sid": sid, "row": row, "column": column},
            attrs=ultra_cdf_attrs.ultra_l1a_attrs.output(),
        )

    return dataset


def get_event_time(decom_ultra_dict: dict):
    """Get event times using data from events and aux packets.

    Parameters
    ----------
    decom_ultra_dict: dict
        Events and aux data.

    Returns
    -------
    decom_events : dict
        Ultra events data with calculated events timestamps.

    Equation for event time:
    t = t_(spin start) + t_(spin start sub)/1000 +
    t_(spin duration)/1000 * phase_angle/720
    """
    event_times, durations, spin_starts = ([] for _ in range(3))
    decom_aux = decom_ultra_dict[ULTRA_AUX.apid[0]]
    decom_events = decom_ultra_dict[ULTRA_EVENTS.apid[0]]

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


def create_dataset(decom_ultra_dict: dict):
    """Create xarray for packet.

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
            attrs = dataclasses.replace(
                ultra_cdf_attrs.ultra_support_attrs,
                catdesc=key.lower(),  # TODO: short and long descriptions
                fieldname=key.lower(),
                label_axis=key.lower(),
                depend_1="sid",
            ).output()
            dims = ["epoch", "sid"]
        # AUX enums require string attibutes
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
            attrs = dataclasses.replace(
                ultra_cdf_attrs.string_base,
                catdesc=key.lower(),  # TODO: short and long descriptions
                fieldname=key.lower(),
                depend_0="epoch",
            ).output()
            dims = ["epoch"]
        # TOF packetdata has multiple dimensions
        elif key == "PACKETDATA":
            attrs = dataclasses.replace(
                ultra_cdf_attrs.ultra_support_attrs,
                catdesc=key.lower(),  # TODO: short and long descriptions
                fieldname=key.lower(),
                label_axis=key.lower(),
                depend_1="sid",
                depend_2="row",
                depend_3="column",
                units="pixels",
                variable_purpose="primary_var",
            ).output()
            dims = ["epoch", "sid", "row", "column"]
        # Use metadata with a single dimension for
        # all other data products
        else:
            attrs = dataclasses.replace(
                ultra_cdf_attrs.ultra_support_attrs,
                catdesc=key.lower(),  # TODO: short and long descriptions
                fieldname=key.lower(),
                label_axis=key.lower(),
            ).output()
            dims = ["epoch"]

        dataset[key] = xr.DataArray(
            value,
            name=key.lower(),
            dims=dims,
            attrs=attrs,
        )

    return dataset


def ultra_l1a(packet_file: dict, xtce: Path, output_filepath: Path):
    """
    Process ULTRA L0 data into L1A CDF files at output_filepath.

    Parameters
    ----------
    packet_file : dict
        Dictionary containing paid and path to the CCSDS data packet file.
    xtce : Path
        Path to the XTCE packet definition file.
    output_filepath : Path
        Full directory and filename for CDF file
    """
    if ULTRA_EVENTS.apid[0] in packet_file.keys():
        # For events data we need aux data to calculate event times
        apid = ULTRA_EVENTS.apid[0]
        decom_ultra_events = decom_ultra_apids(packet_file[apid], xtce, apid)
        decom_ultra_aux = decom_ultra_apids(
            packet_file[ULTRA_AUX.apid[0]], xtce, ULTRA_AUX.apid[0]
        )
        decom_ultra_dict = {
            ULTRA_EVENTS.apid[0]: decom_ultra_events,
            ULTRA_AUX.apid[0]: decom_ultra_aux,
        }
    else:
        apid = next(iter(packet_file.keys()))
        decom_ultra_dict = {apid: decom_ultra_apids(packet_file[apid], xtce, apid)}

    dataset = create_dataset(decom_ultra_dict)
    write_cdf(dataset, Path(output_filepath))
    logging.info(f"Created CDF file at {output_filepath}")
