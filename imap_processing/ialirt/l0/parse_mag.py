"""Functions to support I-ALiRT MAG packet parsing."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.utils import load_cdf
from imap_processing.ialirt.l0.mag_l0_ialirt_data import (
    Packet0,
    Packet1,
    Packet2,
    Packet3,
)
from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.mag.l1a.mag_l1a_data import TimeTuple
from imap_processing.mag.l1b.mag_l1b import calibrate_vector, shift_time
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)


def get_pkt_counter(status_values: xr.DataArray) -> xr.DataArray:
    """
    Get the packet counters.

    Parameters
    ----------
    status_values : xr.DataArray
        Status data.

    Returns
    -------
    pkt_counters : xr.DataArray
        Packet counters.
    """
    # mag_status is a 24 bit unsigned field
    # The leading 2 bits of STATUS are a 2 bit 0-3 counter
    pkt_counter = (status_values >> 22) & 0x03

    return pkt_counter


def get_status_data(status_values: xr.DataArray, pkt_counters: xr.DataArray) -> dict:
    """
    Get the status data.

    Parameters
    ----------
    status_values : xr.DataArray
        Status data.
    pkt_counters : xr.DataArray
        Packet counters.

    Returns
    -------
    combined_packets : dict
        Decoded packets.
    """
    decoders = {
        0: Packet0,
        1: Packet1,
        2: Packet2,
        3: Packet3,
    }

    combined_packets = {}

    for pkt_num, decoder in decoders.items():
        status_subset = status_values[pkt_counters == pkt_num]
        decoded_packet = decoder(int(status_subset))
        combined_packets.update(vars(decoded_packet))

    return combined_packets


def get_bytes(val: int) -> list[int]:
    """
    Extract three bytes from a 24-bit integer.

    Parameters
    ----------
    val : int
        24-bit integer value.

    Returns
    -------
    list[int]
        List of three extracted bytes.
    """
    return [
        (val >> 16) & 0xFF,  # Most significant byte (Byte2)
        (val >> 8) & 0xFF,  # Middle byte (Byte1)
        (val >> 0) & 0xFF,  # Least significant byte (Byte0)
    ]


def extract_magnetic_vectors(science_values: xr.DataArray) -> dict:
    """
    Extract the magnetic vectors.

    Parameters
    ----------
    science_values : xr.DataArray
        Science data.

    Returns
    -------
    vectors : dict
        Magnetic vectors.
    """
    # Primary sensor:
    pri_x = (int(science_values[0]) >> 8) & 0xFFFF
    pri_y = ((int(science_values[0]) << 8) & 0xFF00) | (
        (int(science_values[1]) >> 16) & 0xFF
    )
    pri_z = int(science_values[1]) & 0xFFFF

    # Secondary sensor:
    sec_x = (int(science_values[2]) >> 8) & 0xFFFF
    sec_y = ((int(science_values[2]) << 8) & 0xFF00) | (
        (int(science_values[3]) >> 16) & 0xFF
    )
    sec_z = int(science_values[3]) & 0xFFFF

    vectors = {
        "pri_x": pri_x,
        "pri_y": pri_y,
        "pri_z": pri_z,
        "sec_x": sec_x,
        "sec_y": sec_y,
        "sec_z": sec_z,
    }

    return vectors


def get_time(
    grouped_data: xr.Dataset,
    group: int,
    pkt_counter: xr.DataArray,
    time_shift_mago: int,
    time_shift_magi: int,
) -> dict:
    """
    Get the time for the grouped data.

    Parameters
    ----------
    grouped_data : xr.Dataset
        Grouped data.
    group : int
        Group number.
    pkt_counter : xr.DataArray
        Packet counter.
    time_shift_mago : int
        Time shift value mago.
    time_shift_magi : int
        Time shift value magi.

    Returns
    -------
    time_data : dict
        Coarse and fine time for Primary and Secondary Sensors.
    """
    pri_coarsetm = grouped_data["mag_acq_tm_coarse"][
        (grouped_data["group"] == group).values
    ][pkt_counter == 0]

    pri_fintm = grouped_data["mag_acq_tm_fine"][
        (grouped_data["group"] == group).values
    ][pkt_counter == 0]

    sec_coarsetm = grouped_data["mag_acq_tm_coarse"][
        (grouped_data["group"] == group).values
    ][pkt_counter == 2]

    sec_fintm = grouped_data["mag_acq_tm_fine"][
        (grouped_data["group"] == group).values
    ][pkt_counter == 2]

    time_data = {
        "pri_coarsetm": int(pri_coarsetm),
        "pri_fintm": int(pri_fintm),
        "sec_coarsetm": int(sec_coarsetm),
        "sec_fintm": int(sec_fintm),
    }

    primary_time = TimeTuple(time_data["pri_coarsetm"], time_data["pri_fintm"])
    secondary_time = TimeTuple(time_data["sec_coarsetm"], time_data["sec_fintm"])
    time_data["pri_met"] = primary_time.to_seconds()
    time_data["primary_ttj2000ns"] = met_to_ttj2000ns(time_data["pri_met"])
    # TODO: is this ok?
    time_data["primary_epoch"] = shift_time(
        time_data["primary_ttj2000ns"], time_shift_mago
    )
    time_data["sec_met"] = secondary_time.to_seconds()
    time_data["secondary_ttj2000ns"] = met_to_ttj2000ns(time_data["sec_met"])
    time_data["secondary_epoch"] = shift_time(
        time_data["secondary_ttj2000ns"], time_shift_magi
    )

    return time_data


def calculate_l1b(
    grouped_data,
    group: int,
    pkt_counter: xr.DataArray,
    science_data: dict,
    status_data: dict,
):
    # Get calibration data
    (
        calibration_matrix_mago,
        time_shift_mago,
        calibration_matrix_magi,
        time_shift_magi,
    ) = get_calibration()

    # Get time values for each group.
    time_data = get_time(
        grouped_data, group, pkt_counter, time_shift_mago, time_shift_magi
    )

    input_vector_mago = np.array(
        [
            science_data["pri_x"],
            science_data["pri_y"],
            science_data["pri_z"],
            status_data["fob_range"],
        ]
    )
    input_vector_magi = np.array(
        [
            science_data["sec_x"],
            science_data["sec_y"],
            science_data["sec_z"],
            status_data["fib_range"],
        ]
    )

    updated_vector_mago = calibrate_vector(input_vector_mago, calibration_matrix_mago)
    updated_vector_magi = calibrate_vector(input_vector_magi, calibration_matrix_magi)

    return updated_vector_mago, updated_vector_magi, time_data


def process_packet(accumulated_data: xr.Dataset) -> list[dict]:
    """
    Parse the MAG packets.

    Parameters
    ----------
    accumulated_data : xr.Dataset
        Packets dataset accumulated over 1 min.

    Returns
    -------
    mag_data : list[dict]
        Dictionaries of the parsed data product.
    """
    logger.info(
        f"Parsing MAG for time: {accumulated_data['mag_acq_tm_coarse'].min().values} - "
        f"{accumulated_data['mag_acq_tm_coarse'].max().values}."
    )

    # Note that the fine time second is split into 65535.
    time_seconds = calculate_time(
        accumulated_data["mag_acq_tm_coarse"],
        accumulated_data["mag_acq_tm_fine"],
        65535,
    )

    # Add required parameters.
    accumulated_data["time_seconds"] = time_seconds
    sorted_data = accumulated_data.sortby("time_seconds", ascending=True)
    pkt_counter = get_pkt_counter(sorted_data["mag_status"])
    sorted_data["pkt_counter"] = pkt_counter

    grouped_data = find_groups(sorted_data, (0, 3), "pkt_counter", "time_seconds")

    unique_groups = np.unique(grouped_data["group"])
    mag_data = []

    for group in unique_groups:
        # Get status values for each group.
        status_values = grouped_data["mag_status"][
            (grouped_data["group"] == group).values
        ]
        pkt_counter = grouped_data["pkt_counter"][
            (grouped_data["group"] == group).values
        ]

        if not np.array_equal(pkt_counter, np.arange(4)):
            logger.warning(
                f"Group {group} does not contain all values from 0 to "
                f"3 without duplicates."
            )
            continue

        # Get decoded status data.
        status_data = get_status_data(status_values, pkt_counter)

        # Get science values for each group.
        science_values = grouped_data["mag_data"][
            (grouped_data["group"] == group).values
        ]
        science_data = extract_magnetic_vectors(science_values)
        updated_vector_mago, updated_vector_magi, time_data = calculate_l1b(
            grouped_data, group, pkt_counter, science_data, status_data
        )

        # TODO: do I need to add the range values to the science data?
        # TODO: how do we know if it is mago or magi?
        science_data.update(
            {
                "calibrated_pri_x": updated_vector_mago[0],
                "calibrated_pri_y": updated_vector_mago[1],
                "calibrated_pri_z": updated_vector_mago[2],
            }
        )

        mag_data.append({**status_data, **science_data, **time_data})

    return mag_data


def get_calibration():
    # TODO: This should definitely be loaded from AWS
    calibration_dataset = load_cdf(
        Path(__file__).resolve().parents[2]
        / "mag"
        / "l1b"
        / "imap_calibration_mag_20240229_v01.cdf"
    )
    logger.info("Using default test calibration file.")

    calibration_matrix_mago = calibration_dataset["MFOTOURFO"]
    time_shift_mago = calibration_dataset["OTS"]
    calibration_matrix_magi = calibration_dataset["MFITOURFI"]
    time_shift_magi = calibration_dataset["ITS"]

    return (
        calibration_matrix_mago,
        time_shift_mago,
        calibration_matrix_magi,
        time_shift_magi,
    )
