"""Methods for decomming packets, processing to level 1A, and writing CDFs for MAG."""

import dataclasses
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import calc_start_time, write_cdf
from imap_processing.mag import mag_cdf_attrs
from imap_processing.mag.l0 import decom_mag
from imap_processing.mag.l0.mag_l0_data import MagL0
from imap_processing.mag.l1a.mag_l1a_data import (
    MagL1a,
    MagL1aPacketProperties,
    TimeTuple,
)
from imap_processing.mag.mag_cdf_attrs import DataMode, MagGlobalCdfAttributes, Sensor

logger = logging.getLogger(__name__)


def mag_l1a(packet_filepath, data_version: str) -> list[Path]:
    """
    Process MAG L0 data into L1A CDF files at cdf_filepath.

    Parameters
    ----------
    data_version: str
        Version of the CDF file to output, in the format "vXXX"
    packet_filepath : Path
        Packet files for processing

    Returns
    -------
    generated_files: list[Path]
        A list of generated filenames
    """
    packets = decom_mag.decom_packets(packet_filepath)

    norm_data = packets["norm"]
    burst_data = packets["burst"]

    input_files = [packet_filepath.name]

    generated_files = process_and_write_data(
        norm_data, DataMode.NORM, input_files, data_version
    )
    generated_files += process_and_write_data(
        burst_data, DataMode.BURST, input_files, data_version
    )

    return generated_files


def process_and_write_data(
    packet_data: list[MagL0],
    data_mode: DataMode,
    input_files: list[str],
    data_version: str,
) -> list[Path]:
    """
    Process MAG L0 data into L1A, then create and write out CDF files.

    Norm and burst mode descriptors are distinguished with the passed in attrs.

    Parameters
    ----------
    data_version: str
        Version of the CDF file to output, in the format "vXXX"
    packet_data: list[MagL0]
        List of MagL0 packets to process, containing primary and secondary sensor data
    data_mode: DataMode
        Enum for distinguishing between norm and burst mode data
    input_files: list[str]
        List of dependent filenames for generating the CDF files.

    Returns
    -------
    generated_files: list[Path]
        A list of generated filenames
    """
    if not packet_data:
        return []

    generation_date = np.datetime64(
        "now",
    ).astype(str)

    mag_raw = decom_mag.generate_dataset(
        packet_data,
        MagGlobalCdfAttributes(
            data_mode, Sensor.RAW, generation_date, input_files, data_version
        ).attribute_dict,
    )

    filepath = write_cdf(mag_raw)
    logger.info(f"Created RAW CDF file at {filepath}")

    generated_files = [filepath]

    l1a = process_packets(packet_data)

    # TODO: Rearrange generate_dataset to combine these two for loops
    for _, mago in l1a["mago"].items():
        norm_mago_output = generate_dataset(
            mago,
            MagGlobalCdfAttributes(
                data_mode, Sensor.MAGO, generation_date, input_files, data_version
            ).attribute_dict,
        )
        filepath = write_cdf(norm_mago_output)
        logger.info(f"Created L1a MAGo CDF file at {filepath}")
        generated_files.append(filepath)

    for _, magi in l1a["magi"].items():
        norm_magi_output = generate_dataset(
            magi,
            MagGlobalCdfAttributes(
                data_mode, Sensor.MAGI, generation_date, input_files, data_version
            ).attribute_dict,
        )
        filepath = write_cdf(norm_magi_output)
        logger.info(f"Created L1a MAGi CDF file at {filepath}")
        generated_files.append(filepath)

    return generated_files


def process_packets(
    mag_l0_list: list[MagL0],
) -> dict[str, dict[np.datetime64, MagL1a]]:
    """
    Given a list of MagL0 packets, process them into MagO and MagI L1A data classes.

    Parameters
    ----------
    mag_l0_list : list[MagL0]
        List of Mag L0 packets to process

    Returns
    -------
    packet_dict: dict[str, dict[np.datetime64, MagL1a]]
        Dictionary containing two keys: "mago" which points to a dictionary of mago
         MagL1A objects, and "magi" which points to a dictionary of magi MagL1A objects.
         Each dictionary has keys of days and values of MagL1A objects, so each day
         corresponds to one MagL1A object.

    """
    magi = {}
    mago = {}

    for mag_l0 in mag_l0_list:
        if mag_l0.COMPRESSION:
            raise NotImplementedError("Unable to process compressed data")

        primary_start_time = TimeTuple(mag_l0.PRI_COARSETM, mag_l0.PRI_FNTM)
        secondary_start_time = TimeTuple(mag_l0.SEC_COARSETM, mag_l0.SEC_FNTM)

        mago_is_primary = mag_l0.PRI_SENS == 0

        primary_day = calc_start_time(primary_start_time.to_seconds()).astype(
            "datetime64[D]"
        )
        secondary_day = calc_start_time(secondary_start_time.to_seconds()).astype(
            "datetime64[D]"
        )

        primary_packet_data = MagL1aPacketProperties(
            mag_l0.SHCOARSE,
            primary_start_time,
            mag_l0.PRI_VECSEC,
            mag_l0.PUS_SSUBTYPE,
            mag_l0.ccsds_header.SRC_SEQ_CTR,
            mag_l0.COMPRESSION,
            mag_l0.MAGO_ACT,
            mag_l0.MAGI_ACT,
            mago_is_primary,
        )

        secondary_packet_data = dataclasses.replace(
            primary_packet_data,
            start_time=secondary_start_time,
            vecsec=mag_l0.SEC_VECSEC,
            pus_ssubtype=mag_l0.PUS_SSUBTYPE,
        )
        # now we know the number of secs of data in the packet, and the data rates of
        # each sensor, we can calculate how much data is in this packet and where the
        # byte boundaries are.

        primary_vectors, secondary_vectors = MagL1a.process_vector_data(
            mag_l0.VECTORS,
            primary_packet_data.total_vectors,
            secondary_packet_data.total_vectors,
        )

        primary_timestamped_vectors = MagL1a.calculate_vector_time(
            primary_vectors,
            primary_packet_data.vectors_per_second,
            primary_packet_data.start_time,
        )
        secondary_timestamped_vectors = MagL1a.calculate_vector_time(
            secondary_vectors,
            secondary_packet_data.vectors_per_second,
            secondary_packet_data.start_time,
        )

        # Sort primary and secondary into MAGo and MAGi by 24 hour chunks
        mago_day = primary_day if mago_is_primary else secondary_day
        magi_day = primary_day if not mago_is_primary else secondary_day

        if mago_day not in mago:
            mago[mago_day] = MagL1a(
                True,
                mag_l0.MAGO_ACT,
                mag_l0.SHCOARSE,
                primary_timestamped_vectors
                if mago_is_primary
                else secondary_timestamped_vectors,
                primary_packet_data if mago_is_primary else secondary_packet_data,
            )
        else:
            mago[mago_day].append_vectors(
                (
                    primary_timestamped_vectors
                    if mago_is_primary
                    else secondary_timestamped_vectors
                ),
                primary_packet_data if mago_is_primary else secondary_packet_data,
            )

        if magi_day not in magi:
            magi[magi_day] = MagL1a(
                False,
                mag_l0.MAGI_ACT,
                mag_l0.SHCOARSE,
                primary_timestamped_vectors
                if not mago_is_primary
                else secondary_timestamped_vectors,
                primary_packet_data if not mago_is_primary else secondary_packet_data,
            )
        else:
            magi[magi_day].append_vectors(
                (
                    primary_timestamped_vectors
                    if not mago_is_primary
                    else secondary_timestamped_vectors
                ),
                primary_packet_data if not mago_is_primary else secondary_packet_data,
            )

    return {"mago": mago, "magi": magi}


def generate_dataset(single_file_l1a: MagL1a, dataset_attrs: dict) -> xr.Dataset:
    """
    Generate a Xarray dataset for L1A data to output to CDF files.

    Global_attrs should contain all info about mago/magi and burst/norm distinction, as
     well as any general info in the global attributes.

     Assumes each MagL1a object is a single day of data, so one MagL1a object has one
     CDF file output.

    Parameters
    ----------
    single_file_l1a: MagL1a
        L1A data covering one day to process into a xarray dataset.
    dataset_attrs: dict
        Global attributes for the dataset, as created by mag_attrs

    Returns
    -------
    dataset : xr.Dataset
        One xarray dataset with proper CDF attributes and shape containing MAG L1A data.
    """
    # TODO: add:
    # gaps_in_data global attr
    # magl1avectordefinition data
    #

    # TODO: Just leave time in datetime64 type with vector as dtype object to avoid this
    # Get the timestamp from the end of the vector
    time_data = single_file_l1a.vectors[:, 4].astype(
        np.dtype("datetime64[ns]"), copy=False
    )

    direction = xr.DataArray(
        np.arange(4),
        name="direction",
        dims=["direction"],
        attrs=mag_cdf_attrs.direction_attrs.output(),
    )

    # TODO: Epoch here refers to the start of the sample. Confirm that this is
    # what mag is expecting, and if it is, CATDESC needs to be updated.
    epoch_time = xr.DataArray(
        time_data,
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    vectors = xr.DataArray(
        single_file_l1a.vectors[:, :4],
        name="vectors",
        dims=["epoch", "direction"],
        attrs=mag_cdf_attrs.vector_attrs.output(),
    )

    output = xr.Dataset(
        coords={"epoch": epoch_time, "direction": direction},
        attrs=dataset_attrs,
    )

    output["vectors"] = vectors

    # TODO: Put is_mago and active in the header

    return output
