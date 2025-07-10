"""Methods for decomming packets, processing to level 1A, and writing CDFs for MAG."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag.constants import DataMode, PrimarySensor
from imap_processing.mag.l0 import decom_mag
from imap_processing.mag.l0.mag_l0_data import MagL0
from imap_processing.mag.l1a.mag_l1a_data import (
    MagL1a,
    MagL1aPacketProperties,
    TimeTuple,
)

logger = logging.getLogger(__name__)


def mag_l1a(packet_filepath: Path) -> list[xr.Dataset]:
    """
    Will process MAG L0 data into L1A CDF files at cdf_filepath.

    For each packet file passed in, this will output between 2 and 6 CDF files.

    Nominally, we expect raw, burst, and norm mode data for the mago and magi sensors.
    It is possible to only have norm or burst mode, or to only have data from one
    sensor.

    Parameters
    ----------
    packet_filepath : pathlib.Path
        Packet files for processing.

    Returns
    -------
    generated_files : list[xarray.Dataset]
        A list of generated filenames.
    """
    packets = decom_mag.decom_packets(packet_filepath)

    norm_data = packets["norm"]
    burst_data = packets["burst"]

    # Create attribute manager and add MAG L1A attributes and global variables
    attribute_manager = ImapCdfAttributes()
    attribute_manager.add_instrument_global_attrs("mag")
    attribute_manager.add_instrument_variable_attrs("mag", "l1a")
    attribute_manager.add_global_attribute(
        "Generation_date",
        np.datetime64(
            "now",
        ).astype(str),
    )

    generated_datasets = create_l1a(norm_data, DataMode.NORM, attribute_manager)
    generated_datasets += create_l1a(burst_data, DataMode.BURST, attribute_manager)

    return generated_datasets


def create_l1a(
    packet_data: list[MagL0], data_mode: DataMode, attribute_manager: ImapCdfAttributes
) -> list[xr.Dataset]:
    """
    Will process MAG L0 data into L1A, then create and write out CDF files.

    Norm and burst mode descriptors are distinguished with the passed in attrs.

    Parameters
    ----------
    packet_data : list[MagL0]
        List of MagL0 packets to process, containing primary and secondary sensor data.

    data_mode : DataMode
        Enum for distinguishing between norm and burst mode data.

    attribute_manager : ImapCdfAttributes
        Attribute manager for CDF files for MAG L1A.

    Returns
    -------
    generated_files : list[xarray.Dataset]
        A list of generated filenames.
    """
    if not packet_data:
        return []

    mag_raw = decom_mag.generate_dataset(packet_data, data_mode, attribute_manager)

    generated_datasets = [mag_raw]

    l1a = process_packets(packet_data)

    # given the processed packets in MagL1A classes, we can generate the MAGo and MAGi
    # datasets. A given set of packets may have only mago or only magi.
    logical_file_id = f"imap_mag_l1a_{data_mode.value.lower()}-mago"
    if l1a["mago"] is not None:
        norm_mago_output = generate_dataset(
            l1a["mago"], logical_file_id, attribute_manager
        )
        generated_datasets.append(norm_mago_output)

    logical_file_id = f"imap_mag_l1a_{data_mode.value.lower()}-magi"
    if l1a["magi"] is not None:
        norm_magi_output = generate_dataset(
            l1a["magi"],
            logical_file_id,
            attribute_manager,
        )
        generated_datasets.append(norm_magi_output)

    return generated_datasets


def process_packets(
    mag_l0_list: list[MagL0],
) -> dict[str, MagL1a | None]:
    """
    Given a list of MagL0 packets, process them into MagO and MagI L1A data classes.

    This splits the MagL0 packets into MagO and MagI data, returning a dictionary with
    keys "mago" and "magi."

    Parameters
    ----------
    mag_l0_list : list[MagL0]
        List of Mag L0 packets to process.

    Returns
    -------
    packet_dict : dict[str, MagL1a | None]
        Dictionary containing two keys: "mago" which points to a dictionary of mago
         MagL1A objects, and "magi" which points to a dictionary of magi MagL1A objects.
         Each dictionary has keys of days and values of MagL1A objects, so each day
         corresponds to one MagL1A object.
    """
    magi = None
    mago = None

    for mag_l0 in mag_l0_list:
        primary_start_time = TimeTuple(mag_l0.PRI_COARSETM, mag_l0.PRI_FNTM)
        secondary_start_time = TimeTuple(mag_l0.SEC_COARSETM, mag_l0.SEC_FNTM)

        mago_is_primary = mag_l0.PRI_SENS == PrimarySensor.MAGO.value

        primary_packet_properties = MagL1aPacketProperties(
            mag_l0.SHCOARSE,
            primary_start_time,
            mag_l0.PRI_VECSEC,
            mag_l0.PUS_SSUBTYPE,
            mag_l0.ccsds_header.SRC_SEQ_CTR,
            mag_l0.COMPRESSION,
            mago_is_primary,
            int(mag_l0.VECTORS[0]),
        )

        secondary_packet_data = dataclasses.replace(
            primary_packet_properties,
            start_time=secondary_start_time,
            vectors_per_second=mag_l0.SEC_VECSEC,
            pus_ssubtype=mag_l0.PUS_SSUBTYPE,
            first_byte=int(mag_l0.VECTORS[0]),
        )
        # now we know the number of secs of data in the packet, and the data rates of
        # each sensor, we can calculate how much data is in this packet and where the
        # byte boundaries are.
        primary_vectors, secondary_vectors = MagL1a.process_vector_data(
            mag_l0.VECTORS,
            primary_packet_properties.total_vectors,
            secondary_packet_data.total_vectors,
            mag_l0.COMPRESSION,
        )

        primary_timestamped_vectors = MagL1a.calculate_vector_time(
            primary_vectors,
            primary_packet_properties.vectors_per_second,
            primary_packet_properties.start_time,
        )
        secondary_timestamped_vectors = MagL1a.calculate_vector_time(
            secondary_vectors,
            secondary_packet_data.vectors_per_second,
            secondary_packet_data.start_time,
        )

        # Sort primary and secondary into MAGo and MAGi by 24 hour chunks

        if mago is None:
            mago = MagL1a(
                True,
                mag_l0.MAGO_ACT,
                mag_l0.SHCOARSE,
                primary_timestamped_vectors
                if mago_is_primary
                else secondary_timestamped_vectors,
                primary_packet_properties if mago_is_primary else secondary_packet_data,
            )
        else:
            mago.append_vectors(
                (
                    primary_timestamped_vectors
                    if mago_is_primary
                    else secondary_timestamped_vectors
                ),
                primary_packet_properties if mago_is_primary else secondary_packet_data,
            )

        if magi is None:
            magi = MagL1a(
                False,
                mag_l0.MAGI_ACT,
                mag_l0.SHCOARSE,
                primary_timestamped_vectors
                if not mago_is_primary
                else secondary_timestamped_vectors,
                primary_packet_properties
                if not mago_is_primary
                else secondary_packet_data,
            )
        else:
            magi.append_vectors(
                (
                    primary_timestamped_vectors
                    if not mago_is_primary
                    else secondary_timestamped_vectors
                ),
                primary_packet_properties
                if not mago_is_primary
                else secondary_packet_data,
            )

    return {"mago": mago, "magi": magi}


def generate_dataset(
    single_file_l1a: MagL1a,
    logical_file_id: str,
    attribute_manager: ImapCdfAttributes,
) -> xr.Dataset:
    """
    Generate a Xarray dataset for L1A data to output to CDF files.

    Global_attrs should contain all info about mago/magi and burst/norm distinction, as
     well as any general info in the global attributes.

     Assumes each MagL1a object is a single day of data, so one MagL1a object has one
     CDF file output.

    Parameters
    ----------
    single_file_l1a : MagL1a
        L1A data covering one day to process into a xarray dataset.
    logical_file_id : str
        Indicates which sensor (MagO or MAGi) and mode (burst or norm) the data is from.
        This is used to retrieve the global attributes from attribute_manager.
    attribute_manager : ImapCdfAttributes
        Attributes for the dataset, as created by ImapCdfAttributes.

    Returns
    -------
    dataset : xarray.Dataset
        One xarray dataset with proper CDF attributes and shape containing MAG L1A data.
    """
    # Get the timestamp from the end of the vector
    time_data = single_file_l1a.vectors[:, 4]

    compression = xr.DataArray(
        np.arange(2),
        name="compression",
        dims=["compression"],
        attrs=attribute_manager.get_variable_attributes(
            "compression_attrs", check_schema=False
        ),
    )

    direction = xr.DataArray(
        np.arange(4),
        name="direction",
        dims=["direction"],
        attrs=attribute_manager.get_variable_attributes(
            "direction_attrs", check_schema=False
        ),
    )

    epoch_time = xr.DataArray(
        time_data,
        name="epoch",
        dims=["epoch"],
        attrs=attribute_manager.get_variable_attributes("epoch", check_schema=False),
    )

    vectors = xr.DataArray(
        single_file_l1a.vectors[:, :4],
        name="vectors",
        dims=["epoch", "direction"],
        attrs=attribute_manager.get_variable_attributes("vector_attrs"),
    )

    compression_flags = xr.DataArray(
        single_file_l1a.compression_flags,
        name="compression_flags",
        dims=["epoch", "compression"],
        attrs=attribute_manager.get_variable_attributes("compression_flags_attrs"),
    )

    direction_label = xr.DataArray(
        direction.values.astype(str),
        name="direction_label",
        dims=["direction_label"],
        attrs=attribute_manager.get_variable_attributes(
            "direction_label", check_schema=False
        ),
    )

    compression_label = xr.DataArray(
        compression.values.astype(str),
        name="compression_label",
        dims=["compression_label"],
        attrs=attribute_manager.get_variable_attributes(
            "compression_label", check_schema=False
        ),
    )
    global_attributes = attribute_manager.get_global_attributes(logical_file_id)
    global_attributes["is_mago"] = str(bool(single_file_l1a.is_mago))
    global_attributes["is_active"] = str(bool(single_file_l1a.is_active))
    global_attributes["vectors_per_second"] = (
        single_file_l1a.vectors_per_second_attribute()
    )
    # empty arrays are removed in cdflib
    global_attributes["missing_sequences"] = (
        single_file_l1a.missing_sequences
        if single_file_l1a.missing_sequences
        else "None"
    )

    output = xr.Dataset(
        coords={
            "epoch": epoch_time,
            "direction": direction,
            "compression": compression,
        },
        attrs=global_attributes,
    )

    output["direction_label"] = direction_label
    output["compression_label"] = compression_label
    output["vectors"] = vectors
    output["compression_flags"] = compression_flags

    return output
