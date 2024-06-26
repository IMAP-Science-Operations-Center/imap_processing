"""Decommutate HIT CCSDS data and create L1a data products."""

import logging
import typing
from collections import defaultdict
from dataclasses import fields
from enum import IntEnum
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing import decom, imap_module_directory, utils
from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.hit import hit_cdf_attrs
from imap_processing.hit.l0.data_classes.housekeeping import Housekeeping

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)


class HitAPID(IntEnum):
    """
    HIT APID Mappings.

    Attributes
    ----------
    HIT_HSKP: int
        Housekeeping
    HIT_SCIENCE : int
        Science
    HIT_IALRT : int
        I-ALiRT
    """

    HIT_HSKP = 1251
    HIT_SCIENCE = 1252
    HIT_IALRT = 1253


def hit_l1a(packet_file: typing.Union[Path, str], data_version: str):
    """
    Will process HIT L0 data into L1A data products.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    cdf_filepaths : dict
        List of file paths to CDF data product files.
    """
    # Decom, sort, and group packets by apid
    packets = decom_packets(packet_file)
    sorted_packets = utils.sort_by_time(packets, "SHCOARSE")
    grouped_data = group_data(sorted_packets)

    # Create datasets
    # TODO define keys to skip for each apid. Currently just have
    #  a list for housekeeping. Some of these may change later.
    #  leak_i_raw can be handled in the housekeeping class as an
    #  InitVar so that it doesn't show up when you extract the object's
    #  field names.
    skip_keys = [
        "shcoarse",
        "ground_sw_version",
        "packet_file_name",
        "ccsds_header",
        "leak_i_raw",
    ]
    datasets = create_datasets(grouped_data, skip_keys)

    for dataset in datasets.values():
        # TODO: update to use the add_global_attribute() function
        dataset.attrs["Data_version"] = data_version
    return list(datasets.values())


def decom_packets(packet_file: str):
    """
    Unpack and decode packets using CCSDS file and XTCE packet definitions.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.

    Returns
    -------
    unpacked_packets : list
        List of all the unpacked data.
    """
    # TODO: update path to use a combined packets xtce file
    xtce_file = imap_module_directory / "hit/packet_definitions/P_HIT_HSKP.xml"
    logger.debug(f"Unpacking {packet_file} using xtce definitions in {xtce_file}")
    unpacked_packets = decom.decom_packets(packet_file, xtce_file)
    logger.debug(f"{packet_file} unpacked")
    return unpacked_packets


def group_data(unpacked_data: list):
    """
    Group data by apid.

    Parameters
    ----------
    unpacked_data : list
        Packet list.

    Returns
    -------
    grouped_data : dict
        Grouped data by apid.
    """
    logger.debug("Grouping packet values for each apid")
    grouped_data = utils.group_by_apid(unpacked_data)

    # Create data classes for each packet
    for apid in grouped_data:
        if apid == HitAPID.HIT_HSKP:
            logger.debug(f"Grouping housekeeping packets - APID: {apid}")
            grouped_data[apid] = [
                Housekeeping(packet, "0.0", "hskp_sample.ccsds")
                for packet in grouped_data[apid]
            ]
        else:
            raise RuntimeError(f"Encountered unexpected APID [{apid}]")

    logger.debug("Finished grouping packet data")
    return grouped_data


def create_datasets(data: dict, skip_keys=None):
    """
    Create a dataset for each APID in the data.

    Parameters
    ----------
    data : dict
        A single dictionary containing data for all instances of an APID.
    skip_keys : list, Optional
        Keys to skip in the metadata.

    Returns
    -------
    processed_data : dict
        A dictionary containing xarray.Dataset for each APID. Each dataset in the
        dictionary will be converted to a CDF.
    """
    logger.info("Creating datasets for HIT L1A data")
    processed_data = {}
    for apid, data_packets in data.items():
        metadata_arrays = defaultdict(list)
        for packet in data_packets:
            # Add metadata to an array
            for field in fields(packet):
                field_name = field.name
                field_value = getattr(packet, field_name)
                # convert key to lower case to match SPDF requirement
                data_key = field_name.lower()
                metadata_arrays[data_key].append(field_value)

        # Convert integers into datetime64[s]
        epoch_converted_times = utils.met_to_j2000ns(metadata_arrays["shcoarse"])

        # Create xarray data arrays for dependencies
        epoch_time = xr.DataArray(
            epoch_converted_times,
            name="epoch",
            dims=["epoch"],
            attrs=ConstantCoordinates.EPOCH,
        )

        adc_channels = xr.DataArray(
            np.array(np.arange(64), dtype=np.uint16),
            name="adc_channels",
            dims=["adc_channels"],
            attrs=hit_cdf_attrs.l1a_hk_attrs["adc_channels"].output(),
        )

        # Create xarray dataset
        dataset = xr.Dataset(
            coords={"epoch": epoch_time, "adc_channels": adc_channels},
            attrs=hit_cdf_attrs.hit_hk_l1a_attrs.output(),
        )

        # Create xarray data array for each metadata field
        for key, value in metadata_arrays.items():
            if key not in skip_keys:
                if key == "leak_i":
                    # 2D array - needs two dims
                    dataset[key] = xr.DataArray(
                        value,
                        dims=["epoch", "adc_channels"],
                        attrs=hit_cdf_attrs.l1a_hk_attrs[key].output(),
                    )
                else:
                    dataset[key] = xr.DataArray(
                        value,
                        dims=["epoch"],
                        attrs=hit_cdf_attrs.l1a_hk_attrs[key].output(),
                    )
        processed_data[apid] = dataset
    logger.info("HIT L1A datasets created")
    return processed_data
