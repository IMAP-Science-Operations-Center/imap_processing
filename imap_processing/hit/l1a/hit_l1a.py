"""Decommutate HIT CCSDS data and create L1a data products."""

import logging
from collections import defaultdict
from dataclasses import fields
from enum import IntEnum

import numpy as np
import xarray as xr

from imap_processing import decom, imap_module_directory, utils
from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import write_cdf
from imap_processing.hit import hit_cdf_attrs
from imap_processing.hit.l0.data_classes.housekeeping import Housekeeping

logger = logging.getLogger(__name__)


class HitAPID(IntEnum):
    """
    HIT APID Mappings.

    Attributes
    ----------
    HIT_AUT : int
        Autonomy
    HIT_HSKP: int
        Housekeeping
    HIT_SCIENCE : int
        Science
    HIT_IALRT : int
        I-ALiRT
    HIT_MEMDUMP : int
        Memory dump
    """

    HIT_AUT = 1250  # Autonomy
    HIT_HSKP = 1251  # Housekeeping
    HIT_SCIENCE = 1252  # Science
    HIT_IALRT = 1253  # I-ALiRT
    HIT_MEMDUMP = 1255  # Memory dump


def hit_l1a_data(packet_file: Path | str):
    """
    Process HIT L0 data into L1A data products.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.

    Returns
    -------
    dict
        List of file paths to CDF data product files
    """
    # Decom, sort, and group packets by apid
    packets = decom_packets(packet_file)
    sorted_packets = sorted(packets, key=lambda x: x.data["SHCOARSE"].derived_value)
    grouped_data = group_data(sorted_packets)

    # Create datasets
    # TODO define keys to skip for each apid. Currently just have
    #  a list for housekeeping.
    skip_keys = [
        "shcoarse",
        "ground_sw_version",
        "packet_file_name",
        "ccsds_header",
        "leak_i_raw",
    ]
    datasets = create_datasets(grouped_data, skip_keys)

    # Create CDF files
    logger.info("Creating CDF files for HIT L1A data")
    cdf_filepaths = []
    for dataset in datasets.values():
        cdf_file = write_cdf(dataset)
        cdf_filepaths.append(cdf_file)
    logger.info("L1A CDF files created")
    return cdf_filepaths


def decom_packets(packet_file: str):
    """
    Unpack and decode packets using CCSDS file and XTCE packet definitions.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.

    Returns
    -------
    list
        List of all the unpacked data
    """
    # TODO: update path to use a combined packets xtce file
    xtce_file = imap_module_directory / "hit/packet_definitions/P_HIT_HSKP.xml"
    logger.info(f"Unpacking {packet_file} using xtce definitions in {xtce_file}")
    unpacked_packets = decom.decom_packets(packet_file, xtce_file)
    logger.info(f"{packet_file} unpacked")
    return unpacked_packets


def group_data(unpacked_data: list):
    """Group data by apid.

    Parameters
    ----------
    unpacked_data : list
        packet list

    Returns
    -------
    dict
        grouped data by apid
    """
    logger.info("Grouping packet values for each apid")
    grouped_data = utils.group_by_apid(unpacked_data)

    # Create data classes for each packet
    for apid in grouped_data.keys():
        if apid == HitAPID.HIT_HSKP:
            logger.info(f"Grouping housekeeping packets - APID: {apid}")
            for i, packet in enumerate(grouped_data[apid]):
                # convert data to data classes
                grouped_data[apid][i] = Housekeeping(packet, "0.0", "hskp_sample.ccsds")
        else:
            raise RuntimeError(f"Encountered unexpected APID [{apid}]")

    logger.info("Finished grouping packet data")
    return grouped_data


def create_datasets(data: dict, skip_keys=None):
    """
    Create a dataset for each APID in the data.

    Parameters
    ----------
    data : dict
        A single dictionary containing data for all instances of an APID.
    skip_keys: list, Optional
        Keys to skip in the metadata

    Returns
    -------
    dict
        A dictionary containing xr.Dataset for each APID. Each dataset in the
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
                print(f"{field_name}: {field_value}")
                # convert key to lower case to match SPDF requirement
                data_key = field_name.lower()
                metadata_arrays[data_key].append(field_value)

        # Create xarray data arrays for dependencies
        epoch_time = xr.DataArray(
            np.array(metadata_arrays["shcoarse"], dtype="datetime64[s]"),
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

        # Create a xarray dataset
        dataset = xr.Dataset(
            coords={"epoch": epoch_time, "adc_channels": adc_channels},
            attrs=hit_cdf_attrs.hit_hk_l1a_attrs.output(),
        )

        # Create a xarray data array for each metadata field
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
