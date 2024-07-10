"""Decommutate HIT CCSDS data and create L1a data products."""

import logging
from collections import defaultdict
from dataclasses import fields
from enum import IntEnum

import numpy as np
import xarray as xr

from imap_processing import decom, imap_module_directory, utils
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import met_to_j2000ns
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


def hit_l1a(packet_file: str, data_version: str) -> list[xr.Dataset]:
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
    cdf_filepaths : list[xarray.Dataset]
        List of Datasets of L1A processed data.
    """
    # Decom, sort, and group packets by apid
    packets = decom_packets(packet_file)
    sorted_packets = utils.sort_by_time(packets, "SHCOARSE")
    grouped_data = group_data(sorted_packets)

    # create the attribute manager for this data level
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="hit")
    attr_mgr.add_instrument_variable_attrs(instrument="hit", level="l1a")
    attr_mgr.add_global_attribute("Data_version", data_version)

    # Create datasets
    datasets = create_datasets(grouped_data, attr_mgr)

    return list(datasets.values())


def decom_packets(packet_file: str) -> list:
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
    unpacked_packets: list = decom.decom_packets(packet_file, xtce_file)
    logger.debug(f"{packet_file} unpacked")
    return unpacked_packets


def group_data(unpacked_data: list) -> dict:
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
    grouped_data: dict = utils.group_by_apid(unpacked_data)

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


def create_datasets(data: dict, attr_mgr: ImapCdfAttributes) -> dict:
    """
    Create a dataset for each APID in the data.

    Parameters
    ----------
    data : dict
        A single dictionary containing data for all instances of an APID.
    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the data product field's attributes.

    Returns
    -------
    processed_data : dict
        A dictionary containing xarray.Dataset for each APID. Each dataset in the
        dictionary will be converted to a CDF.
    """
    logger.info("Creating datasets for HIT L1A data")

    skip_keys = [
        "shcoarse",
        "ground_sw_version",
        "packet_file_name",
        "ccsds_header",
        "leak_i_raw",
    ]

    processed_data = {}
    for apid, data_packets in data.items():
        if apid == HitAPID.HIT_HSKP:
            logical_source = "imap_hit_l1a_hk"
            # TODO define keys to skip for each apid. Currently just have
            #  a list for housekeeping. Some of these may change later.
            #  leak_i_raw can be handled in the housekeeping class as an
            #  InitVar so that it doesn't show up when you extract the object's
            #  field names.
        elif apid == HitAPID.HIT_SCIENCE:
            logical_source = "imap_hit_l1a_sci-counts"
            # TODO what about pulse height? It has the same apid.
            #  Will need to approach this differently
        else:
            raise Exception(f"Unknown APID [{apid}]")
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
        epoch_converted_times = met_to_j2000ns(metadata_arrays["shcoarse"])

        # Create xarray data arrays for dependencies
        epoch_time = xr.DataArray(
            epoch_converted_times,
            name="epoch",
            dims=["epoch"],
            attrs=attr_mgr.get_variable_attributes("epoch"),
        )

        adc_channels = xr.DataArray(
            np.arange(64, dtype=np.uint16),
            name="adc_channels",
            dims=["adc_channels"],
            attrs=attr_mgr.get_variable_attributes("adc_channels"),
        )

        # NOTE: LABL_PTR_1 should be CDF_CHAR.
        adc_channels_label = xr.DataArray(
            adc_channels.values.astype(str),
            name="adc_channels_label",
            dims=["adc_channels_label"],
            attrs=attr_mgr.get_variable_attributes("adc_channels_label"),
        )

        # Create xarray dataset
        dataset = xr.Dataset(
            coords={
                "epoch": epoch_time,
                "adc_channels": adc_channels,
                "adc_channels_label": adc_channels_label,
            },
            attrs=attr_mgr.get_global_attributes(logical_source),
        )

        # Create xarray data array for each metadata field
        for field, data in metadata_arrays.items():  # type: ignore[assignment]
            # TODO Error, Incompatible types in assignment
            # (expression has type "str", variable has type "Field[Any]")
            # AND
            # Incompatible types in assignment
            # (expression has type "list[Any]", variable has type "dict[Any, Any]")
            if field not in skip_keys:  # type: ignore[comparison-overlap]
                # TODO Error, Non-overlapping container check
                # (element type: "Field[Any]", container item type: "str")

                # Create a list of all the dimensions using the DEPEND_I keys in the
                # attributes
                dims = [
                    value
                    for key, value in attr_mgr.get_variable_attributes(field).items()  # type: ignore[arg-type]
                    if "DEPEND" in key
                ]
                if field == "leak_i":  # type: ignore[comparison-overlap]
                    # TODO Error,  Non-overlapping equality check
                    # (left operand type: "Field[Any]",
                    # right operand type: "Literal['leak_i']")

                    # 2D array - needs two dims
                    dataset[field] = xr.DataArray(
                        data,
                        dims=dims,
                        attrs=attr_mgr.get_variable_attributes(field),  # type: ignore[arg-type]
                    )
                else:
                    dataset[field] = xr.DataArray(
                        data,
                        dims=dims,
                        attrs=attr_mgr.get_variable_attributes(field),  # type: ignore[arg-type]
                    )
        processed_data[apid] = dataset
    logger.info("HIT L1A datasets created")
    return processed_data
