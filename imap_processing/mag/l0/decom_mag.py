"""Methods for processing raw MAG packets into CDF files for level 0 and level 1a."""
from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr
from space_packet_parser import parser, xtcedef

from imap_processing import imap_module_directory
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import calc_start_time
from imap_processing.mag import mag_cdf_attrs
from imap_processing.mag.l0.mag_l0_data import MagL0, Mode

logger = logging.getLogger(__name__)


def decom_packets(packet_file_path: str | Path) -> list[MagL0]:
    """Decom MAG data packets using MAG packet definition.

    Parameters
    ----------
    packet_file_path : str
        Path to data packet path with filename.

    Returns
    -------
    data : list[MagL0]
        A list of MAG L0 data classes, including both burst and normal packets. (the
        packet type is defined in each instance of L0.)
    """
    # Define paths
    xtce_document = Path(
        f"{imap_module_directory}/mag/packet_definitions/MAG_SCI_COMBINED.xml"
    )

    packet_definition = xtcedef.XtcePacketDefinition(xtce_document)
    mag_parser = parser.PacketParser(packet_definition)

    data_list = []

    with open(packet_file_path, "rb") as binary_data:
        mag_packets = mag_parser.generator(binary_data)

        for packet in mag_packets:
            apid = packet.header["PKT_APID"].derived_value
            if apid in (Mode.BURST, Mode.NORMAL):
                values = [
                    item.derived_value
                    if item.derived_value is not None
                    else item.raw_value
                    for item in packet.data.values()
                ]
                data_list.append(MagL0(CcsdsData(packet.header), *values))

        return data_list


def export_to_xarray(l0_data: list[MagL0]):
    """Generate xarray files for "raw" MAG CDF files from MagL0 data.

    Mag outputs "RAW" CDF files just after decomming. These have the immediately
    post-decom data, with raw binary data for the vectors instead of vector values.

    Parameters
    ----------
    l0_data: list[MagL0]
        A list of MagL0 datapoints

    Returns
    -------
    norm_data : xr.Dataset
        xarray dataset for generating burst data CDFs
    burst_data : xr.Dataset
        xarray dataset for generating burst data CDFs
    """
    # TODO split by mago and magi using primary sensor
    norm_data = []
    burst_data = []

    for packet in l0_data:
        if packet.ccsds_header.PKT_APID == Mode.NORMAL:
            norm_data.append(packet)
        if packet.ccsds_header.PKT_APID == Mode.BURST:
            burst_data.append(packet)

    norm_dataset = None
    burst_dataset = None

    if len(norm_data) > 0:
        norm_dataset = generate_dataset(norm_data)
    if len(burst_data) > 0:
        burst_dataset = generate_dataset(burst_data)

    return norm_dataset, burst_dataset


def generate_dataset(l0_data: list[MagL0]):
    """
    Generate a CDF dataset from the sorted L0 MAG data.

    Used to create 2 similar datasets, for norm and burst data.

    Parameters
    ----------
    l0_data : list[MagL0]
        List of sorted L0 MAG data.

    Returns
    -------
    dataset : xr.Dataset
        xarray dataset with proper CDF attributes and shape.
    """
    vector_data = np.zeros((len(l0_data), len(l0_data[0].VECTORS)))
    shcoarse_data = np.zeros(len(l0_data))

    support_data = defaultdict(list)

    for index, datapoint in enumerate(l0_data):
        vector_len = len(datapoint.VECTORS)
        if vector_len > vector_data.shape[1]:
            # If the new vector is longer than the existing shape, first reshape
            # vector_data and pad the existing vectors with zeros.
            vector_data = np.pad(
                vector_data,
                (
                    (
                        0,
                        0,
                    ),
                    (0, vector_len - vector_data.shape[1]),
                ),
                "constant",
                constant_values=(0,),
            )
        vector_data[index, :vector_len] = datapoint.VECTORS

        shcoarse_data[index] = calc_start_time(datapoint.SHCOARSE)

        # Add remaining pieces to arrays
        for key, value in dataclasses.asdict(datapoint).items():
            if key not in ("ccsds_header", "VECTORS", "SHCOARSE"):
                support_data[key].append(value)

    # Used in L1A vectors
    direction = xr.DataArray(
        np.arange(vector_data.shape[1]),
        name="Direction",
        dims=["Direction"],
        attrs=mag_cdf_attrs.direction_attrs.output(),
    )

    epoch_time = xr.DataArray(
        shcoarse_data,
        name="Epoch",
        dims=["Epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    # TODO: raw vectors units
    raw_vectors = xr.DataArray(
        vector_data,
        name="Raw_Vectors",
        dims=["Epoch", "Direction"],
        attrs=mag_cdf_attrs.mag_vector_attrs.output(),
    )

    # TODO add norm to attrs somehow
    output = xr.Dataset(
        coords={"Epoch": epoch_time, "Direction": direction},
        attrs=mag_cdf_attrs.mag_l1a_attrs.output(),
    )

    output["RAW_VECTORS"] = raw_vectors

    for key, value in support_data.items():
        # Time varying values
        if key not in ["SHCOARSE", "VECTORS"]:
            output[key] = xr.DataArray(
                value,
                name=key,
                dims=["Epoch"],
                attrs=dataclasses.replace(
                    mag_cdf_attrs.mag_support_attrs,
                    catdesc=f"Raw {key} values varying by time",
                    fieldname=f"{key}",
                    # TODO: label_axis should be as close to 6 letters as possible
                    label_axis=key,
                    display_type="no_plot",
                ).output(),
            )

    return output
