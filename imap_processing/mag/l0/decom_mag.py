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
    """
    # TODO split by mago and magi using primary sensor
    # TODO split by norm and burst
    norm_data = defaultdict(list)
    burst_data = norm_data.copy()

    for datapoint in l0_data:
        if datapoint.ccsds_header.PKT_APID == Mode.NORMAL:
            for key, value in dataclasses.asdict(datapoint).items():
                if key != "ccsds_header":
                    norm_data[key].append(value)
        if datapoint.ccsds_header.PKT_APID == Mode.BURST:
            burst_data["SHCOARSE"].append(datapoint.SHCOARSE)
            burst_data["raw_vectors"].append(datapoint.VECTORS)

    # Used in L1A vectors
    direction_norm = xr.DataArray(
        np.arange(len(norm_data["VECTORS"][0])),
        name="Direction",
        dims=["Direction"],
        attrs=mag_cdf_attrs.direction_attrs.output(),
    )

    norm_epoch_time = xr.DataArray(
        [calc_start_time(shcoarse) for shcoarse in norm_data["SHCOARSE"]],
        name="Epoch",
        dims=["Epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    # TODO: raw vectors units
    norm_raw_vectors = xr.DataArray(
        norm_data["VECTORS"],
        name="Raw_Vectors",
        dims=["Epoch", "Direction"],
        attrs=mag_cdf_attrs.mag_vector_attrs.output(),
    )

    # TODO add norm to attrs somehow
    norm_dataset = xr.Dataset(
        coords={"Epoch": norm_epoch_time, "Direction": direction_norm},
        attrs=mag_cdf_attrs.mag_l1a_attrs.output(),
    )

    norm_dataset["RAW_VECTORS"] = norm_raw_vectors

    # TODO: retrieve the doc for the CDF description (etattr(MagL0, "__doc__", {}))

    for key, value in norm_data.items():
        # Time varying values
        if key not in ["SHCOARSE", "VECTORS"]:
            norm_datarray = xr.DataArray(
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
            norm_dataset[key] = norm_datarray

    return norm_dataset
