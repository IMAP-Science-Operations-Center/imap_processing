"""Various classes and functions used throughout CoDICE processing.

This module contains utility classes and functions that are used by various
other CoDICE processing modules.
"""

import collections
import dataclasses
from enum import IntEnum

import space_packet_parser
import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.codice import cdf_attrs


class CODICEAPID(IntEnum):
    """Create ENUM for CoDICE APIDs.

    Parameters
    ----------
    IntEnum : IntEnum
    """

    COD_AUT = 1120
    COD_NHK = 1136
    COD_EVTMSG = 1137
    COD_MEMDMP = 1138
    COD_SHK = 1139
    COD_RTS = 1141
    COD_DIAG_SNSR_HV = 1145
    COD_DIAG_OPTC_HV = 1146
    COD_DIAG_APDFPGA = 1147
    COD_DIAG_SSDFPGA = 1148
    COD_DIAG_FSW = 1149
    COD_DIAG_SYSVARS = 1150
    COD_LO_IAL = 1152
    COD_LO_PHA = 1153
    COD_LO_INSTRUMENT_COUNTERS = 1154
    COD_LO_PRIORITY_COUNTS = 1155
    COD_LO_SW_SPECIES_COUNTS = 1156
    COD_LO_NSW_SPECIES_COUNTS = 1157
    COD_LO_SW_ANGULAR_COUNTS = 1158
    COD_LO_NSW_ANGULAR_COUNTS = 1159
    COD_HI_IAL = 1168
    COD_HI_PHA = 1169
    COD_HI_INSTRUMENTCOUNTERS = 1170
    COD_HI_OMNI_SPECIES_COUNTS = 1172
    COD_HI_SECT_SPECIES_COUNTS = 1173
    COD_CSTOL_CONFIG = 2457


class CoDICECompression(IntEnum):
    """Create ENUM for CoDICE compression algorithms.

    Parameters
    ----------
    IntEnum : IntEnum
    """

    NO_COMPRESSION = 1
    LOSSY_A = 2
    LOSSY_B = 3
    LOSSLESS = 4
    LOSSY_A_LOSSLESS = 5
    LOSSY_B_LOSSLESS = 6


def add_metadata_to_array(
    packet: space_packet_parser.parser.Packet, metadata_arrays: dict
) -> dict:
    """Add metadata to the metadata_arrays.

    Parameters
    ----------
    packet : space_packet_parser.parser.Packet
        CODICE data packet
    metadata_arrays : dict
        Metadata arrays

    Returns
    -------
    metadata_arrays : dict
        Updated metadata arrays with values
    """
    for key, value in packet.header.items():
        metadata_arrays.setdefault(key, []).append(value.raw_value)

    for key, value in packet.data.items():
        metadata_arrays.setdefault(key, []).append(value.raw_value)

    return metadata_arrays


def create_dataset(packets: list[space_packet_parser.parser.Packet]) -> xr.Dataset:
    """Create dataset for each metadata field.

    Parameters
    ----------
    packets : list[space_packet_parser.parser.Packet]
        The list of packets to process

    Returns
    -------
    xarray.Dataset
        xarray dataset containing the metadata
    """
    metadata_arrays = collections.defaultdict(list)

    for packet in packets:
        add_metadata_to_array(packet, metadata_arrays)

    epoch_time = xr.DataArray(
        metadata_arrays["SHCOARSE"],
        name="Epoch",
        dims=["Epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    dataset = xr.Dataset(
        coords={"Epoch": epoch_time},
        attrs=cdf_attrs.codice_l1a_global_attrs.output(),
    )

    for key, value in metadata_arrays.items():
        if key == "SHCOARSE":
            continue
        else:
            dataset[key] = xr.DataArray(
                value,
                dims=["Epoch"],
                attrs=dataclasses.replace(
                    cdf_attrs.codice_metadata_attrs,
                    catdesc=key,
                    fieldname=key,
                    label_axis=key,
                    depend_0="Epoch",
                ).output(),
            )

    return dataset
