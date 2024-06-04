"""Various classes and functions used throughout CoDICE processing.

This module contains utility classes and functions that are used by various
other CoDICE processing modules.
"""

import collections
from enum import IntEnum

import numpy as np
import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import calc_start_time


class CODICEAPID(IntEnum):
    """Create ENUM for CoDICE APIDs.

    Parameters
    ----------
    IntEnum : IntEnum
    """

    COD_AUT = 1120
    COD_BOOT_HK = 1121
    COD_BOOT_MEMDMP = 1122
    COD_COUNTS_COMMON = 1135
    COD_NHK = 1136
    COD_EVTMSG = 1137
    COD_MEMDMP = 1138
    COD_SHK = 1139
    COD_RTS = 1141
    COD_DIAG_CDHFPGA = 1144
    COD_DIAG_SNSR_HV = 1145
    COD_DIAG_OPTC_HV = 1146
    COD_DIAG_APDFPGA = 1147
    COD_DIAG_SSDFPGA = 1148
    COD_DIAG_FSW = 1149
    COD_DIAG_SYSVARS = 1150
    COD_LO_IAL = 1152
    COD_LO_PHA = 1153
    COD_LO_INSTRUMENT_COUNTERS = 1154
    COD_LO_SW_PRIORITY_COUNTS = 1155
    COD_LO_SW_SPECIES_COUNTS = 1156
    COD_LO_NSW_SPECIES_COUNTS = 1157
    COD_LO_SW_ANGULAR_COUNTS = 1158
    COD_LO_NSW_ANGULAR_COUNTS = 1159
    COD_LO_NSW_PRIORITY_COUNTS = 1160
    COD_HI_IAL = 1168
    COD_HI_PHA = 1169
    COD_HI_INSTRUMENT_COUNTERS = 1170
    COD_HI_OMNI_SPECIES_COUNTS = 1172
    COD_HI_SECT_SPECIES_COUNTS = 1173
    COD_CSTOL_CONFIG = 2457


class CoDICECompression(IntEnum):
    """Create ENUM for CoDICE compression algorithms.

    Parameters
    ----------
    IntEnum : IntEnum
    """

    NO_COMPRESSION = 0
    LOSSY_A = 1
    LOSSY_B = 2
    LOSSLESS = 3
    LOSSY_A_LOSSLESS = 4
    LOSSY_B_LOSSLESS = 5


def add_metadata_to_array(packet, metadata_arrays: dict) -> dict:
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
    ignore_list = [
        "SPARE_1",
        "SPARE_2",
        "SPARE_3",
        "SPARE_4",
        "SPARE_5",
        "SPARE_6",
        "CHECKSUM",
    ]

    for key, value in packet.header.items():
        metadata_arrays.setdefault(key, []).append(value.raw_value)

    for key, value in packet.data.items():
        if key not in ignore_list:
            metadata_arrays.setdefault(key, []).append(value.raw_value)

    return metadata_arrays


def create_hskp_dataset(packets) -> xr.Dataset:
    """Create dataset for each metadata field for housekeeping data.

    Parameters
    ----------
    packets : list[space_packet_parser.parser.Packet]
        The list of packets to process

    Returns
    -------
    xarray.Dataset
        xarray dataset containing the metadata
    """
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1a")

    metadata_arrays = collections.defaultdict(list)

    for packet in packets:
        add_metadata_to_array(packet, metadata_arrays)

    # TODO: Is there a way to get the attrs from the YAML-based method?
    epoch = xr.DataArray(
        [
            calc_start_time(
                item, launch_time=np.datetime64("2010-01-01T00:01:06.184", "ns")
            )
            for item in metadata_arrays["SHCOARSE"]
        ],
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    dataset = xr.Dataset(
        coords={"epoch": epoch},
        attrs={
            **cdf_attrs.get_global_attributes(),
            **cdf_attrs.global_attributes["imap_codice_l1a_hskp"],
        },
    )

    # TODO: Change 'TBD' catdesc and fieldname
    # Once packet definition files are re-generated, can get this info from
    # something like this:
    #    for key, value in (packet.header | packet.data).items():
    #      fieldname = value.short_description
    #      catdesc = value.short_description
    # I am holding off making this change until I acquire updated housekeeping
    # packets/validation data that match the latest telemetry definitions
    # I may also be able to replace this function with utils.create_dataset(?)
    for key, value in metadata_arrays.items():
        attrs = cdf_attrs.variable_attributes["codice_support_attrs"]
        attrs["CATDESC"] = "TBD"
        attrs["DEPEND_0"] = "epoch"
        attrs["FIELDNAM"] = "TBD"
        attrs["LABLAXIS"] = key

        dataset[key] = xr.DataArray(value, dims=["epoch"], attrs=attrs)

    return dataset
