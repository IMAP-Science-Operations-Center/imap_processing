"""
Various classes and functions used throughout CoDICE processing.

This module contains utility classes and functions that are used by various
other CoDICE processing modules.
"""

import collections
from enum import IntEnum

import numpy as np
import space_packet_parser
import xarray as xr

from imap_processing.cdf import epoch_attrs
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import met_to_j2000ns


class CODICEAPID(IntEnum):
    """Create ENUM for CoDICE APIDs."""

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
    COD_LO_SW_PRIORITY_COUNTS = 1155
    COD_LO_SW_SPECIES_COUNTS = 1156
    COD_LO_NSW_SPECIES_COUNTS = 1157
    COD_LO_SW_ANGULAR_COUNTS = 1158
    COD_LO_NSW_ANGULAR_COUNTS = 1159
    COD_LO_NSW_PRIORITY_COUNTS = 1160
    COD_LO_INST_COUNTS_AGGREGATED = 1161
    COD_LO_INST_COUNTS_SINGLES = 1162
    COD_HI_IAL = 1168
    COD_HI_PHA = 1169
    COD_HI_INST_COUNTS_AGGREGATED = 1170
    COD_HI_INST_COUNTS_SINGLES = 1171
    COD_HI_OMNI_SPECIES_COUNTS = 1172
    COD_HI_SECT_SPECIES_COUNTS = 1173
    COD_CSTOL_CONFIG = 2457


class CoDICECompression(IntEnum):
    """Create ENUM for CoDICE compression algorithms."""

    NO_COMPRESSION = 0
    LOSSY_A = 1
    LOSSY_B = 2
    LOSSLESS = 3
    LOSSY_A_LOSSLESS = 4
    LOSSY_B_LOSSLESS = 5


def add_metadata_to_array(packet: space_packet_parser, metadata_arrays: dict) -> dict:
    """
    Add metadata to the metadata_arrays.

    Parameters
    ----------
    packet : space_packet_parser.parser.Packet
        CODICE data packet.
    metadata_arrays : dict
        Metadata arrays.

    Returns
    -------
    metadata_arrays : dict
        Updated metadata arrays with values.
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


# TODO: Correct the type of "packets"
def create_hskp_dataset(  # type: ignore[no-untyped-def]
    packets,
    data_version: str,
) -> xr.Dataset:
    """
    Create dataset for each metadata field for housekeeping data.

    Parameters
    ----------
    packets : list[space_packet_parser.parser.Packet]
        The list of packets to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    dataset : xarray.Dataset
        Xarray dataset containing the metadata.
    """
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1a")
    cdf_attrs.add_global_attribute("Data_version", data_version)

    metadata_arrays: dict = collections.defaultdict(list)

    for packet in packets:
        add_metadata_to_array(packet, metadata_arrays)

    # TODO: Is there a way to get the attrs from the YAML-based method?
    epoch = xr.DataArray(
        met_to_j2000ns(
            metadata_arrays["SHCOARSE"],
            reference_epoch=np.datetime64("2010-01-01T00:01:06.184", "ns"),
        ),
        name="epoch",
        dims=["epoch"],
        attrs=epoch_attrs,
    )

    dataset = xr.Dataset(
        coords={"epoch": epoch},
        attrs=cdf_attrs.get_global_attributes("imap_codice_l1a_hskp"),
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
        attrs = cdf_attrs.get_variable_attributes("codice_support_attrs")
        attrs["CATDESC"] = "TBD"
        attrs["DEPEND_0"] = "epoch"
        attrs["FIELDNAM"] = "TBD"
        attrs["LABLAXIS"] = key

        dataset[key] = xr.DataArray(value, dims=["epoch"], attrs=attrs)

    return dataset
