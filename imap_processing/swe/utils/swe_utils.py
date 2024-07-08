"""Various utility classes and functions to support SWE processing."""

from enum import IntEnum

import space_packet_parser


class SWEAPID(IntEnum):
    """Create ENUM for apid."""

    SWE_SCIENCE = 1344


def add_metadata_to_array(
    data_packet: space_packet_parser.parser.Packet, metadata_arrays: dict
) -> dict:
    """
    Add metadata to the metadata_arrays.

    Parameters
    ----------
    data_packet : space_packet_parser.parser.Packet
        SWE data packet.
    metadata_arrays : dict
        Metadata arrays.

    Returns
    -------
    metadata_arrays : dict
        The metadata_array with metadata from input data packet.
    """
    for key, value in data_packet.data.items():
        if key == "SCIENCE_DATA":
            continue
        metadata_arrays.setdefault(key, []).append(value.raw_value)

    return metadata_arrays
