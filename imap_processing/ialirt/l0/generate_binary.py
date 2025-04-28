"""I-ALiRT data to populate binary blob database."""

from pathlib import Path
from typing import Union

from space_packet_parser import definitions


def generate_binary(
    packet_file: Union[str, Path],
    xtce_packet_definition: Union[str, Path],
) -> tuple[list, list]:
    """
    Generate binary blob and SCLK data for each packet.

    Parameters
    ----------
    packet_file : str
        Path to data packet path with filename.
    xtce_packet_definition : str
        Path to XTCE file with filename.

    Returns
    -------
    binary_blob_data : list
        Binary blob data for each packet.
    met_data : list
        SCLK time in seconds.
    """
    met_data = []
    binary_blob_data = []

    # Set up the parser from the input packet definition
    packet_definition = definitions.XtcePacketDefinition(xtce_packet_definition)
    with packet_file.open("rb") as binary_data:
        packet_generator = packet_definition.packet_generator(binary_data)

        # Iterate over the packets and access the raw binary data
        for packet in packet_generator:
            binary_blob = packet.raw_data
            # Subsecond time conversion specified in 7516-9054 GSW-FSW ICD.
            # Value of SCLK subseconds, unsigned, (LSB = 1/256 sec)
            met = (
                packet.user_data["SC_SCLK_SEC"]
                + packet.user_data["SC_SCLK_SUB_SEC"] * 256
            )
            binary_blob_data.append(binary_blob)
            met_data.append(met)

    return binary_blob_data, met_data
