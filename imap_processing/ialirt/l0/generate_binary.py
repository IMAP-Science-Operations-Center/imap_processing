"""I-ALiRT data to populate binary blob database."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from space_packet_parser import definitions


def generate_binary(
    packet_file: Union[str, Path],
    xtce_packet_definition: Union[str, Path],
) -> list[dict]:
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
    ingest_data : list[dict]
        Dictionary final data product.
    """
    now = datetime.now(timezone.utc)
    ingest_data = []

    # Set up the parser from the input packet definition
    packet_definition = definitions.XtcePacketDefinition(xtce_packet_definition)
    with packet_file.open("rb") as binary_data:
        packet_generator = packet_definition.packet_generator(binary_data)

        # Iterate over the packets and access the raw binary data
        for packet in packet_generator:
            # Subsecond time conversion specified in 7516-9054 GSW-FSW ICD.
            # Value of SCLK subseconds, unsigned, (LSB = 1/256 sec)
            met = (
                packet.user_data["SC_SCLK_SEC"]
                + packet.user_data["SC_SCLK_SUB_SEC"] * 256
            )
            ingest_data.append(
                {
                    "apid": 478,
                    "met": met,
                    "ingest_time": now,
                    "packet_blob": packet.raw_data,
                }
            )

    return ingest_data
