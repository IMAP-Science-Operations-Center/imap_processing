"""Decommutates SWE CCSDS data packets."""

from imap_processing import decom, imap_module_directory


def decom_packets(packet_file: str) -> list:
    """
    Decom SWE data packets using SWE packet definition.

    Parameters
    ----------
    packet_file : str
        Path to data packet path with filename.

    Returns
    -------
    List
        List of all the unpacked data.
    """
    unpacked_data: list
    xtce_document = (
        f"{imap_module_directory}/swe/packet_definitions/swe_packet_definition.xml"
    )
    unpacked_data = decom.decom_packets(packet_file, xtce_document)
    return unpacked_data
