"""Decom IMAP-Hi data."""

from imap_processing import decom, imap_module_directory


def decom_packets(packet_file_path: str) -> list:
    """
    Decom IMAP-Hi data using its packet definition.

    Parameters
    ----------
    packet_file_path : str
        File path to the packet.

    Returns
    -------
    list
        Decompressed file packets.
    """
    packet_def_file = (
        imap_module_directory / "hi/packet_definitions/hi_packet_definition.xml"
    )
    decom_file_packets: list = decom.decom_packets(packet_file_path, packet_def_file)
    return decom_file_packets
