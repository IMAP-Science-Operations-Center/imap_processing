"""Decom IMAP-Hi data."""
from imap_processing import decom, imap_module_directory


def decom_packets(packet_file_path: str):
    """Decom IMAP-Hi data using its packet definition."""
    packet_def_file = (
        imap_module_directory / "hi/packet_definitions/hi_packet_definition.xml"
    )
    return decom.decom_packets(packet_file_path, packet_def_file)
