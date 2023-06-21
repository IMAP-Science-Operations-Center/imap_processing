
from imap_processing import decom


def decom_packets(packet_file: str):
    """Decom SWE data packets using SWE packet definition
    Parameters
    ----------
    packet_file : str
        Path to data packet path with filename

    Returns
    -------
    List
        List of all the unpacked data
    """
    xtce_document = "imap_processing/packet_definitions/swe_packet_definition.xml"
    return decom.decom_packets(packet_file, xtce_document)
