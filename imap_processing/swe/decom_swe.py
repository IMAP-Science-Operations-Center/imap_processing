from imap_processing import decom


def decom_swe_packets(packet_file: str, xtce_packet_definition: str):
    """Decom SWE data packets

    Parameters
    ----------
    packet_file : str
        Path to data packet path with filename
    xtce_packet_definition : str
        Path to XTCE file with filename

    Returns
    -------
    List
        List of all the unpacked data
    """
    return decom.decom_packets(packet_file, xtce_packet_definition)
