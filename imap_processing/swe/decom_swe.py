from imap_processing import decom, packet_definition_directory


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
    xtce_document = f"{packet_definition_directory}/swe_packet_definition.xml"
    return decom.decom_packets(packet_file, xtce_document)
