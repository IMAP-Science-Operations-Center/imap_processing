from imap_processing import decom
from pathlib import Path


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

    # This helps gets current directory of the file irrespective
    # of where it is being called
    current_directory = Path(__file__).parent

    # Relative to current_directory, set path of packet definitions directory.
    packet_definition_directory = f"{current_directory}/../packet_definitions/"
    xtce_document = f"{packet_definition_directory}/swe_packet_definition.xml"
    return decom.decom_packets(packet_file, xtce_document)
