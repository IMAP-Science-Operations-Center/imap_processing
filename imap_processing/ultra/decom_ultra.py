from imap_processing import decom_csv, packet_definition_directory


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
    csv_document = f"{packet_definition_directory}/ultra/0x371_TLM_U45FM_2023-08-22.csv"
    return decom_csv.decom_csv_packets(packet_file, csv_document)
