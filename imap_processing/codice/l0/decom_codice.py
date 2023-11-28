from imap_processing import decom, imap_module_directory


def decom_packets(packet_file: str):
    """Decom CoDICE data packets using CoDICE packet definition.

    Parameters
    ----------
    packet_file : str
        Path to data packet path with filename.

    Returns
    -------
    list : list
        all the unpacked data.
    """
    xtce_document = f"{imap_module_directory}/codice/packet_definitions/P_COD_NHK.xml"
    return decom.decom_packets(packet_file, xtce_document)
