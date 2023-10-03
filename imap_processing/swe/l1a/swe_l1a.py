import logging

from imap_processing.swe.l0 import decom_swe
from imap_processing.swe.l1a.swe_science import swe_science


def swe_l1a(packet_file: str):
    """Process SWE l0 data into l1a data.

    Receive all L0 data file. Based on appId, it
    call its function to process. If appId is science, it requires more work
    than other appId such as appId of housekeeping.

    Parameters
    ----------
    packet_file : str
        The path and filename to the L0 file to read
    """
    decom_data = decom_swe.decom_packets(packet_file)
    logging.info(f"Unpacking data from {packet_file}")

    # If appId is science, then the file should contain all data of science appId
    if decom_data[0].header["PKT_APID"].raw_value == 1344:
        logging.info("Processing science data")
        return swe_science(decom_data=decom_data)
    else:
        return decom_data
