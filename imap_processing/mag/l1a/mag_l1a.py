"""Methods for decomming packets, processing to level 1A, and writing CDFs for MAG."""

import logging

from imap_processing.cdf.utils import write_cdf
from imap_processing.mag.l0 import decom_mag

logger = logging.getLogger(__name__)


def mag_l1a(packet_filepath):
    """
    Process MAG L0 data into L1A CDF files at cdf_filepath.

    Parameters
    ----------
    packet_filepath :
        Packet files for processing
    """
    mag_l0 = decom_mag.decom_packets(packet_filepath)

    mag_norm, mag_burst = decom_mag.export_to_xarray(mag_l0)

    if mag_norm is not None:
        file = write_cdf(mag_norm)
        logger.info(f"Created CDF file at {file}")

    if mag_burst is not None:
        file = write_cdf(mag_burst)
        logger.info(f"Created CDF file at {file}")


if __name__ == "__main__":
    filepath_burst = Path("mag_IT_data/MAG_SCI_BURST.bin")
    filepath_norm = Path("mag_IT_data/MAG_SCI_NORM.bin")
    mag_l1a(filepath_norm)
    mag_l1a(filepath_burst)
