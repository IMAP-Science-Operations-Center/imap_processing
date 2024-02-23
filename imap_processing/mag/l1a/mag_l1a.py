"""Methods for decomming packets, processing to level 1A, and writing CDFs for MAG."""
import logging
from pathlib import Path

from imap_processing.cdf.utils import write_cdf
from imap_processing.mag.l0 import decom_mag

logger = logging.getLogger(__name__)


def mag_l1a(packet_filepath, output_filepath_norm, ouptput_filepath_burst):
    """
    Process MAG L0 data into L1A CDF files at cdf_filepath.

    Parameters
    ----------
    packet_filepath :
        Packet files for processing
    output_filepath_norm :
        Full directory and filename for raw-norm CDF file
    ouptput_filepath_burst :
        Full directory and filename for raw-burst CDF file
    """
    mag_l0 = decom_mag.decom_packets(packet_filepath)

    mag_norm, mag_burst = decom_mag.export_to_xarray(mag_l0)

    if mag_norm is not None:
        write_cdf(mag_norm, Path(output_filepath_norm))
        logging.info(f"Created CDF file at {output_filepath_norm}")

    if mag_burst is not None:
        write_cdf(mag_burst, Path(ouptput_filepath_burst))
        logging.info(f"Created CDF file at {output_filepath_norm}")
