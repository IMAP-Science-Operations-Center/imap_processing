"""Methods for decomming packets, processing to level 1A, and writing CDFs for MAG."""


from imap_processing.cdf.utils import write_cdf
from imap_processing.mag.l0 import decom_mag


def mag_l1a(packet_filepath, cdf_directory):
    """
    Process MAG L0 data into L1A CDF files at cdf_filepath.

    Parameters
    ----------
    packet_filepath:
        Packet files for processing
    cdf_directory:
        Directory for output
    """
    mag_l0 = decom_mag.decom_packets(packet_filepath)

    mag_datasets = decom_mag.export_to_xarray(mag_l0)

    write_cdf(mag_datasets, descriptor="norm", directory=cdf_directory)