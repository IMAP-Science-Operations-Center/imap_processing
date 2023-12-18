from imap_processing.mag.l0 import decom_mag
from imap_processing.cdf.utils import write_cdf

from pathlib import Path
def mag_l1a(packets, cdf_filepath):
    """
    Process MAG L0 data into L1A CDF files at cdf_filepath.

    Parameters
    ----------
    packets
    cdf_filepath

    Returns
    -------

    """

    mag_l0 = decom_mag.decom_packets(packets)

    mag_datasets = decom_mag.export_to_xarray(mag_l0)

    write_cdf(mag_datasets, mode="norm", directory="cdf_files")

if __name__ == '__main__':
    current_directory = Path(__file__).parent
    packets = current_directory.parent / "tests" / "mag_multiple_packets.pkts"
    mag_l1a(packets, current_directory)