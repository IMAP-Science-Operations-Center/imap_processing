"""Define utils and classes related to coordinates of the ENA maps."""

from enum import Enum


class CoordNames(Enum):
    """Enumeration of the names of the coordinates in the L1C and L2 ENA datasets."""

    TIME = "epoch"
    ENERGY = "energy_bin_center"
    HEALPIX_INDEX = "healpix_pixel_index"

    # The names of the az/el angular coordinates may differ between L1C and L2 data
    AZIMUTH_L1C = "longitude_bin_center"
    ELEVATION_L1C = "latitude_bin_center"
    AZIMUTH_L2 = "longitude_bin_center"
    ELEVATION_L2 = "latitude_bin_center"
