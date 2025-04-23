"""IMAP-HI l2 processing module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.utils import load_cdf
from imap_processing.ena_maps.ena_maps import PointingSet
from imap_processing.hi.l1c.hi_l1c import N_SPIN_BINS
from imap_processing.spice.geometry import SpiceFrame


class HiPointingSet(PointingSet):
    """
    PointingSet object specific to Hi L1C PSet data.

    Parameters
    ----------
    dataset : xarray.Dataset
        Hi L1C pointing set data loaded in an xarray.DataArray.
    """

    def __init__(self, dataset: xr.Dataset):
        self.spice_reference_frame = SpiceFrame.ECLIPJ2000
        self.data = dataset
        self.epoch = self.data["epoch"].values[0]
        self.num_points = N_SPIN_BINS
        self.az_el_points = np.column_stack(
            (
                np.squeeze(self.data["hae_longitude"]),
                np.squeeze(self.data["hae_latitude"]),
            )
        )
        self.spatial_coords = ("spin_angle_bin",)

    @classmethod
    def from_cdf(cls, cdf_path: Path) -> HiPointingSet:
        """
        Generate a HiPointingSet object from a CDF file.

        Parameters
        ----------
        cdf_path : str | Path
            Location of Hi L1C CDF file.

        Returns
        -------
        hi_pointing_set : HiPointingSet
            Input CDF file data loaded into HiPointingSet.
        """
        return cls(load_cdf(cdf_path))
