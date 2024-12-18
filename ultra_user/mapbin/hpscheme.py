from typing import List, Tuple

import numpy as np
import healpy as hp
import ultra_user.mapbin.binscheme as binscheme


class HpScheme(binscheme.BinScheme):
    def __init__(self, name='healpix ~2d', nside=32,
                 nest=True, lonlat=True, frame='ECLIPJ2000'):
        super().__init__()
        self.name = name
        self.nside = nside
        self.nest = nest
        self.lonlat = lonlat
        self.frame = frame

    def bin_number(self, lon: float, lat: float) -> int:
        return hp.ang2pix(self.nside, lon, lat, nest=self.nest, lonlat=self.lonlat)

    def center_position(self, bin_number: int) -> Tuple[float, float]:
        return hp.pix2ang(self.nside, bin_number, nest=self.nest, lonlat=self.lonlat)

    def nbins(self):
        return hp.nside2npix(self.nside)
