from typing import List, Tuple

import numpy as np

import ultra_user.mapbin.binscheme as binscheme


class LonLatScheme(binscheme.BinScheme):
    def __init__(self, name='lon/lat 2x2', lon_range=[-180, 180],
                 lat_range=[-90, 90], dlon=2, dlat=2, frame='ECLIPJ2000'):
        super().__init__()
        if lon_range is None:
            lon_range = [0, 360]
        self.name = name
        self.lon_range = lon_range
        self.lat_range = lat_range
        if np.abs(((lon_range[1] - lon_range[0]) % dlon) > 1.e-6 or
                  np.abs((lat_range[1] - lat_range[0]) % dlat) > 1.e-6):
            raise ValueError('Invalid grid specification')
        self.dlon = dlon
        self.dlat = dlat
        self.frame = frame
        self.nlon = np.round(np.abs((lon_range[1] - lon_range[0]) / dlon)).astype(int)
        self.nlat = np.round(np.abs((lat_range[1] - lat_range[0]) / dlat)).astype(int)

    def bin_number(self, lon: float, lat: float) -> int:
        if lon < self.lon_range[0] or lon > self.lon_range[1]:
            raise ValueError('longitude out of range')
        if lat < self.lat_range[0] or lat > self.lat_range[1]:
            raise ValueError('latitude out of range')

        n = np.floor((lon - self.lon_range[0]) / self.dlon).astype(int)
        m = np.floor((lat - self.lat_range[0]) / self.dlat).astype(int)
        return int(n + m * self.nlon)

    def center_position(self, bin_number: int) -> Tuple[float, float]:
        if bin_number <0 or bin_number >= self.nlon*self.nlat:
            raise ValueError('Illegal bin number')
        n, m = self.getnm(bin_number)
        return self.lon_range[0] + (n + 0.5) * self.dlon, self.lat_range[0] + (m + 0.5) * self.dlat

    def nbins(self) -> int:
        return self.nlon*self.nlat
    def boundary_position(self, bin_number) -> List[Tuple[float, float]]:
        if bin_number <0 or bin_number >= self.nlon*self.nlat:
            raise ValueError('Illegal bin number')
        n, m = self.getnm(bin_number)
        p0 = (self.lon_range[0] + n * self.dlon, self.lat_range[0] + m * self.dlat)
        p1 = (self.lon_range[0] + n * self.dlon, self.lat_range[0] + (m + 1) * self.dlat)
        p2 = (self.lon_range[0] + (n + 1) * self.dlon, self.lat_range[0] + (m + 1) * self.dlat)
        p3 = (self.lon_range[0] + (n + 1) * self.dlon, self.lat_range[0] + m * self.dlat)
        return [p0, p1, p2, p3]

    def getnm(self, bin_number):
        return ((bin_number % self.nlon),
                np.floor(bin_number * self.dlon / (self.lon_range[1] - self.lon_range[0])))
