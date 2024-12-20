import numpy as np
import ultra_user.mapbin.binscheme as binscheme


class MapBinner():
    def __init__(self, bin_scheme: binscheme.BinScheme, initial_value=0):
        self.bin_scheme = bin_scheme
        n_bin = bin_scheme.nbins()
        self.bins = np.full(n_bin, initial_value)

    def add_value(self, value, longitude, latitude):
        self.bins[self.bin_scheme.bin_number(longitude, latitude)] += value
