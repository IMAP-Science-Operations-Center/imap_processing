import numpy as np
import ultra_user.mapbin.lonLatscheme as lls
import ultra_user.mapbin.hpscheme as hps
import importlib
import healpy as hp

importlib.reload(hps)
importlib.reload(lls)

binner0 = lls.LonLatScheme()

b00 = binner0.bin_number(0,0)

p00 = binner0.center_position(b00)
c00 = binner0.boundary_position(b00)

binner1 = hps.HpScheme()
b11 = binner1.bin_number(0,0)
p00 = binner1.center_position(b11)


# pixels in 2d rectangular tess
np0 = 180*90
ns0=hp.npix2nside(np0)
np1 = hp.nside2npix(ns0)

