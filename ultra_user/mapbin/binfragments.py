import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ultra_user.mapbin.lonLatscheme as lls
import ultra_user.mapbin.hpscheme as hps
import ultra_user.mapbin.mapBinner as mapBinner
import importlib
import healpy as hp
import pickle

#matplotlib.use('TkAgg')
#%matplotlib inline
plt.interactive(True)
importlib.reload(hps)
importlib.reload(lls)

binner0 = lls.LonLatScheme()

b00 = binner0.bin_number(0,0)

p00 = binner0.center_position(b00)
c00 = binner0.boundary_position(b00)

binner1 = hps.HpScheme()
b11 = binner1.bin_number(0,0)
p00 = binner1.center_position(b11)


# get L1B data in
pkl = open('data/pkls/dataset_l1b.pkl','rb')
l1b = pickle.load(pkl)
pkl.close()

v1b =l1b['v']
energy = l1b['energy']
vmag = l1b['v_mag']


az = np.degrees(np.atan2(v1b[1,:],v1b[0,:]))
el = np.degrees(np.arccos(v1b[2,:]/vmag))
lat = 90-el

mp0 = mapBinner.MapBinner(binner0)
mp1 = mapBinner.MapBinner(binner1)
for ic,oneaz in enumerate(az):
    mp0.add_value(1,oneaz,lat[ic])
    mp1.add_value(1,oneaz,lat[ic])

llmap = binner0.asGrid(mp0.bins)

plt.imshow(np.transpose(llmap))
plt.colorbar()
plt.show()

hp.cartview(mp1.bins)
hp.graticule()
plt.show()
