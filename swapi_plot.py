"""Plot testings."""

import matplotlib.pyplot as plt
import numpy as np
from cdflib.xarray import cdf_to_xarray

# read swapi cdf file
cdf_file = "imap_swapi_l1_sci_20240924_v001.cdf"
ds = cdf_to_xarray(cdf_file, to_datetime=True)
print(ds)
# plot data using time in x-axis and energy in y-axis
x = ds["epoch"].data[144:]
pcem_count = ds["swp_pcem_counts"].data
# print(np.sum(pcem_count[144:], axis=1))
y = np.sum(pcem_count[144:], axis=1)
print(x.shape, y.shape)
plt.ylim(0, 1000)
plt.plot(x, y)
# # save plot as png file
plt.savefig("swapi_plot.png")
plt.show()
# plot another plot using time in x-axis and counts in y-axis
