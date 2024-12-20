import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib

import imap_processing.ultra.constants as constants
import ultra_user.energy_check.event_dataset as ed
import ultra_user.pipeline.test_data as td
import imap_processing.ultra.l1b.ultra_l1b_extended as l1b_ext
import ultra_user.energy_check.de_extended_calcs as de_calcs
plt.interactive(True)

importlib.reload(ed)
importlib.reload(td)
importlib.reload(de_calcs)

d = ed.EventDataset()
dtest = td.de_dataset()

l1b=de_calcs.get_1bdict(dtest)
v = np.asarray(l1b["v"])
v_mag = l1b["v_mag"]

v_mag0=np.sqrt(np.sum((v*v),0))

ehist,echan = np.histogram(l1b["energy"],bins=20,range = (20,60))
vhist,vchan = np.histogram(v_mag,bins=20,range = (100,5000))

plt.plot(vchan[1:],vhist)
plt.xlabel("Velocity (km/s)")
plt.title("Histogram of test dataset")
plt.show()

plt.plot(echan[1:],ehist)
plt.xlabel("Energy (keV)")
plt.title("Histogram of test dataset")
plt.show()


d.ctof_ph_plot()

v0 = d.get_v_ctof()
v1 = d.get_v_r()


################# older fragments
ctof = d.get_cTOF()
ph = d.get_ph_or_e()
bin = d.get_cTOF_bin()
f = pd.read_csv(testcsv)
df_filt = df[df["StartType"] != -1]

cTOF = df_filt["cTOF"].astype("float").values
bin = df_filt["ComputedBin"].astype("float").values
e_xx = np.array(df_filt["Energy"].astype("float"))
e_yy = np.array(df_filt["EnergyOrPH"].astype("float"))
r = np.array(df_filt["r"].astype("float"))
dmin = constants.UltraConstants.DMIN

v = dmin/cTOF *1.e7  #(unit conversion mm/tenths of a ns)
amu = 1.66054e-27

Ej = .05*amu*v*v
J_per_Kev = 1.60218e-16
Ekev = Ej/J_per_Kev
hbins = 10**(np.arange(20)/20)
Ehist,edges = np.histogram(Ekev,bins=hbins)
binhist,binedge = np.histogram(bin,bins=np.arange(64))

bina_index = np.where(np.logical_and( bin >5 ,bin < 10))
binb_index = np.where(np.logical_and( bin >21 ,bin < 27))
binc_index = np.where(np.logical_and( bin >40 ,bin < 50))

Ehista,edges = np.histogram(Ekev[bina_index],bins=hbins)
Ehistb,edges = np.histogram(Ekev[binb_index],bins=hbins)
Ehistc,edges = np.histogram(Ekev[binc_index],bins=hbins)


plt.plot(np.sqrt(hbins[0:-1]*hbins[1:]),Ehista)
plt.plot(np.sqrt(hbins[0:-1]*hbins[1:]),Ehistb)
plt.plot(np.sqrt(hbins[0:-1]*hbins[1:]),Ehistc)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Energy")
plt.ylabel("Number of events")
plt.show()

plt.plot(Ekev[bina_index],e_xx[bina_index],"o")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Ekev")
plt.ylabel("E_ph")
plt.show()
