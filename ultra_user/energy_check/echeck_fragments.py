import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import imap_processing.ultra.constants as constants





testcsv = "imap_processing/tests/ultra/test_data/l0/ultra45_raw_sc_ultrarawimg_withFSWcalcs_FM45_40P_Phi28p5_BeamCal_LinearScan_phi2850_theta-000_20240207T102740.csv"

df = pd.read_csv(testcsv)
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
