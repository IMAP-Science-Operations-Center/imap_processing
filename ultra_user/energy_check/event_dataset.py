import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imap_processing.ultra.constants as constants


class EventDataset:
    def __init__(self, clean=True,
                 csv="imap_processing/tests/ultra/test_data/l0/"+
                     "ultra45_raw_sc_ultrarawimg_withFSWcalcs_FM45_40P_Phi28p5_BeamCal_LinearScan_phi2850_theta-000_20240207T102740.csv"):
        df_raw = pd.read_csv(csv)
        if clean:
            self.df = df_raw[df_raw["StartType"] != -1]
        else:
            self.df = df_raw


    def get_cTOF(self):
        return np.array(self.df["cTOF"].astype("float").values)

    def get_ph_or_e(self):
        return np.array(self.df["EnergyOrPH"].astype("float"))

    def get_cTOF_bin(self):
        return np.array(self.df["ComputedBin"].astype("float"))

    def get_v_ctof(self):
        cTOF = self.get_cTOF()
        issd = self.ssd_indices()
        dmin_mcp = constants.UltraConstants.DMIN # will change to DMIN_PH_CTOF
        dmin_ssd = constants.UltraConstants.DMIN_SSD_CTOF

        v = dmin_mcp / cTOF
        v[issd] = dmin_ssd/cTOF[issd]
        return v * 1.e7  # (unit conversion mm/tenths of a ns)

    def get_v_r(self):
        r = np.array(self.df["r"].astype("float"))
        tof = np.array(self.df["TOF"].astype("float"))
        return (r / tof) * 1.e5

    def get_stop_type(self):
        return np.array(self.df["StopType"].astype("int"))

    def ssd_indices(self):
        stop_type = self.get_stop_type()
        return np.where(stop_type >2)
    def mcp_indices(self):
        stop_type = self.get_stop_type()
        return np.where(stop_type <=2)

    def ctof_ph_plot(self,binlim=50, xr=[0, 300], cmap='viridis', vmin=None, vmax=None):
        ctof = self.get_cTOF()
        ph = self.get_ph_or_e()
        bin = self.get_cTOF_bin()
        ibin = np.where(bin < binlim)
        plt.scatter(ctof[ibin], ph[ibin], c=bin[ibin], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.xlim(xr)
        plt.xlabel("cTOF")
        plt.ylabel("Pulse Height")
        plt.show()
