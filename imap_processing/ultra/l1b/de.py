"""Calculates Annotated Direct Events."""

import numpy as np


def calculate_de(data_dict):
    """
    Create dictionary with defined datatypes for Direct Event Data.

    Parameters
    ----------
    data_dict: : dict
        L1a data dictionary.

    Returns
    -------
    de_dict : dict
        Dictionary containing the data.
    """
    de_dict = {}

    dataset = data_dict["imap_ultra_l1a_45sensor-de"]
    epoch = dataset.coords["epoch"].values

    # Placeholder for calculations
    de_dict["epoch"] = epoch
    de_dict["x_front"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["y_front"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["x_back"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["y_back"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["x_coin"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["tof_start_stop"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["tof_stop_coin"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["tof_corrected"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["eventtype"] = np.zeros(len(epoch), dtype=np.uint64)
    de_dict["vx_ultra"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vy_ultra"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vz_ultra"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["energy"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["species"] = np.zeros(len(epoch), dtype=np.uint64)
    de_dict["event_efficiency"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vx_sc"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vy_sc"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vz_sc"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vx_dps_sc"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vy_dps_sc"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vz_dps_sc"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vx_dps_helio"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vy_dps_helio"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vz_dps_helio"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["eventtimes"] = np.zeros(len(epoch), dtype=np.float64)

    return de_dict
