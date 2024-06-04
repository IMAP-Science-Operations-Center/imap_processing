"""Calculates Extended Spin."""

import numpy as np


def calculate_extendedspin(data_dict):
    """
    Create dictionary with defined datatypes for Extended Spin Data.

    Parameters
    ----------
    data_dict: : dict
        L1a data dictionary.

    Returns
    -------
    extendedspin_dict : dict
        Dictionary containing the data.
    """
    extendedspin_dict = {}

    dataset = data_dict["imap_ultra_l1a_45sensor-rates"]
    epoch = dataset.coords["epoch"].values

    # Placeholder for calculations
    extendedspin_dict["epoch"] = epoch
    extendedspin_dict["spin_number"] = np.zeros(len(epoch), dtype=np.uint64)
    extendedspin_dict["spin_start_time"] = np.zeros(len(epoch), dtype=np.float64)
    extendedspin_dict["avg_spin_period"] = np.zeros(len(epoch), dtype=np.float64)
    extendedspin_dict["rate_start_pulses"] = np.zeros(len(epoch), dtype=np.float64)
    extendedspin_dict["rate_stop_pulses"] = np.zeros(len(epoch), dtype=np.float64)
    extendedspin_dict["rate_coin_pulses"] = np.zeros(len(epoch), dtype=np.float64)
    extendedspin_dict["rate_processed_events"] = np.zeros(len(epoch), dtype=np.float64)
    extendedspin_dict["rate_rejected_events"] = np.zeros(len(epoch), dtype=np.float64)
    extendedspin_dict["quality_hk"] = np.zeros(len(epoch), dtype=np.uint16)
    extendedspin_dict["quality_attitude"] = np.zeros(len(epoch), dtype=np.uint16)
    extendedspin_dict["quality_instruments"] = np.zeros(len(epoch), dtype=np.uint16)

    return extendedspin_dict
