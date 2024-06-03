"""Calculates Extended Spin."""

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def calculate_extended_spin(data_dict):
    """
    Create dictionary with defined datatypes for Extended Spin Data.

    Parameters
    ----------
    data_dict: : dict
        L1a data dictionary.

    Returns
    -------
    extended_spin_dict : dict
        Dictionary containing the data.
    """
    extended_spin_dict = defaultdict(list)

    dataset = data_dict["imap_ultra_l1a_45sensor-rates"]
    epoch = dataset.coords["epoch"].values

    # Placeholder for calculations
    extended_spin_dict["epoch"] = epoch
    extended_spin_dict["spin_number"] = np.zeros(len(epoch), dtype=np.uint64)
    extended_spin_dict["spin_start_time"] = np.zeros(len(epoch), dtype=np.float64)
    extended_spin_dict["avg_spin_period"] = np.zeros(len(epoch), dtype=np.float64)
    extended_spin_dict["rate_start_pulses"] = np.zeros(len(epoch), dtype=np.float64)
    extended_spin_dict["rate_stop_pulses"] = np.zeros(len(epoch), dtype=np.float64)
    extended_spin_dict["rate_coin_pulses"] = np.zeros(len(epoch), dtype=np.float64)
    extended_spin_dict["rate_processed_events"] = np.zeros(len(epoch), dtype=np.float64)
    extended_spin_dict["rate_rejected_events"] = np.zeros(len(epoch), dtype=np.float64)
    extended_spin_dict["quality_hk"] = np.zeros(len(epoch), dtype=np.uint16)
    extended_spin_dict["quality_attitude"] = np.zeros(len(epoch), dtype=np.uint16)
    extended_spin_dict["quality_instruments"] = np.zeros(len(epoch), dtype=np.uint16)

    return extended_spin_dict
