"""Calculates Badtimes."""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def calculate_badtimes(extended_spin_dict):
    """
    Create dictionary with defined datatypes for Badtimes Data.

    Parameters
    ----------
    extended_spin_dict: : dict
        L1b data dictionary.

    Returns
    -------
    badtimes_dict : dict
        Dictionary containing the data.
    """
    badtimes_dict = defaultdict(list)

    # Placeholder for bitwise filtering based on quality flags
    badtimes_dict["epoch"] = extended_spin_dict["epoch"]
    badtimes_dict["spin_number"] = extended_spin_dict["spin_number"]
    badtimes_dict["spin_start_time"] = extended_spin_dict["spin_start_time"]
    badtimes_dict["avg_spin_period"] = extended_spin_dict["avg_spin_period"]
    badtimes_dict["rate_start_pulses"] = extended_spin_dict["rate_start_pulses"]
    badtimes_dict["rate_stop_pulses"] = extended_spin_dict["rate_stop_pulses"]
    badtimes_dict["rate_coin_pulses"] = extended_spin_dict["rate_coin_pulses"]
    badtimes_dict["rate_processed_events"] = extended_spin_dict["rate_processed_events"]
    badtimes_dict["rate_rejected_events"] = extended_spin_dict["rate_rejected_events"]

    return badtimes_dict
