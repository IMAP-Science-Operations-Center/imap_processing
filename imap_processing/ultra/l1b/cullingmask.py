"""Calculates Culling Mask."""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def calculate_culling_mask(extended_spin_dict):
    """
    Create dictionary with defined datatype for Culling Mask Data.

    Parameters
    ----------
    extended_spin_dict: : dict
        L1b data dictionary.

    Returns
    -------
    culling_mask_dict : dict
        Dictionary containing the data.
    """
    culling_mask_dict = defaultdict(list)

    # Placeholder for bitwise filtering based on quality flags
    culling_mask_dict["epoch"] = extended_spin_dict["epoch"]
    culling_mask_dict["spin_number"] = extended_spin_dict["spin_number"]
    culling_mask_dict["spin_start_time"] = extended_spin_dict["spin_start_time"]
    culling_mask_dict["avg_spin_period"] = extended_spin_dict["avg_spin_period"]
    culling_mask_dict["rate_start_pulses"] = extended_spin_dict["rate_start_pulses"]
    culling_mask_dict["rate_stop_pulses"] = extended_spin_dict["rate_stop_pulses"]
    culling_mask_dict["rate_coin_pulses"] = extended_spin_dict["rate_coin_pulses"]
    culling_mask_dict["rate_processed_events"] = extended_spin_dict[
        "rate_processed_events"
    ]
    culling_mask_dict["rate_rejected_events"] = extended_spin_dict[
        "rate_rejected_events"
    ]

    return culling_mask_dict
