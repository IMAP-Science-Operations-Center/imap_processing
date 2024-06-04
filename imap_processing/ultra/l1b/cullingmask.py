"""Calculates Culling Mask."""


def calculate_cullingmask(extended_spin_dict):
    """
    Create dictionary with defined datatype for Culling Mask Data.

    Parameters
    ----------
    extended_spin_dict: : dict
        L1b data dictionary.

    Returns
    -------
    cullingmask_dict : dict
        Dictionary containing the data.
    """
    cullingmask_dict = {}

    # Placeholder for bitwise filtering based on quality flags
    cullingmask_dict["epoch"] = extended_spin_dict["epoch"]
    cullingmask_dict["spin_number"] = extended_spin_dict["spin_number"]
    cullingmask_dict["spin_start_time"] = extended_spin_dict["spin_start_time"]
    cullingmask_dict["avg_spin_period"] = extended_spin_dict["avg_spin_period"]
    cullingmask_dict["rate_start_pulses"] = extended_spin_dict["rate_start_pulses"]
    cullingmask_dict["rate_stop_pulses"] = extended_spin_dict["rate_stop_pulses"]
    cullingmask_dict["rate_coin_pulses"] = extended_spin_dict["rate_coin_pulses"]
    cullingmask_dict["rate_processed_events"] = extended_spin_dict[
        "rate_processed_events"
    ]
    cullingmask_dict["rate_rejected_events"] = extended_spin_dict[
        "rate_rejected_events"
    ]

    return cullingmask_dict
