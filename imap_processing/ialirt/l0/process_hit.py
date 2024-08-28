"""Functions to support HIT processing."""

from enum import Enum


class HITPrefixes(Enum):
    """Create ENUM for each rate type."""

    FAST_RATE_1 = [
        f"{prefix}_{i:02d}"
        for i in range(15)
        for prefix in ["L1A_TRIG", "IA_EVNT_TRIG", "A_EVNT_TRIG", "L3A_TRIG"]
    ]
    FAST_RATE_2 = [
        f"{prefix}_{i:02d}"
        for i in range(15)
        for prefix in ["L1B_TRIG", "IB_EVNT_TRIG", "B_EVNT_TRIG", "L3B_TRIG"]
    ]
    SLOW_RATE = [
        "L1A",
        "L2A",
        "L3A",
        "L1A0AHG",
        "L1B0AHG",
        "L1C0AHG",
        "L4IAHG",
        "L4OAHG",
        "SLOW_RATE_08",
        "SLOW_RATE_09",
        "SLOW_RATE_10",
        "L1A0BHG",
        "L1B0BHG",
        "L1C0BHG",
        "L4IBHG",
        "L4OBHG",
        *[f"IALRT_RATE_{i}" for i in range(1, 21)],
        "TRIG_IA_EVNT",
        "TRIG_IB_EVNT",
        "NASIDE_IALRT",
        "NBSIDE_IALRT",
        *[f"ERATE_{i}" for i in range(1, 6)],
        "L12A",
        "L123A",
        "PENA",
        "L12B",
        "L123B",
        "PENB",
        "SLOW_RATE_51",
        "SLOW_RATE_52",
        "SLOW_RATE_53",
        "SLOW_RATE_54",
        "H_06_08",
        "H_12_15",
        "H_15_70",
        "HE4_06_08",
        "HE4_15_70",
    ]


def create_l1(fast_rate_1, fast_rate_2, slow_rate):
    """
    Create L1 data dictionary.

    Parameters
    ----------
    fast_rate_1 : xr.DataArray
        Fast rate 1 DataArray.
    fast_rate_2 : xr.DataArray
        Fast rate 1 DataArray.
    slow_rate : xr.DataArray
        Slow rate DataArray.

    Returns
    -------
    l1 : dict
        Dictionary containing parsed L0 packet data.
    """
    fast_rate_1_dict = {
        prefix: value
        for prefix, value in zip(HITPrefixes.FAST_RATE_1.value, fast_rate_1.data)
    }
    fast_rate_2_dict = {
        prefix: value
        for prefix, value in zip(HITPrefixes.FAST_RATE_2.value, fast_rate_2.data)
    }
    slow_rate_dict = {
        prefix: value
        for prefix, value in zip(HITPrefixes.SLOW_RATE.value, slow_rate.data)
    }

    l1 = {**fast_rate_1_dict, **fast_rate_2_dict, **slow_rate_dict}

    return l1


def process_hit(xarray_data):
    """
    Create L1 data dictionary.

    Parameters
    ----------
    xarray_data : dict(xr.Dataset)
        Dictionary of xarray data including a single
        set for processing.

    Returns
    -------
    hit_data : dict
        Dictionary final data product.
    """
    # Extract the HIT data from the xarray dataset
    fast_rate_1 = xarray_data["HIT"]["HIT_FAST_RATE_1"]
    fast_rate_2 = xarray_data["HIT"]["HIT_SLOW_RATE"]
    slow_rate = xarray_data["HIT"]["HIT_SLOW_RATE"]

    # Combine all prefixes and corresponding data into a single dictionary
    l1 = create_l1(fast_rate_1, fast_rate_2, slow_rate)

    # Structure the data in a dictionary format suitable for DynamoDB
    hit_data = {
        "HIT_lo_energy_e_A_side": (l1["IALRT_RATE_1"] + l1["IALRT_RATE_2"]).item(),
        "HIT_medium_energy_e_A_side": (l1["IALRT_RATE_5"] + l1["IALRT_RATE_6"]).item(),
        "HIT_high_energy_e_A_side": l1["IALRT_RATE_7"].item(),
        "HIT_low_energy_e_B_side": (l1["IALRT_RATE_11"] + l1["IALRT_RATE_12"]).item(),
        "HIT_medium_energy_e_B_side": (
            l1["IALRT_RATE_15"] + l1["IALRT_RATE_16"]
        ).item(),
        "HIT_high_energy_e_B_side": l1["IALRT_RATE_17"].item(),
        "HIT_medium_energy_H_omni": (l1["H_12_15"] + l1["H_15_70"]).item(),
        "HIT_high_energy_H_A_side": l1["IALRT_RATE_8"].item(),
        "HIT_high_energy_H_B_side": l1["IALRT_RATE_18"].item(),
        "HIT_low_energy_He_omni": l1["HE4_06_08"].item(),
        "HIT_high_energy_He_omni": l1["HE4_15_70"].item(),
    }

    return hit_data
