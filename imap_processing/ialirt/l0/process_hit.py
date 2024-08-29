"""Functions to support HIT processing."""

import logging
from enum import Enum
from typing import ClassVar

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class HITPrefixes(Enum):
    """Create ENUM for each rate type."""

    FAST_RATE_1: ClassVar[list[str]] = [
        f"{prefix}_{i:02d}"
        for i in range(15)
        for prefix in ["L1A_TRIG", "IA_EVNT_TRIG", "A_EVNT_TRIG", "L3A_TRIG"]
    ]
    FAST_RATE_2: ClassVar[list[str]] = [
        f"{prefix}_{i:02d}"
        for i in range(15)
        for prefix in ["L1B_TRIG", "IB_EVNT_TRIG", "B_EVNT_TRIG", "L3B_TRIG"]
    ]
    SLOW_RATE: ClassVar[list[str]] = [
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


def find_groups(data: xr.Dataset) -> xr.Dataset:
    """
    Find groups in the data.

    Parameters
    ----------
    data : xr.Dataset
        HIT Dataset.

    Returns
    -------
    grouped_data : xr.Dataset
        Grouped data.
    """
    subcom_range = (0, 59)

    data = data.sortby("HIT_SC_TICK", ascending=True)

    # Filter out data before the first subcom=0 and after the last subcom=59
    start_sc_ticks = data["HIT_SC_TICK"][(data["HIT_SUBCOM"] == subcom_range[0]).values]
    start_sc_tick = start_sc_ticks.data.min()
    last_sc_ticks = data["HIT_SC_TICK"][([data["HIT_SUBCOM"] == subcom_range[-1]][-1])]
    last_sc_tick = last_sc_ticks.data.max()

    grouped_data = data.where(
        (data["HIT_SC_TICK"] >= start_sc_tick) & (data["HIT_SC_TICK"] <= last_sc_tick),
        drop=True,
    )

    # Create group labels
    group_labels = np.searchsorted(
        start_sc_ticks, grouped_data["HIT_SC_TICK"], side="right"
    )
    grouped_data["group"] = ("group", group_labels)

    return grouped_data


def create_l1(
    fast_rate_1: xr.DataArray,
    fast_rate_2: xr.DataArray,
    slow_rate: xr.DataArray,
) -> dict[str, float]:
    """
    Create L1 data dictionary.

    Parameters
    ----------
    fast_rate_1 : xr.DataArray
        Fast rate 1 DataArray.
    fast_rate_2 : xr.DataArray
        Fast rate 2 DataArray.
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


def process_hit(xarray_data: xr.Dataset) -> list[dict]:
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
    hit_data = []
    grouped_data = find_groups(xarray_data)

    for group in np.unique(grouped_data["group"]):
        fast_rate_1 = grouped_data["HIT_FAST_RATE_1"][
            (grouped_data["group"] == group).values
        ]
        fast_rate_2 = grouped_data["HIT_FAST_RATE_2"][
            (grouped_data["group"] == group).values
        ]
        slow_rate = grouped_data["HIT_SLOW_RATE"][
            (grouped_data["group"] == group).values
        ]
        met = int(grouped_data["HIT_MET"][(grouped_data["group"] == group).values][0])

        # Verify that each group has 60 datapoints
        if len(slow_rate.data) != 60:
            logger.debug(
                f"Incorrect number of packets: {len(slow_rate.data)} "
                f"for met {met}. Skipping."
            )
            continue

        l1 = create_l1(fast_rate_1, fast_rate_2, slow_rate)

        hit_data.append(
            {
                "met": met,
                "hit_lo_energy_e_A_side": float(
                    l1["IALRT_RATE_1"] + l1["IALRT_RATE_2"]
                ),
                "hit_medium_energy_e_A_side": float(
                    l1["IALRT_RATE_5"] + l1["IALRT_RATE_6"]
                ),
                "hit_high_energy_e_A_side": float(l1["IALRT_RATE_7"]),
                "hit_low_energy_e_B_side": float(
                    l1["IALRT_RATE_11"] + l1["IALRT_RATE_12"]
                ),
                "hit_medium_energy_e_B_side": float(
                    l1["IALRT_RATE_15"] + l1["IALRT_RATE_16"]
                ),
                "hit_high_energy_e_B_side": float(l1["IALRT_RATE_17"]),
                "hit_medium_energy_H_omni": float(l1["H_12_15"] + l1["H_15_70"]),
                "hit_high_energy_H_A_side": float(l1["IALRT_RATE_8"]),
                "hit_high_energy_H_B_side": float(l1["IALRT_RATE_18"]),
                "hit_low_energy_He_omni": float(l1["HE4_06_08"]),
                "hit_high_energy_He_omni": float(l1["HE4_15_70"]),
            }
        )

    return hit_data
