"""Functions to support HIT processing."""

import logging

import numpy as np
import xarray as xr

from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import met_to_ttj2000ns, met_to_utc

logger = logging.getLogger(__name__)


# HIT prefixes as defined by Table 37 of the HIT Algorithm Document.
HIT_PREFIX_TO_RATE_TYPE = {
    "FAST_RATE_1": [
        f"{prefix}_{i:02d}"
        for i in range(15)
        for prefix in ["L1A_TRIG", "IA_EVNT_TRIG", "A_EVNT_TRIG", "L3A_TRIG"]
    ],
    "FAST_RATE_2": [
        f"{prefix}_{i:02d}"
        for i in range(15)
        for prefix in ["L1B_TRIG", "IB_EVNT_TRIG", "B_EVNT_TRIG", "L3B_TRIG"]
    ],
    "SLOW_RATE": [
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
    ],
}


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
        for prefix, value in zip(
            HIT_PREFIX_TO_RATE_TYPE["FAST_RATE_1"], fast_rate_1.data
        )
    }
    fast_rate_2_dict = {
        prefix: value
        for prefix, value in zip(
            HIT_PREFIX_TO_RATE_TYPE["FAST_RATE_2"], fast_rate_2.data
        )
    }
    slow_rate_dict = {
        prefix: value
        for prefix, value in zip(HIT_PREFIX_TO_RATE_TYPE["SLOW_RATE"], slow_rate.data)
    }

    l1 = {**fast_rate_1_dict, **fast_rate_2_dict, **slow_rate_dict}

    return l1


def process_hit(xarray_data: xr.Dataset) -> list[dict]:
    """
    Create L1 data dictionary.

    Parameters
    ----------
    xarray_data : xr.Dataset
        Parsed data.

    Returns
    -------
    hit_data : list[dict]
        Dictionary final data product.
    """
    hit_data = []

    # Subsecond time conversion specified in 7516-9054 GSW-FSW ICD.
    # Value of SCLK subseconds, unsigned, (LSB = 1/256 sec)
    met = calculate_time(
        xarray_data["sc_sclk_sec"], xarray_data["sc_sclk_sub_sec"], 256
    )

    # Add required parameters.
    xarray_data["met"] = met

    grouped_data = find_groups(xarray_data, (0, 59), "hit_subcom", "met")
    unique_groups = np.unique(grouped_data["group"])

    for group in unique_groups:
        # Subcom values for the group should be 0-59 with no duplicates.
        subcom_values = grouped_data["hit_subcom"][
            (grouped_data["group"] == group).values
        ]

        # Ensure no duplicates and all values from 0 to 59 are present
        if not np.array_equal(subcom_values, np.arange(60)):
            logger.warning(
                f"Group {group} does not contain all values from 0 to "
                f"59 without duplicates."
            )
            continue

        fast_rate_1 = grouped_data["hit_fast_rate_1"][
            (grouped_data["group"] == group).values
        ]
        fast_rate_2 = grouped_data["hit_fast_rate_2"][
            (grouped_data["group"] == group).values
        ]
        slow_rate = grouped_data["hit_slow_rate"][
            (grouped_data["group"] == group).values
        ]
        met = int(grouped_data["met"][(grouped_data["group"] == group).values][0])

        l1 = create_l1(fast_rate_1, fast_rate_2, slow_rate)

        hit_data.append(
            {
                "apid": 478,
                "met": int(met),
                "met_in_utc": met_to_utc(met).split(".")[0],
                "ttj2000ns": int(met_to_ttj2000ns(met)),
                "hit_e_a_side_low_en": int(l1["IALRT_RATE_1"] + l1["IALRT_RATE_2"]),
                "hit_e_a_side_med_en": int(l1["IALRT_RATE_5"] + l1["IALRT_RATE_6"]),
                "hit_e_a_side_high_en": int(l1["IALRT_RATE_7"]),
                "hit_e_b_side_low_en": int(l1["IALRT_RATE_11"] + l1["IALRT_RATE_12"]),
                "hit_e_b_side_med_en": int(l1["IALRT_RATE_15"] + l1["IALRT_RATE_16"]),
                "hit_e_b_side_high_en": int(l1["IALRT_RATE_17"]),
                "hit_h_omni_med_en": int(l1["H_12_15"] + l1["H_15_70"]),
                "hit_h_a_side_high_en": int(l1["IALRT_RATE_8"]),
                "hit_h_b_side_high_en": int(l1["IALRT_RATE_18"]),
                "hit_he_omni_low_en": int(l1["HE4_06_08"]),
                "hit_he_omni_high_en": int(l1["HE4_15_70"]),
            }
        )

    return hit_data
