"""Functions to support HIT processing."""

import logging
from decimal import Decimal

import numpy as np
import xarray as xr

from imap_processing.hit.l0.decom_hit import decompress_rates_16_to_32
from imap_processing.hit.l1b.hit_l1b import livetime_fraction_calculation
from imap_processing.ialirt.utils.grouping import (
    _populate_instrument_header_items,
    find_groups,
)
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import met_to_ttj2000ns

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
        Slow rate DataArray. Values are compressed 16-bit rates and are
        decompressed here.

    Returns
    -------
    l1 : dict
        Dictionary containing parsed L0 packet data.
    """
    # Slow rates are compressed onboard with the same 16-bit algorithm used for
    # the science rates (HIT Algorithm Document Sec. 5.1), confirmed by the HIT
    # team. Decompress them before use.
    slow_rate_decompressed = [
        decompress_rates_16_to_32(int(value)) for value in slow_rate.data
    ]

    fast_rate_1_dict = {
        prefix: value
        for prefix, value in zip(
            HIT_PREFIX_TO_RATE_TYPE["FAST_RATE_1"], fast_rate_1.data, strict=False
        )
    }
    fast_rate_2_dict = {
        prefix: value
        for prefix, value in zip(
            HIT_PREFIX_TO_RATE_TYPE["FAST_RATE_2"], fast_rate_2.data, strict=False
        )
    }
    slow_rate_dict = {
        prefix: value
        for prefix, value in zip(
            HIT_PREFIX_TO_RATE_TYPE["SLOW_RATE"],
            slow_rate_decompressed,
            strict=False,
        )
    }

    l1 = {**fast_rate_1_dict, **fast_rate_2_dict, **slow_rate_dict}

    return l1


def calculate_livetime_fraction(l1: dict) -> float:
    """
    Calculate the livetime fraction from the I-ALiRT livetime counter.

    The livetime counter (ERATE_1, slow rate subcom 40) is the same LIVE_TIME
    counter used by HIT science processing, so the same conversion
    (HIT Algorithm Document Sec. 6.2, Eq. 8) is applied.

    If ERATE_1 through ERATE_5 are all zero, no livetime was reported for the
    group (e.g. before the ERATES were populated in I-ALiRT on 2026-05-18, or
    the first group after a HIT restart) and NaN is returned.

    Parameters
    ----------
    l1 : dict
        Decompressed L1 rates for one group, as returned by ``create_l1``.

    Returns
    -------
    livetime_fraction : float
        Livetime fraction, or NaN if no livetime was reported.
    """
    if not any(l1[f"ERATE_{i}"] for i in range(1, 6)):
        return np.nan

    livetime = livetime_fraction_calculation(xr.DataArray([l1["ERATE_1"]]))
    return float(livetime.values[0])


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
    incomplete_groups = []
    status_groups = []

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
        status_values = grouped_data["hit_status"][
            (grouped_data["group"] == group).values
        ]

        if np.any(status_values == 0):
            status_groups.append(group)

        # Subcom values for the group should be 0-59 with no duplicates.
        subcom_values = grouped_data["hit_subcom"][
            (grouped_data["group"] == group).values
        ]

        # Ensure no duplicates and all values from 0 to 59 are present
        if not np.array_equal(subcom_values, np.arange(60)):
            incomplete_groups.append(group)
            continue

        hit_met = int(
            grouped_data["hit_met"][(grouped_data["group"] == group).values].values[0]
        )

        fast_rate_1 = grouped_data["hit_fast_rate_1"][
            (grouped_data["group"] == group).values
        ]
        fast_rate_2 = grouped_data["hit_fast_rate_2"][
            (grouped_data["group"] == group).values
        ]
        slow_rate = grouped_data["hit_slow_rate"][
            (grouped_data["group"] == group).values
        ]
        met = grouped_data["met"][(grouped_data["group"] == group).values]

        l1 = create_l1(fast_rate_1, fast_rate_2, slow_rate)

        hit_data.append(
            _populate_instrument_header_items(met)
            | {
                "instrument": "hit",
                "hit_epoch": int(met_to_ttj2000ns(hit_met)),
                "hit_e_a_side_low_en": Decimal(
                    f"{l1['IALRT_RATE_1'] + l1['IALRT_RATE_2']:.3f}"
                ),
                "hit_e_a_side_med_en": Decimal(
                    f"{l1['IALRT_RATE_5'] + l1['IALRT_RATE_6']:.3f}"
                ),
                "hit_e_a_side_high_en": Decimal(f"{l1['IALRT_RATE_7']:.3f}"),
                "hit_e_b_side_low_en": Decimal(
                    f"{l1['IALRT_RATE_11'] + l1['IALRT_RATE_12']:.3f}"
                ),
                "hit_e_b_side_med_en": Decimal(
                    f"{l1['IALRT_RATE_15'] + l1['IALRT_RATE_16']:.3f}"
                ),
                "hit_e_b_side_high_en": Decimal(f"{l1['IALRT_RATE_17']:.3f}"),
                "hit_h_omni_low_en": Decimal(f"{l1['H_06_08']:.3f}"),
                "hit_h_omni_med_en": Decimal(f"{l1['H_12_15']:.3f}"),
                "hit_h_a_side_high_en": Decimal(f"{l1['IALRT_RATE_8']:.3f}"),
                "hit_h_b_side_high_en": Decimal(f"{l1['IALRT_RATE_18']:.3f}"),
                "hit_he_omni_low_en": Decimal(f"{l1['HE4_06_08']:.3f}"),
                "hit_he_omni_high_en": Decimal(f"{l1['HE4_15_70']:.3f}"),
            }
        )

    if incomplete_groups:
        logger.info(
            f"The following hit groups were skipped due to "
            f"missing or duplicate pkt_counter values: "
            f"{incomplete_groups}"
        )
    if status_groups:
        logger.warning(
            f"The following hit groups have zero status values: {status_groups}"
        )

    return hit_data
