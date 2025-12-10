"""SWAPI L2 processing module."""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.swapi.constants import NUM_ENERGY_STEPS

logger = logging.getLogger(__name__)


SWAPI_LIVETIME = 0.145  # seconds


def solve_full_sweep_energy(
    esa_lvl5_data: np.ndarray,
    sweep_table: np.ndarray,
    esa_table_df: pd.DataFrame,
    lut_notes_df: pd.DataFrame,
    data_time: npt.NDArray[np.datetime64],
) -> npt.NDArray:
    """
    Calculate the energy of each full sweep data.

    Get the fixed energy values for steps 0-62 using the
    esa_table_df information. It's important to ensure
    that the correct fixed energy values are selected for
    the specified time, as the sweep table can contain
    different values depending on the operational phase
    (e.g., I+T, pre-launch, post-launch). There may be
    more fixed energy added in the future. TODO: add
    document section once SWAPI document is updated.

    Now, find the last 9 fine energy values using steps
    noted in the section x in the algorithm document.

    Parameters
    ----------
    esa_lvl5_data : numpy.ndarray
        The L1 data input.
    sweep_table : numpy.ndarray
        Sweep table information.
    esa_table_df : pandas.DataFrame
        The ESA unit conversion table that contains first 63 energies.
    lut_notes_df : pandas.DataFrame
        The LUT notes table that contains the last 9 fine energies.
    data_time : numpy.ndarray
        The collection time of the data.

    Returns
    -------
    energy : numpy.ndarray
        The energy of each full sweep data.
    """
    # Convert timestamp from string to datetime
    # and to the same format as data_time
    esa_table_df["timestamp"] = pd.to_datetime(
        esa_table_df["timestamp"], format="%m/%d/%Y %H:%M"
    )
    esa_table_df["timestamp"] = esa_table_df["timestamp"].to_numpy(
        dtype="datetime64[ns]"
    )

    # Initialize the output energy array
    # Each sweep will have different energies for each step.
    # The first 63 energies are coarse steps, then followed by 9 fine steps.
    # The 9 fine steps may be defined in the main table (fixed steps), or "solve"
    # which requires a separate lookup in the lut-notes table.
    energy_data = np.empty((len(sweep_table), NUM_ENERGY_STEPS), dtype=float)

    for i_sweep, (time, sweep_id, esa_lvl5_val) in enumerate(
        zip(data_time, sweep_table, esa_lvl5_data, strict=True)
    ):
        # Find the sweep's ESA data for the given time and sweep_id
        subset = esa_table_df[
            (esa_table_df["timestamp"] <= time) & (esa_table_df["Sweep #"] == sweep_id)
        ]
        if subset.empty:
            # Get the earliest timestamp available
            earliest_time = esa_table_df["timestamp"].min()

            # Find the sweep's ESA data for the earliest time and sweep_id
            earliest_subset = esa_table_df[
                (esa_table_df["timestamp"] == earliest_time)
                & (esa_table_df["Sweep #"] == sweep_id)
            ]
            if earliest_subset.empty:
                raise ValueError(
                    f"No matching ESA table entry found for sweep ID {sweep_id} "
                    f"at time {time}, and no entries found for earliest time "
                    f"{earliest_time}."
                )
            subset = earliest_subset

        # Subset data can contain multiple sweeps of 72 energy values.
        # We need to sort by time and ESA step to maintain correct
        # order. Then take the last sweep (72 values).
        subset = subset.sort_values(["timestamp", "ESA Step #"]).iloc[
            -NUM_ENERGY_STEPS:
        ]
        sweep_esa_energies = subset["Energy"].values

        # Solve steps are the fine sweep steps. This can be variable numbers and is
        # not always the final 9 steps. They are negative values when reading in the df
        solve_steps = sweep_esa_energies < 0
        energy_data[i_sweep, ~solve_steps] = sweep_esa_energies[~solve_steps]
        if not np.any(solve_steps):
            # No solve steps, we've already filled all energies continue to next sweep
            continue

        # Page 31 of algorithm document
        # Get the last energy step index for use in looking up the fine sweep values
        # Find the index of the matching ESA DAC value
        matching_indices = np.nonzero(
            lut_notes_df["ESA DAC (Hex)"].values == esa_lvl5_val
        )[0]
        if len(matching_indices) == 0:
            raise ValueError(
                f"ESA DAC value '{esa_lvl5_val}' not found in LUT notes table "
                f"for sweep {i_sweep} at time {time}"
            )
        last_energy_step_index = matching_indices[0]

        # The ESA Index Number contains the offset indices for the fine energy values
        fine_offsets = subset["ESA Index Number"].values[solve_steps]
        # Since we are backtracking from the final index, we need to subtract that
        # offset from all of the other indices.
        fine_offsets -= fine_offsets[-1]
        fine_lut_indices = last_energy_step_index + fine_offsets

        # NOTE: Per SWAPI instruction, set every index that result in negative
        # indices during back tracking to zero index. SWAPI calls this
        # "flooring" the index. For example, if the 71st energy step index results
        # in less than 32, then it would result in some negative indices. Eg.
        #    71st index = 31
        #    nine fine energy indices = [31, 27, 23, 19, 15, 11, 7, 3, -1]
        #    flooring = [31, 27, 23, 19, 15, 11, 7, 3, 0]
        fine_lut_indices[fine_lut_indices < 0] = 0  # Ensure no negative indices

        energy_data[i_sweep, solve_steps] = lut_notes_df["Energy"].values[
            fine_lut_indices
        ]

    return energy_data


def swapi_l2(
    l1_dataset: xr.Dataset,
    esa_table_df: pd.DataFrame,
    lut_notes_df: pd.DataFrame,
) -> xr.Dataset:
    """
    Produce science data to L2.

    To process science data to L2, we need to:
    - convert counts to rates. This is done by dividing the counts by the
        SWAPI_LIVETIME time. LIVETIME is data acquisition time. It will
        be constant, SWAPI_LIVETIME = 0.145 s.

    - update uncertainty. Calculate new uncertainty value using
        SWP_PCEM_ERR data from level one and divide by SWAPI_LIVETIME. Eg.
            SWP_PCEM_UNC = SWP_PCEM_ERR / SWAPI_LIVETIME
        Do the same for SCEM and COIN data.

    Parameters
    ----------
    l1_dataset : xarray.Dataset
        The L1 data input.
    esa_table_df : pandas.DataFrame
        The ESA unit conversion table that contains first 63 energies.
    lut_notes_df : pandas.DataFrame
        The LUT notes table that contains the last 9 fine energies.

    Returns
    -------
    data : xarray.Dataset
        Processed data to L2.
    """
    # Load the CDF attributes
    cdf_manager = ImapCdfAttributes()
    cdf_manager.add_instrument_global_attrs("swapi")
    cdf_manager.add_instrument_variable_attrs(instrument="swapi", level=None)

    # Copy over only certain variables from L1 to L2 dataset
    l1_data_keys = [
        "epoch",
        "esa_lvl5",
        "esa_step",
        "esa_step_label",
        "fpga_rev",
        "fpga_type",
        "lut_choice",
        "plan_id",
        "sci_start_time",
        "sweep_table",
        "swp_l1a_flags",
    ]
    l2_dataset = l1_dataset[l1_data_keys]

    # Find energy of each full sweep data
    # -----------------------------------
    # Convert unpacked ESA_LVL5 values to hex to match the LUT table
    # value
    esa_lvl5_hex = np.vectorize(lambda x: format(x, "04X"))(
        l1_dataset["esa_lvl5"].values
    )

    # Turn the string start times into numpy datetime64
    sci_start_time = l1_dataset["sci_start_time"].values.astype("datetime64[ns]")

    esa_energy = solve_full_sweep_energy(
        esa_lvl5_hex,
        l1_dataset["sweep_table"].data,
        esa_table_df=esa_table_df,
        lut_notes_df=lut_notes_df,
        data_time=sci_start_time,
    )

    l2_dataset["esa_energy"] = xr.DataArray(
        esa_energy,
        name="esa_energy",
        dims=["epoch", "esa_step"],
        attrs=cdf_manager.get_variable_attributes("esa_energy"),
    )

    # Update L2 specific attributes
    l2_global_attrs = cdf_manager.get_global_attributes("imap_swapi_l2_sci")
    l2_dataset.attrs["Data_type"] = l2_global_attrs["Data_type"]
    l2_dataset.attrs["Logical_source"] = l2_global_attrs["Logical_source"]
    l2_dataset.attrs["Logical_source_description"] = l2_global_attrs[
        "Logical_source_description"
    ]

    # convert counts to rate
    l2_dataset["swp_pcem_rate"] = l1_dataset["swp_pcem_counts"] / SWAPI_LIVETIME
    l2_dataset["swp_scem_rate"] = l1_dataset["swp_scem_counts"] / SWAPI_LIVETIME
    l2_dataset["swp_coin_rate"] = l1_dataset["swp_coin_counts"] / SWAPI_LIVETIME

    # NOTE: The counts can be negative from FILLVAL in l1a data. We want to ignore those
    #       and propagate nans
    for var in ["swp_pcem_rate", "swp_scem_rate", "swp_coin_rate"]:
        l2_dataset[var] = l2_dataset[var].where(l2_dataset[var] >= 0, np.nan)
    # update attrs
    l2_dataset["swp_pcem_rate"].attrs = cdf_manager.get_variable_attributes("pcem_rate")
    l2_dataset["swp_scem_rate"].attrs = cdf_manager.get_variable_attributes("scem_rate")
    l2_dataset["swp_coin_rate"].attrs = cdf_manager.get_variable_attributes("coin_rate")

    # update uncertainty
    l2_dataset["swp_pcem_rate_stat_uncert_plus"] = (
        l1_dataset["swp_pcem_counts_stat_uncert_plus"] / SWAPI_LIVETIME
    )
    l2_dataset["swp_pcem_rate_stat_uncert_minus"] = (
        l1_dataset["swp_pcem_counts_stat_uncert_minus"] / SWAPI_LIVETIME
    )
    l2_dataset["swp_scem_rate_stat_uncert_plus"] = (
        l1_dataset["swp_scem_counts_stat_uncert_plus"] / SWAPI_LIVETIME
    )
    l2_dataset["swp_scem_rate_stat_uncert_minus"] = (
        l1_dataset["swp_scem_counts_stat_uncert_minus"] / SWAPI_LIVETIME
    )
    l2_dataset["swp_coin_rate_stat_uncert_plus"] = (
        l1_dataset["swp_coin_counts_stat_uncert_plus"] / SWAPI_LIVETIME
    )
    l2_dataset["swp_coin_rate_stat_uncert_minus"] = (
        l1_dataset["swp_coin_counts_stat_uncert_minus"] / SWAPI_LIVETIME
    )
    # update attrs
    l2_dataset[
        "swp_pcem_rate_stat_uncert_plus"
    ].attrs = cdf_manager.get_variable_attributes("pcem_rate_uncertainty")
    l2_dataset[
        "swp_pcem_rate_stat_uncert_minus"
    ].attrs = cdf_manager.get_variable_attributes("pcem_rate_uncertainty")
    l2_dataset[
        "swp_scem_rate_stat_uncert_plus"
    ].attrs = cdf_manager.get_variable_attributes("scem_rate_uncertainty")
    l2_dataset[
        "swp_scem_rate_stat_uncert_minus"
    ].attrs = cdf_manager.get_variable_attributes("scem_rate_uncertainty")
    l2_dataset[
        "swp_coin_rate_stat_uncert_plus"
    ].attrs = cdf_manager.get_variable_attributes("coin_rate_uncertainty")
    l2_dataset[
        "swp_coin_rate_stat_uncert_minus"
    ].attrs = cdf_manager.get_variable_attributes("coin_rate_uncertainty")

    # TODO: add thruster firing flag
    # TODO: add other flags
    logger.info("SWAPI L2 processing complete")

    return l2_dataset
