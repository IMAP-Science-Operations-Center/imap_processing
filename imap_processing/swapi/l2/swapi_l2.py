"""SWAPI L2 processing module."""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes

logger = logging.getLogger(__name__)


TIME_PER_BIN = 0.167  # seconds


def solve_full_sweep_energy(
    esa_lvl5_data: np.ndarray, esa_table_df: pd.DataFrame, lut_notes_df: pd.DataFrame
) -> npt.NDArray:
    """
    Calculate the energy of each full sweep data.

    Parameters
    ----------
    esa_lvl5_data : numpy.ndarray
        The L1 data input.
    esa_table_df : pandas.DataFrame
        The ESA unit conversion table that contains first 63 energies.
    lut_notes_df : pandas.DataFrame
        The LUT notes table that contains the last 9 fine energies.

    Returns
    -------
    energy : numpy.ndarray
        The energy of each full sweep data.
    """
    # Read 0 - 62 energy steps' fixed energy value
    fixed_energy_values = esa_table_df["Energy"].values[:63]

    # Find last 9 fine energy values of all sweeps data
    # -------------------------------------------------
    # First, verify that all values in the LUT-notes table's 'ESA DAC (Hex)' column
    # exactly matches a value in the esa_lvl5_data.
    has_exact_match = np.isin(esa_lvl5_data, lut_notes_df["ESA DAC (Hex)"].values)
    if not np.all(has_exact_match):
        raise ValueError(
            "These ESA_LVL5 values not found in lut-notes table: "
            f"{esa_lvl5_data[np.where(~has_exact_match)[0]]} "
        )

    # Find index of 71st energy step for all sweeps data in lut-notes table.
    # Tried using np.where(np.isin(...)) or df.index[np.isin(...)] to find the index
    # of each value in esa_lvl5_data within the LUT table. However, these methods
    # return only the unique matching indices — not one index per input value.
    # For example, given the input:
    #   ['12F1', '12F1', '12F1', '12F1']
    # np.where(np.isin(...)) would return:
    #   [336]
    # because it finds that '12F1' exists in the LUT and only returns its position once.
    # What we actually need is:
    #   [336, 336, 336, 336]
    # — one index for *each* occurrence in the input, preserving its shape and order.
    # Therefore, instead of relying on np.isin or similar, we explicitly use
    # np.where in a loop to find the index of each value in esa_lvl5_data individually,
    # ensuring the output array has the same shape as the input.

    last_energy_step_indices = np.array(
        [
            np.where(lut_notes_df["ESA DAC (Hex)"].values == val)[0][0]
            for val in esa_lvl5_data
        ]
    )
    # Use back tracking steps to find all 9 fine energy value indices
    # Eg. [0, -4, -8, ..., -28, -32]
    steps = np.arange(9) * -4

    # Find indices of last 9 fine energy values of all sweeps data
    fine_energy_indices = last_energy_step_indices[:, None] + steps

    # NOTE: Per SWAPI instruction, set every index that result in negative
    # indices during back tracking to zero index. SWAPI calls this
    # "flooring" the index. For example, if the 71st energy step index results
    # in less than 32, then it would result in some negative indices. Eg.
    #    71st index = 31
    #    nine fine energy indices = [31, 27, 23, 19, 15, 11, 7, 3, -1]
    #    flooring = [31, 27, 23, 19, 15, 11, 7, 3, 0]
    fine_energy_indices[fine_energy_indices < 0] = 0

    energy_values = lut_notes_df["Energy"].values[fine_energy_indices]

    # In above steps, we were calculating energy for these energy steps
    # in this order:
    #   [72, 71, 70, 69, 68, 67, 66, 65, 64]
    # Now, we need to reverse the order of these energy steps to match the
    # order it should be in:
    #  [64, 65, 66, 67, 68, 69, 70, 71, 72]
    energy_values = np.flip(energy_values, axis=1)
    # Expand to match the number of rows in energy_values
    first_63_values = np.tile(
        fixed_energy_values, (energy_values.shape[0], 1)
    )  # (epoch, 63)

    # Append the first_63_values in front of energy_values
    sweeps_energy_value = np.hstack((first_63_values, energy_values))

    return sweeps_energy_value


def swapi_l2(
    l1_dataset: xr.Dataset,
    esa_table_df: pd.DataFrame,
    lut_notes_df: pd.DataFrame,
) -> xr.Dataset:
    """
    Produce science data to L2.

    To process science data to L2, we need to:
    - convert counts to rates. This is done by dividing the counts by the
        TIME_PER_BIN time. TIME_PER_BIN is the exposure time per energy bin which is
        obtained by dividing the time for one complete sweep
        (12 s, coarse + fine sweep) by the total energy steps (72),
        i.e., TIME_PER_BIN = 12/72 = 0.167 s. This will be constant.

    - update uncertainty. Calculate new uncertainty value using
        SWP_PCEM_ERR data from level one and divide by TIME_PER_BIN. Eg.
            SWP_PCEM_UNC = SWP_PCEM_ERR / TIME_PER_BIN
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
    esa_lvl5_hex = np.vectorize(lambda x: format(x, "X"))(l1_dataset["esa_lvl5"].values)
    esa_energy = solve_full_sweep_energy(
        esa_lvl5_hex,
        esa_table_df=esa_table_df,
        lut_notes_df=lut_notes_df,
    )

    l2_dataset["swp_esa_energy"] = xr.DataArray(
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
    l2_dataset["swp_pcem_rate"] = l1_dataset["swp_pcem_counts"] / TIME_PER_BIN
    l2_dataset["swp_scem_rate"] = l1_dataset["swp_scem_counts"] / TIME_PER_BIN
    l2_dataset["swp_coin_rate"] = l1_dataset["swp_coin_counts"] / TIME_PER_BIN
    # update attrs
    l2_dataset["swp_pcem_rate"].attrs = cdf_manager.get_variable_attributes("pcem_rate")
    l2_dataset["swp_scem_rate"].attrs = cdf_manager.get_variable_attributes("scem_rate")
    l2_dataset["swp_coin_rate"].attrs = cdf_manager.get_variable_attributes("coin_rate")

    # update uncertainty
    l2_dataset["swp_pcem_rate_stat_uncert_plus"] = (
        l1_dataset["swp_pcem_counts_stat_uncert_plus"] / TIME_PER_BIN
    )
    l2_dataset["swp_pcem_rate_stat_uncert_minus"] = (
        l1_dataset["swp_pcem_counts_stat_uncert_minus"] / TIME_PER_BIN
    )
    l2_dataset["swp_scem_rate_stat_uncert_plus"] = (
        l1_dataset["swp_scem_counts_stat_uncert_plus"] / TIME_PER_BIN
    )
    l2_dataset["swp_scem_rate_stat_uncert_minus"] = (
        l1_dataset["swp_scem_counts_stat_uncert_minus"] / TIME_PER_BIN
    )
    l2_dataset["swp_coin_rate_stat_uncert_plus"] = (
        l1_dataset["swp_coin_counts_stat_uncert_plus"] / TIME_PER_BIN
    )
    l2_dataset["swp_coin_rate_stat_uncert_minus"] = (
        l1_dataset["swp_coin_counts_stat_uncert_minus"] / TIME_PER_BIN
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
