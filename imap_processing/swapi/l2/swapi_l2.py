"""SWAPI L2 processing module."""

import logging

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes

logger = logging.getLogger(__name__)


TIME_PER_BIN = 0.167  # seconds


def swapi_l2(l1_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
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
    data_version : str
        Version of the data product being created.

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
        "energy",
        "energy_label",
        "swp_l1a_flags",
        "sweep_table",
        "plan_id",
        "lut_choice",
        "fpga_type",
        "fpga_rev",
    ]
    l2_dataset = l1_dataset[l1_data_keys]

    # Update L2 specific attributes
    l2_dataset.attrs["Data_version"] = data_version
    l2_global_attrs = cdf_manager.get_global_attributes("imap_swapi_l2_sci")
    l2_dataset.attrs["Data_level"] = l2_global_attrs["Data_level"]
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
    l2_dataset["swp_pcem_rate_err_plus"] = (
        l1_dataset["swp_pcem_counts_err_plus"] / TIME_PER_BIN
    )
    l2_dataset["swp_pcem_rate_err_minus"] = (
        l1_dataset["swp_pcem_counts_err_minus"] / TIME_PER_BIN
    )
    l2_dataset["swp_scem_rate_err_plus"] = (
        l1_dataset["swp_scem_counts_err_plus"] / TIME_PER_BIN
    )
    l2_dataset["swp_scem_rate_err_minus"] = (
        l1_dataset["swp_scem_counts_err_minus"] / TIME_PER_BIN
    )
    l2_dataset["swp_coin_rate_err_plus"] = (
        l1_dataset["swp_coin_counts_err_plus"] / TIME_PER_BIN
    )
    l2_dataset["swp_coin_rate_err_minus"] = (
        l1_dataset["swp_coin_counts_err_minus"] / TIME_PER_BIN
    )
    # update attrs
    l2_dataset["swp_pcem_rate_err_plus"].attrs = cdf_manager.get_variable_attributes(
        "pcem_uncertainty"
    )
    l2_dataset["swp_pcem_rate_err_minus"].attrs = cdf_manager.get_variable_attributes(
        "pcem_uncertainty"
    )
    l2_dataset["swp_scem_rate_err_plus"].attrs = cdf_manager.get_variable_attributes(
        "scem_uncertainty"
    )
    l2_dataset["swp_scem_rate_err_minus"].attrs = cdf_manager.get_variable_attributes(
        "scem_uncertainty"
    )
    l2_dataset["swp_coin_rate_err_plus"].attrs = cdf_manager.get_variable_attributes(
        "coin_uncertainty"
    )
    l2_dataset["swp_coin_rate_err_minus"].attrs = cdf_manager.get_variable_attributes(
        "coin_uncertainty"
    )

    # TODO: add thruster firing flag
    # TODO: add other flags
    logger.info("SWAPI L2 processing complete")

    return l2_dataset
