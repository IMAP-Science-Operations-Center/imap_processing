"""SWAPI L2 processing module."""

import logging

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.swapi.swapi_utils import SWAPIAPID

logger = logging.getLogger(__name__)


def process_science(l1_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    Produce science data to L2.

    To process science data to L2, we need to:
    - convert counts to rates. This is done by dividing the counts by the
        t_bin time. t_bin is the exposure time per energy bin which is
        obtained by dividing the time for one complete sweep
        (12 s, coarse + fine sweep) by the total energy steps (72),
        i.e., t_bin = 12/72 = 0.167 s. This will be constant.

    - update uncertainty. Calculate new uncertainty value using
        SWP_PCEM_ERR data from level one and divide by t_bin. Eg.
            SWP_PCEM_UNC = SWP_PCEM_ERR / t_bin
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
    cdf_manager.load_variable_attributes("imap_swapi_variable_attrs.yaml")

    t_bin = 0.167  # seconds
    l2_dataset = l1_dataset.copy()
    l2_dataset.attrs = cdf_manager.get_global_attributes("imap_swapi_l2_sci")
    l2_dataset.attrs["Data_version"] = data_version
    # convert counts to rate
    l2_dataset["swp_pcem_rate"] = l1_dataset["swp_pcem_counts"] / t_bin
    l2_dataset["swp_scem_rate"] = l1_dataset["swp_scem_counts"] / t_bin
    l2_dataset["swp_coin_rate"] = l1_dataset["swp_coin_counts"] / t_bin
    # update attrs
    l2_dataset["swp_pcem_rate"].attrs = cdf_manager.get_variable_attributes("pcem_rate")
    l2_dataset["swp_scem_rate"].attrs = cdf_manager.get_variable_attributes("scem_rate")
    l2_dataset["swp_coin_rate"].attrs = cdf_manager.get_variable_attributes("coin_rate")

    # update uncertainty
    l2_dataset["swp_pcem_unc"] = l1_dataset["swp_pcem_err"] / t_bin
    l2_dataset["swp_scem_unc"] = l1_dataset["swp_scem_err"] / t_bin
    l2_dataset["swp_coin_unc"] = l1_dataset["swp_coin_err"] / t_bin
    # update attrs
    l2_dataset["swp_pcem_unc"].attrs = cdf_manager.get_variable_attributes(
        "pcem_uncertainty"
    )
    l2_dataset["swp_scem_unc"].attrs = cdf_manager.get_variable_attributes(
        "scem_uncertainty"
    )
    l2_dataset["swp_coin_unc"].attrs = cdf_manager.get_variable_attributes(
        "coin_uncertainty"
    )

    # drop the counts and err variables
    l2_dataset = l2_dataset.drop_vars(
        [
            "swp_pcem_counts",
            "swp_scem_counts",
            "swp_coin_counts",
            "swp_pcem_err",
            "swp_scem_err",
            "swp_coin_err",
        ]
    )

    # TODO: add thruster firing flag
    # TODO: add other flags
    logger.info("SWAPI L2 processing complete")

    return l2_dataset


def swapi_l2(l1_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    Generate science data to L2.

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
    apid = int(l1_dataset.attrs["Apid"])
    # Right now, we don't process any other apid besides science packet
    # to L2
    if apid != SWAPIAPID.SWP_SCI:
        raise ValueError(f"APID {apid} is not supported for L2 processing")

    return process_science(l1_dataset, data_version)
