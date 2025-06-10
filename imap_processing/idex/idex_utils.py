"""Contains helper functions to support IDEX processing."""

from typing import Optional

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes


def get_idex_attrs(data_level: str) -> ImapCdfAttributes:
    """
    Load in CDF attributes for IDEX instrument.

    Parameters
    ----------
    data_level : str
        Data level of current processing.

    Returns
    -------
    idex_attrs : ImapCdfAttributes
        The IDEX L1a CDF attributes.
    """
    idex_attrs = ImapCdfAttributes()
    idex_attrs.add_instrument_global_attrs("idex")
    idex_attrs.add_instrument_variable_attrs("idex", data_level)
    return idex_attrs


def setup_dataset(
    dataset: xr.Dataset,
    match_strings: list,
    idex_attrs: ImapCdfAttributes,
    data_vars: Optional[dict] = None,
) -> xr.Dataset:
    """
    Initialize a dataset and copy over any dataArrays.

    Parameters
    ----------
    dataset : xarray.Dataset
        Contains the arrays to copy to the new dataset. The variable named "epoch" is
        required.
    match_strings : list[str]
        Array names to copy to the new dataset.
    idex_attrs : ImapCdfAttributes
        Idex attributes for current data level processing.
    data_vars : dict
        Dictionary of variables to copy over to the new dataset.

    Returns
    -------
    new_dataset : xarray.Dataset
        Initialized dataset.
    """
    epoch_da = xr.DataArray(
        data=dataset["epoch"].data.copy(),
        name="epoch",
        dims="epoch",
        attrs=idex_attrs.get_variable_attributes("epoch", check_schema=False),
    )

    new_dataset = xr.Dataset(coords={"epoch": epoch_da}, data_vars=data_vars)

    vars_to_copy = [
        var for var in dataset.variables if any(match in var for match in match_strings)
    ]
    # Copy arrays over to the new dataset
    for var in vars_to_copy:
        new_dataset[var] = dataset[var].copy()

    return new_dataset
