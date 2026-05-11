"""Contains helper functions to support IDEX processing."""

import pandas as pd
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.idex.idex_constants import IDEX_10_DAY_RANGES_PATH


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
    data_vars: dict | None = None,
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


def get_10_day_window_end_date(start_date: str) -> str:
    """
    Use the start date to find the end date of the 10-day window.

    IDEX l1a data is processed in 10-day windows, so this function will be used to
    determine the end date of the window to process based on the start
    date passed into the job.

    Parameters
    ----------
    start_date : str
        Start date of the window to process.

    Returns
    -------
    end_date : str
        End date of the window to process.
    """
    # This CSV was provided by the IDEX team.
    idex_10_day_ranges = pd.read_csv(IDEX_10_DAY_RANGES_PATH, header=0, dtype=str)
    # Find the row where the input start date is equal to the start date in the df.
    matching_row = idex_10_day_ranges[idex_10_day_ranges["start_date"] == start_date]
    # if there is no match, raise an error. We expect that the start date passed into
    # the job will always be the start date of a 10-day window, so there should always
    # be a match in the csv.
    if matching_row.empty:
        raise ValueError(
            f"Start date {start_date} is not an IDEX defined start date"
            f" for a 10-day window."
        )
    if len(matching_row["end_date"]) > 1:
        raise ValueError(
            f"There should only be one row where start_date is equal "
            f"to {start_date}. Please check lookup table: "
            f"{IDEX_10_DAY_RANGES_PATH}."
        )
    return matching_row["end_date"].values[0]
