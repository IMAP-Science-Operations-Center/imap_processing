"""Functions to support CoDICE Lo processing."""

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def find_groups(data: xr.Dataset) -> xr.Dataset:
    """
    Find all occurrences of the sequential set of 240 values 0-239.

    If a value is missing, or we are starting/ending
    in the middle of a sequence we do not count that as a valid group.

    Parameters
    ----------
    data : xr.Dataset
        CoDICE Lo Dataset.

    Returns
    -------
    grouped_data : xr.Dataset
        Grouped data.
    """
    subcom_range = (0, 239)

    data = data.sortby("cod_lo_acq", ascending=True)

    # Use src_seq_ctr == 0 to define the beginning of the group.
    # Find cod_lo_acq at this index and use it as the beginning time for the group.
    start_sc_ticks = data["cod_lo_acq"][(data["src_seq_ctr"] == subcom_range[0])]
    start_sc_tick = start_sc_ticks.min()
    # Use src_seq_ctr == 239 to define the end of the group.
    last_sc_ticks = data["cod_lo_acq"][([data["src_seq_ctr"] == subcom_range[-1]][-1])]
    last_sc_tick = last_sc_ticks.max()

    # Filter out data before the first src_seq_ctr=0 and after the last src_seq_ctr=239.
    grouped_data = data.where(
        (data["cod_lo_acq"] >= start_sc_tick) & (data["cod_lo_acq"] <= last_sc_tick),
        drop=True,
    )

    # Assign labels based on the cod_lo_acq times.
    group_labels = np.searchsorted(
        start_sc_ticks, grouped_data["cod_lo_acq"], side="right"
    )
    # Example:
    # grouped_data.coords
    # Coordinates:
    #   * epoch    (epoch) int64 7kB 315922822184000000 ... 315923721184000000
    #   * group    (group) int64 7kB 1 1 1 1 1 1 1 1 1 ... 15 15 15 15 15 15 15 15 15
    grouped_data["group"] = ("group", group_labels)

    return grouped_data


def append_cod_lo_data(dataset: xr.Dataset) -> xr.Dataset:
    """
    Append the cod_lo_## data values and create a xarray.

    Parameters
    ----------
    dataset : xr.Dataset
        Original dataset of group.

    Returns
    -------
    appended_dataset : xr.Dataset
        Dataset with cod_lo_## stacked.
    """
    # Number of codice lo data rows
    num_cod_lo_rows = 15
    appended_data = np.empty((0, num_cod_lo_rows))

    # Stack the cod_lo_data values into a single array.
    for ctr in dataset["src_seq_ctr"]:
        row = np.array([dataset[f"cod_lo_data_{i:02}"][int(ctr)].item() for i in range(num_cod_lo_rows)])
        appended_data = np.vstack([appended_data, row])

    # Repeat the other data values to match the number of cod_lo_data rows.
    repeated_data = {}
    for var in dataset.data_vars:
        if not var.startswith("cod_lo_data_"):
            repeated_data[var] = np.repeat(dataset[var].values, num_cod_lo_rows)
    repeated_data["cod_lo_appended"] = (("epoch",), appended_data.flatten())
    repeated_epoch = np.repeat(dataset["epoch"].values, num_cod_lo_rows)

    # Create an appended dataset.
    appended_dataset = xr.Dataset(
        data_vars={name: (("epoch",), values) for name, values in repeated_data.items()},
        coords={"epoch": repeated_epoch}
    )

    return appended_dataset


def process_codicelo(xarray_data: xr.Dataset) -> list[dict]:
    """
    Create data dictionary.

    Parameters
    ----------
    xarray_data : dict(xr.Dataset)
        Dictionary of xarray data including a single
        set for processing.

    Returns
    -------
    codicelo_data : dict
        Dictionary final data product.
    """
    grouped_data = find_groups(xarray_data)
    unique_groups = np.unique(grouped_data["group"])
    codicelo_data = {}

    for group in unique_groups:
        # Src_seq_ctr values for the group should be 0-239 with no duplicates.
        subcom_values = grouped_data["src_seq_ctr"][
            (grouped_data["group"] == group).values
        ]

        # Ensure no duplicates and all values from 0 to 239 are present
        if not np.array_equal(subcom_values, np.arange(240)):
            logger.warning(f"Group {group} does not contain all values from 0 to "
                f"239 without duplicates.")
            continue

        appended_dataset = append_cod_lo_data(grouped_data)

        # TODO: import function to calculate species counts (pg 27 of Algorithm Document)
        # TODO: calculate rates (assume 4 minutes per group)
        # TODO: import function that calculates L2 CoDICE pseudodensities (pg 37 of Algorithm Document)
        # TODO: calculate the public data products

    return codicelo_data
