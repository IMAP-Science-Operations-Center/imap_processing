"""Functions to support I-ALiRT CoDICE Lo processing."""

import logging
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def find_groups(data: xr.Dataset) -> xr.Dataset:
    """
    Find all occurrences of the sequential set of 233 values 0-232.

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
    subcom_range = (0, 232)

    data = data.sortby("cod_lo_acq", ascending=True)

    # Use cod_lo_counter == 0 to define the beginning of the group.
    # Find cod_lo_acq at this index and use it as the beginning time for the group.
    start_sc_ticks = data["cod_lo_acq"][(data["cod_lo_counter"] == subcom_range[0])]
    start_sc_tick = start_sc_ticks.min()
    # Use cod_lo_counter == 232 to define the end of the group.
    last_sc_ticks = data["cod_lo_acq"][
        ([data["cod_lo_counter"] == subcom_range[-1]][-1])
    ]
    last_sc_tick = last_sc_ticks.max()

    # Filter out data before the first cod_lo_counter=0 and
    # after the last cod_lo_counter=232.
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
    Append the cod_lo_## data values and create an xarray.

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
    cod_lo_data = np.stack(
        [dataset[f"cod_lo_data_{i:02}"].values for i in range(num_cod_lo_rows)], axis=1
    )

    repeated_data = {
        var: np.repeat(dataset[var].values, num_cod_lo_rows)
        for var in dataset.data_vars
        if not var.startswith("cod_lo_data_")
    }

    repeated_data["cod_lo_appended"] = cod_lo_data.flatten()
    repeated_epoch = np.repeat(dataset["epoch"].values, num_cod_lo_rows)

    appended_dataset = xr.Dataset(
        data_vars={name: ("epoch", values) for name, values in repeated_data.items()},
        coords={"epoch": repeated_epoch},
    )

    return appended_dataset


def process_codicelo(xarray_data: xr.Dataset) -> list[dict]:
    """
    Create final data products.

    Parameters
    ----------
    xarray_data : xr.Dataset
        Parsed data.

    Returns
    -------
    codicelo_data : list[dict]
        Dictionary of final data product.

    Notes
    -----
    This function is incomplete and will need to be updated to include the
    necessary calculations and data products.
    - Calculate species counts (pg 27 of Algorithm Document)
    - Calculate rates (assume 4 minutes per group)
    - Calculate L2 CoDICE pseudodensities (pg 37 of Algorithm Document)
    - Calculate the public data products
    """
    grouped_data = find_groups(xarray_data)
    unique_groups = np.unique(grouped_data["group"])
    codicelo_data: list[dict[str, Any]] = [{}]

    for group in unique_groups:
        # cod_lo_counter values for the group should be 0-232 with no duplicates.
        subcom_values = grouped_data["cod_lo_counter"][
            (grouped_data["group"] == group).values
        ]

        # Ensure no duplicates and all values from 0 to 232 are present
        if not np.array_equal(subcom_values, np.arange(233)):
            logger.warning(
                f"Group {group} does not contain all values from 0 to "
                f"232 without duplicates."
            )
            continue

        mask = grouped_data["group"] == group
        filtered_indices = np.where(mask)[0]
        group_data = grouped_data.isel(epoch=filtered_indices)

        append_cod_lo_data(group_data)

        # TODO: calculate species counts
        # TODO: calculate rates
        # TODO: calculate L2 CoDICE pseudodensities
        # TODO: calculate the public data products

    return codicelo_data
