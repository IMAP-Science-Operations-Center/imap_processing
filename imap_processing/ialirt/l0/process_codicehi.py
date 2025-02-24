"""Functions to support I-ALiRT CoDICE Hi processing."""

import logging
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def find_groups(data: xr.Dataset) -> xr.Dataset:
    """
    Find all occurrences of the sequential set of 234 values 0-233.

    If a value is missing, or we are starting/ending
    in the middle of a sequence we do not count that as a valid group.

    Parameters
    ----------
    data : xr.Dataset
        CoDICE Hi Dataset.

    Returns
    -------
    grouped_data : xr.Dataset
        Grouped data.
    """
    subcom_range = (0, 233)

    data = data.sortby("cod_hi_acq", ascending=True)

    # Use cod_hi_counter == 0 to define the beginning of the group.
    # Find cod_hi_acq at this index and use it as the beginning time for the group.
    start_sc_ticks = data["cod_hi_acq"][(data["cod_hi_counter"] == subcom_range[0])]
    start_sc_tick = start_sc_ticks.min()
    # Use cod_hi_counter == 233 to define the end of the group.
    last_sc_ticks = data["cod_hi_acq"][
        ([data["cod_hi_counter"] == subcom_range[-1]][-1])
    ]
    last_sc_tick = last_sc_ticks.max()

    # Filter out data before the first cod_hi_counter=0 and
    # after the last cod_hi_counter=233 and cod_hi_counter values != 0-233.
    grouped_data = data.where(
        (data["cod_hi_acq"] >= start_sc_tick)
        & (data["cod_hi_acq"] <= last_sc_tick)
        & (data["cod_hi_counter"] >= subcom_range[0])
        & (data["cod_hi_counter"] <= subcom_range[-1]),
        drop=True,
    )

    # Assign labels based on the cod_hi_acq times.
    group_labels = np.searchsorted(
        start_sc_ticks, grouped_data["cod_hi_acq"], side="right"
    )
    # Example:
    # grouped_data.coords
    # Coordinates:
    #   * epoch    (epoch) int64 7kB 315922822184000000 ... 315923721184000000
    #   * group    (group) int64 7kB 1 1 1 1 1 1 1 1 1 ... 15 15 15 15 15 15 15 15 15
    grouped_data["group"] = ("group", group_labels)

    return grouped_data


def append_cod_hi_data(dataset: xr.Dataset) -> xr.Dataset:
    """
    Append the cod_hi_## data values and create an xarray.

    Parameters
    ----------
    dataset : xr.Dataset
        Original dataset of group.

    Returns
    -------
    appended_dataset : xr.Dataset
        Dataset with cod_hi_## stacked.
    """
    # Number of codice hi data rows
    num_cod_hi_rows = 5
    cod_hi_data = np.stack(
        [dataset[f"cod_hi_data_{i:02}"].values for i in range(num_cod_hi_rows)], axis=1
    )

    repeated_data = {
        var: np.repeat(dataset[var].values, num_cod_hi_rows)
        for var in dataset.data_vars
        if not var.startswith("cod_hi_data_")
    }

    repeated_data["cod_hi_appended"] = cod_hi_data.flatten()
    repeated_epoch = np.repeat(dataset["epoch"].values, num_cod_hi_rows)

    appended_dataset = xr.Dataset(
        data_vars={name: ("epoch", values) for name, values in repeated_data.items()},
        coords={"epoch": repeated_epoch},
    )

    return appended_dataset


def process_codicehi(xarray_data: xr.Dataset) -> list[dict]:
    """
    Create final data products.

    Parameters
    ----------
    xarray_data : xr.Dataset
        Parsed data.

    Returns
    -------
    codicehi_data : list[dict]
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
    codicehi_data: list[dict[str, Any]] = [{}]

    for group in unique_groups:
        # cod_hi_counter values for the group should be 0-233 with no duplicates.
        subcom_values = grouped_data["cod_hi_counter"][
            (grouped_data["group"] == group).values
        ]

        # Ensure no duplicates and all values from 0 to 233 are present
        if not np.array_equal(subcom_values, np.arange(234)):
            logger.warning(
                f"Group {group} does not contain all values from 0 to "
                f"233 without duplicates."
            )
            continue

        mask = grouped_data["group"] == group
        filtered_indices = np.where(mask)[0]
        group_data = grouped_data.isel(epoch=filtered_indices)

        append_cod_hi_data(group_data)

        # TODO: calculate species counts
        # TODO: calculate rates
        # TODO: calculate L2 CoDICE pseudodensities
        # TODO: calculate the public data products

    return codicehi_data
