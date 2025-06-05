"""Creates xarray based on structure of queried DynamoDB."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ialirt.utils.constants import IALIRT_KEYS


def create_xarray_from_records(records: list[dict]) -> xr.Dataset:
    """
    Create dataset from a list of records.

    Parameters
    ----------
    records : list of dict
       Output of querying DynamoDB.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset in standard format.
    """
    cdf_manager = ImapCdfAttributes()
    cdf_manager.add_instrument_global_attrs("ialirt")
    cdf_manager.add_instrument_variable_attrs("ialirt", "l1")

    instrument_keys: set[str] = set(IALIRT_KEYS)
    n = len(records)
    attrs = cdf_manager.get_variable_attributes("default_int64_attrs")
    fillval = attrs.get("FILLVAL")
    ttj2000ns_values = np.full(n, fillval, dtype=np.int64)

    # Collect all keys that start with the instrument prefixes.
    for i, record in enumerate(records):
        ttj2000ns_values[i] = record["ttj2000ns"]

    epoch = xr.DataArray(
        data=ttj2000ns_values,
        name="epoch",
        dims=["epoch"],
        attrs=cdf_manager.get_variable_attributes("epoch"),
    )
    component = xr.DataArray(
        ["x", "y", "z"],
        name="component",
        dims=["component"],
        attrs=cdf_manager.get_variable_attributes("component"),
    )

    coords = {"epoch": epoch, "component": component}
    dataset = xr.Dataset(
        coords=coords,
        attrs=cdf_manager.get_global_attributes("imap_ialirt_l1_realtime"),
    )

    # Create empty dataset for each key.
    for key in instrument_keys:
        attrs = cdf_manager.get_variable_attributes(key)
        fillval = attrs.get("FILLVAL")
        if key.startswith("mag"):
            data = np.full((n, 3), fillval, dtype=np.float32)
            dims = ["epoch", "component"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key == "swe_counterstreaming_electrons":
            data = np.full(n, fillval, dtype=np.uint8)
            dims = ["epoch"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key.startswith(("hit", "swe")):
            data = np.full(n, fillval, dtype=np.uint32)
            dims = ["epoch"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        else:
            data = np.full(n, fillval, dtype=np.float32)
            dims = ["epoch"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)

    # Populate the dataset variables
    for i, record in enumerate(records):
        for key in record.keys():
            val = record[key]
            if key.startswith("mag"):
                dataset[key].data[i] = [direction for direction in val]
            elif key in instrument_keys:
                dataset[key].data[i] = val

    return dataset
