"""Creates xarray based on structure of queried DynamoDB."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes


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

    instrument_prefixes = ("swe", "hit", "mag", "codicelo", "codicehi", "swapi")
    instrument_keys: set[str] = set()

    # Collect all keys that start with the instrument prefixes.
    for record in records:
        instrument_keys.update(
            key for key in record if key.startswith(instrument_prefixes)
        )

    # Create empty dictionaries for each key.
    n = len(records)
    data_dict = {}
    for key in instrument_keys:
        attrs = cdf_manager.get_variable_attributes(key)
        fillval = attrs.get("FILLVAL")
        if key.startswith("mag"):
            data_dict[key] = np.full((n, 3), fillval, dtype=np.float32)
        elif key == "swe_counterstreaming_electrons":
            data_dict[key] = np.full(n, fillval, dtype=np.uint8)
        elif key.startswith(("hit", "swe")):
            data_dict[key] = np.full(n, fillval, dtype=np.uint32)
        else:
            data_dict[key] = np.full(n, fillval, dtype=np.float32)

    attrs = cdf_manager.get_variable_attributes("default_int64_attrs")
    fillval = attrs.get("FILLVAL")
    ttj2000ns_values = np.full(n, fillval, dtype=np.int64)

    # Populate the dictionaries.
    for i, record in enumerate(records):
        ttj2000ns_values[i] = record["ttj2000ns"]
        for key in record.keys():
            if key.startswith("mag"):
                val = record[key]
                data_dict[key][i] = [direction for direction in val]
            elif key in instrument_keys:
                data_dict[key][i] = record[key]

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

    for key in sorted(instrument_keys):
        data = data_dict[key]
        dims = ["epoch", "component"] if key.startswith("mag") else ["epoch"]
        dataset[key] = xr.DataArray(
            data,
            dims=dims,
            attrs=cdf_manager.get_variable_attributes(key, check_schema=False),
        )

    return dataset
