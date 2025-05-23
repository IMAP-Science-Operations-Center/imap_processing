"""Creates cdf based on structure of queried DynamoDB."""
from collections import defaultdict
from decimal import Decimal

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes


def create_dataset_from_records(records: list[dict]) -> xr.Dataset:
    """
    Create xarray.Dataset from a list of flat dictionaries, one per record.

    Parameters
    ----------
    records : list of dict
       Output of querying DynamoDB.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset in standard format.
    """
    # Initialize CDF manager
    cdf_manager = ImapCdfAttributes()
    cdf_manager.add_instrument_global_attrs("ialirt")
    cdf_manager.add_instrument_variable_attrs("ialirt", "l1")

    INSTRUMENT_PREFIXES = ("swe", "hit", "mag", "codicelo", "codicehi", "swapi")
    instrument_keys = set()

    for record in records:
        instrument_keys.update(key for key in record if key.startswith(INSTRUMENT_PREFIXES))

    # Convert to columns, add FILLVALS, and associate with datatype.
    data_dict = defaultdict(list)
    for record in records:
        for key in instrument_keys:
            fillval = cdf_manager.get_variable_attributes(key).get("FILLVAL")
            val = record.get(key, fillval)
            if key == "swe_counterstreaming_electrons":
                val = np.uint8(val)
            elif key.startswith("hit") or key.startswith("swe"):
                val = np.uint32(val)
            else:
                val = np.float32(val)

            data_dict[key].append(val)

    # Convert to arrays
    for key in data_dict:
        data_dict[key] = np.array(data_dict[key])

    # Handle epoch coordinate
    epochs = [
        float(r["epoch"]) if isinstance(r["epoch"], Decimal) else r["epoch"]
        for r in records
    ]
    epoch_coord = xr.DataArray(
        epochs,
        name="epoch",
        dims=["epoch"],
        attrs=cdf_manager.get_variable_attributes("epoch"),
    )

    coords = {"epoch": epoch_coord}
    default_dimension = "epoch"

    # Add component dimension if any mag_ variables exist
    if any(k.startswith("mag_") for k in instrument_keys):
        component = xr.DataArray(
            ["vx", "vy", "vz"],
            name="component",
            dims=["component"],
            attrs=cdf_manager.get_variable_attributes("component"),
        )
        coords["component"] = component

    dataset = xr.Dataset(
        coords=coords, attrs=cdf_manager.get_global_attributes("ialirt")
    )

    for key in sorted(instrument_keys):
        data = data_dict[key]
        dims = ["epoch", "component"] if key.startswith("mag_") else [default_dimension]
        dataset[key] = xr.DataArray(
            data,
            dims=dims,
            attrs=cdf_manager.get_variable_attributes(key, check_schema=False),
        )

    return dataset
