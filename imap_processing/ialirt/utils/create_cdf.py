"""Creates cdf based on structure of queried DynamoDB."""

from collections import defaultdict

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes


def create_dataset_from_records(records: list[dict]) -> xr.Dataset:
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

    for record in records:
        instrument_keys.update(
            key for key in record if key.startswith(instrument_prefixes)
        )

    # Convert to columns, add fillvals, and associate with datatype.
    data_dict = defaultdict(list)
    for record in records:
        for key in instrument_keys:
            fillval = cdf_manager.get_variable_attributes(key).get("FILLVAL")
            val = record.get(key, fillval)
            if key == "swe_counterstreaming_electrons":
                val = np.uint8(val)
            elif key.startswith("hit") or key.startswith("swe"):
                val = np.uint32(val)
            elif key.startswith("mag"):
                # If not empty
                if isinstance(val, (list, tuple)):
                    val = [np.float32(direction) for direction in val]
                # If empty
                else:
                    val = [np.float32(fillval)] * 3
            else:
                val = np.float32(val)

            data_dict[key].append(val)

    # Convert to arrays
    for key in data_dict:
        data_dict[key] = np.array(data_dict[key])

    ttj2000ns_values = []
    for record in records:
        ttj2000ns_values.append(np.int64(record["ttj2000ns"]))

    epoch = xr.DataArray(
        data=np.array(ttj2000ns_values, dtype=np.int64),
        name="epoch",
        dims=["epoch"],
        attrs=cdf_manager.get_variable_attributes("epoch"),
    )
    component = xr.DataArray(
        ["vx", "vy", "vz"],
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
