"""Creates xarray based on structure of queried DynamoDB."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ialirt.utils.constants import IALIRT_KEYS


def create_xarray_from_records(records: list[dict]) -> xr.Dataset:  # noqa: PLR0912
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
        attrs=cdf_manager.get_variable_attributes("epoch", check_schema=False),
    )
    component = xr.DataArray(
        ["x", "y", "z"],
        name="component",
        dims=["component"],
        attrs=cdf_manager.get_variable_attributes("component", check_schema=False),
    )

    rtn_component = xr.DataArray(
        ["radial", "tangential", "normal"],
        name="RTN_component",
        dims=["RTN_component"],
        attrs=cdf_manager.get_variable_attributes("RTN_component", check_schema=False),
    )

    esa_step = xr.DataArray(
        data=np.arange(8, dtype=np.uint8),
        name="esa_step",
        dims=["esa_step"],
        attrs=cdf_manager.get_variable_attributes("esa_step", check_schema=False),
    )

    energy_ranges = xr.DataArray(
        data=np.arange(15, dtype=np.uint8),
        name="codice_hi_h_energy_ranges",
        dims=["codice_hi_h_energy_ranges"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_h_energy_ranges", check_schema=False
        ),
    )

    elevation = xr.DataArray(
        data=np.arange(4, dtype=np.uint8),
        name="codice_hi_h_elevation",
        dims=["codice_hi_h_elevation"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_h_elevation", check_schema=False
        ),
    )

    spin_angle = xr.DataArray(
        data=np.arange(4, dtype=np.uint8),
        name="codice_hi_h_spin_angle",
        dims=["codice_hi_h_spin_angle"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_h_spin_angle", check_schema=False
        ),
    )

    coords = {
        "epoch": epoch,
        "component": component,
        "RTN_component": rtn_component,
        "esa_step": esa_step,
        "codice_hi_h_energy_ranges": energy_ranges,
        "codice_hi_h_elevation": elevation,
        "codice_hi_h_spin_angle": spin_angle,
    }
    dataset = xr.Dataset(
        coords=coords,
        attrs=cdf_manager.get_global_attributes("imap_ialirt_l1_realtime"),
    )

    # Create empty dataset for each key.
    for key in instrument_keys:
        attrs = cdf_manager.get_variable_attributes(key, check_schema=False)
        fillval = attrs.get("FILLVAL")
        if key in ["mag_B_GSE", "mag_B_GSM"]:
            data = np.full((n, 3), fillval, dtype=np.float32)
            dims = ["epoch", "component"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key == "mag_B_RTN":
            data = np.full((n, 3), fillval, dtype=np.float32)
            dims = ["epoch", "RTN_component"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key.startswith("codice_hi"):
            data = np.full((n, 15, 4, 4), fillval, dtype=np.float32)
            dims = [
                "epoch",
                "codice_hi_h_energy_ranges",
                "codice_hi_h_elevation",
                "codice_hi_h_spin_angle",
            ]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key == "swe_counterstreaming_electrons":
            data = np.full(n, fillval, dtype=np.uint8)
            dims = ["epoch"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key.startswith("swe"):
            data = np.full((n, 8), fillval, dtype=np.uint32)
            dims = ["epoch", "esa_step"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key.startswith("hit"):
            data = np.full(n, fillval, dtype=np.uint32)
            dims = ["epoch"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        else:
            data = np.full(n, fillval, dtype=np.float32)
            dims = ["epoch"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)

    # Populate the dataset variables
    for i, record in enumerate(records):
        for key, val in record.items():
            if key in [
                "apid",
                "met",
                "met_in_utc",
                "ttj2000ns",
                "last_modified",
                "sc_position_GSM",
                "sc_position_GSE",
                "sc_velocity_GSM",
                "sc_velocity_GSE",
                "mag_hk_status",
                "spice_kernels",
                "instrument",
            ]:
                continue
            elif key in ["mag_B_GSE", "mag_B_GSM", "mag_B_RTN"]:
                dataset[key].data[i, :] = val
            elif key.startswith("swe_normalized_counts"):
                dataset[key].data[i, :] = val
            elif key.startswith("codice_hi"):
                dataset[key].data[i, :, :, :] = val
            else:
                dataset[key].data[i] = val

    return dataset
