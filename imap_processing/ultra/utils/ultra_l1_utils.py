"""Create dataset."""

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes


def create_dataset(
    data_dict: dict, name: str, level: str, data_version: str
) -> xr.Dataset:
    """
    Create xarray for L1b data.

    Parameters
    ----------
    data_dict : dict
        L1b data dictionary.
    name : str
        Name of the dataset.
    level : str
        Level of the dataset.
    data_version : str
        Version of the data.

    Returns
    -------
    dataset : xarray.Dataset
        Data in xarray format.
    """
    cdf_manager = ImapCdfAttributes()
    cdf_manager.add_instrument_global_attrs("ultra")
    cdf_manager.add_instrument_variable_attrs("ultra", level)
    cdf_manager.add_global_attribute("Data_version", data_version)

    if "spin_number" in data_dict.keys():
        coords = {
            "spin_number": data_dict["spin_number"],
            "energy_bin_geometric_mean": data_dict["energy_bin_geometric_mean"],
            # Start time aligns with the universal spin table
            "epoch": data_dict["spin_start_time"],
        }
        default_dimension = "spin_number"

    else:
        epoch_time = xr.DataArray(
            data_dict["epoch"],
            name="epoch",
            dims=["epoch"],
            attrs=cdf_manager.get_variable_attributes("epoch"),
        )
        if "sensor-de" in name:
            component = xr.DataArray(
                ["vx", "vy", "vz"],
                name="component",
                dims=["component"],
                attrs=cdf_manager.get_variable_attributes("component"),
            )
            coords = {"epoch": epoch_time, "component": component}
        else:
            coords = {"epoch": epoch_time}
        default_dimension = "epoch"

    dataset = xr.Dataset(
        coords=coords,
        attrs=cdf_manager.get_global_attributes(name),
    )

    velocity_keys = {
        "direct_event_velocity",
        "velocity_sc",
        "velocity_dps_sc",
        "velocity_dps_helio",
    }
    rates_keys = {
        "ena_rates",
        "ena_rates_threshold",
        "quality_ena_rates",
    }

    for key in data_dict.keys():
        if key in ["epoch", "spin_number", "energy_bin_geometric_mean"]:
            continue
        elif key in velocity_keys:
            dataset[key] = xr.DataArray(
                data_dict[key],
                dims=["epoch", "component"],
                attrs=cdf_manager.get_variable_attributes(key),
            )
        elif key == "ena_rates_threshold":
            dataset[key] = xr.DataArray(
                data_dict[key],
                dims=["energy_bin_geometric_mean"],
                attrs=cdf_manager.get_variable_attributes(key),
            )
        elif key in rates_keys:
            dataset[key] = xr.DataArray(
                data_dict[key],
                dims=["energy_bin_geometric_mean", "spin_number"],
                attrs=cdf_manager.get_variable_attributes(key),
            )
        else:
            dataset[key] = xr.DataArray(
                data_dict[key],
                dims=[default_dimension],
                attrs=cdf_manager.get_variable_attributes(key),
            )

    return dataset
