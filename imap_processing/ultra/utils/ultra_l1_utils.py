"""Create dataset."""

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes


def create_dataset(  # noqa: PLR0912
    data_dict: dict,
    name: str,
    level: str,
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

    Returns
    -------
    dataset : xarray.Dataset
        Data in xarray format.
    """
    cdf_manager = ImapCdfAttributes()
    cdf_manager.add_instrument_global_attrs("ultra")
    cdf_manager.add_instrument_variable_attrs("ultra", level)

    # L1b extended spin, badtimes, and goodtimes data products
    if "spin_number" in data_dict.keys():
        coords = {
            "spin_number": ("spin_number", data_dict["spin_number"]),
            "energy_bin_geometric_mean": (
                "energy_bin_geometric_mean",
                data_dict["energy_bin_geometric_mean"],
            ),
        }
        default_dimension = "spin_number"
    # L1c pset data products
    elif "pixel_index" in data_dict:
        coords = {
            "epoch": data_dict["epoch"],
            "pixel_index": data_dict["pixel_index"],
            "energy_bin_geometric_mean": data_dict["energy_bin_geometric_mean"],
            "spin_phase_step": data_dict["spin_phase_step"],
        }
        default_dimension = "pixel_index"
    # L1b de data product
    else:
        epoch_time = xr.DataArray(
            data_dict["epoch"],
            name="epoch",
            dims=["epoch"],
            attrs=cdf_manager.get_variable_attributes("epoch", check_schema=False),
        )
        if "sensor-de" in name:
            component = xr.DataArray(
                ["vx", "vy", "vz"],
                name="component",
                dims=["component"],
                attrs=cdf_manager.get_variable_attributes(
                    "component", check_schema=False
                ),
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
        "direct_event_unit_velocity",
        "direct_event_unit_position",
    }
    rates_keys = {
        "ena_rates",
        "ena_rates_threshold",
        "quality_ena_rates",
    }
    rates_pulse_keys = {"start_per_spin", "stop_per_spin", "coin_per_spin"}

    for key, data in data_dict.items():
        if key == "epoch":
            # epoch coordinate already created with correct attrs
            continue
        elif key == "epoch_delta":
            # Create epoch_delta variable
            dataset[key] = xr.DataArray(
                data,
                dims=["epoch"],
                attrs=cdf_manager.get_variable_attributes(key, check_schema=False),
            )
        elif key in [
            "spin_number",
            "energy_bin_geometric_mean",
            "pixel_index",
            "spin_phase_step",
        ]:
            # update attrs
            dataset[key].attrs = cdf_manager.get_variable_attributes(
                key, check_schema=False
            )
        elif key in velocity_keys:
            dataset[key] = xr.DataArray(
                data,
                dims=["epoch", "component"],
                attrs=cdf_manager.get_variable_attributes(key, check_schema=False),
            )
        elif key in [
            "ena_rates_threshold",
            "scatter_threshold",
            "energy_delta_minus",
            "energy_delta_plus",
        ]:
            dataset[key] = xr.DataArray(
                data,
                dims=["energy_bin_geometric_mean"],
                attrs=cdf_manager.get_variable_attributes(key, check_schema=False),
            )
        elif key == "energy_bin_delta":
            dataset[key] = xr.DataArray(
                data,
                dims=["epoch", "energy_bin_geometric_mean"],
                attrs=cdf_manager.get_variable_attributes(key, check_schema=False),
            )
        elif key in rates_pulse_keys:
            dataset[key] = xr.DataArray(
                data,
                dims=["spin_number"],
                attrs=cdf_manager.get_variable_attributes(key, check_schema=False),
            )
        elif key in rates_keys:
            dataset[key] = xr.DataArray(
                data,
                dims=["energy_bin_geometric_mean", "spin_number"],
                attrs=cdf_manager.get_variable_attributes(key, check_schema=False),
            )
        elif key in {"quality_flags", "latitude", "longitude"}:
            dataset[key] = xr.DataArray(
                data,
                dims=["epoch", "pixel_index"],
                attrs=cdf_manager.get_variable_attributes(key, check_schema=False),
            )
        elif key in {
            "counts",
            "background_rates",
            "exposure_factor",
            "helio_exposure_factor",
        }:
            dataset[key] = xr.DataArray(
                data,
                dims=["epoch", "energy_bin_geometric_mean", "pixel_index"],
                attrs=cdf_manager.get_variable_attributes(key, check_schema=False),
            )
        elif key in {
            "geometric_function",
            "scatter_theta",
            "scatter_phi",
            "sensitivity",
            "efficiency",
        }:
            dataset[key] = xr.DataArray(
                data,
                dims=["energy_bin_geometric_mean", "pixel_index"],
                attrs=cdf_manager.get_variable_attributes(key, check_schema=False),
            )
        elif key in {
            "dead_time_ratio",
        }:
            dataset[key] = xr.DataArray(
                data,
                dims=["spin_phase_step"],
                attrs=cdf_manager.get_variable_attributes(key, check_schema=False),
            )
        else:
            dataset[key] = xr.DataArray(
                data,
                dims=[default_dimension],
                attrs=cdf_manager.get_variable_attributes(key, check_schema=False),
            )

    return dataset


def extract_data_dict(dataset: xr.Dataset) -> dict:
    """
    Convert variables and selected coordinates into a dictionary.

    Parameters
    ----------
    dataset : xr.Dataset
        The input xarray Dataset.

    Returns
    -------
    data_dict : dict
        Dictionary with data variables and selected coordinates.
    """
    data_dict = {var: dataset[var].values for var in dataset.data_vars}
    data_dict.update(
        {
            coord: dataset.coords[coord].values
            for coord in ("spin_number", "energy_bin_geometric_mean", "epoch")
            if coord in dataset.coords
        }
    )
    return data_dict
