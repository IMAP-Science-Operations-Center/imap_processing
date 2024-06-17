"""Create dataset."""

from pathlib import Path

import xarray as xr

from imap_processing.cdf.cdf_attribute_manager import CdfAttributeManager


def create_dataset(data_dict, name, level, instrument_id=None):
    """
    Create xarray for L1b data.

    Parameters
    ----------
    data_dict: : dict
        L1b data dictionary.
    name: str
        Name of the dataset.
    level: str
        Level of the dataset.
    instrument_id: int
        Instrument ID.

    Returns
    -------
    dataset : xarray.Dataset
        Data in xarray format.
    """
    cdf_manager = CdfAttributeManager(Path(__file__).parents[2] / "cdf" / "config")
    cdf_manager.load_global_attributes("imap_default_global_cdf_attrs.yaml")
    cdf_manager.load_global_attributes("imap_ultra_global_cdf_attrs.yaml")
    cdf_manager.load_variable_attributes(f"imap_ultra_{level}_variable_attrs.yaml")

    epoch_time = xr.DataArray(
        data_dict["epoch"],
        name="epoch",
        dims=["epoch"],
        attrs=cdf_manager.variable_attributes["epoch"],
    )

    dataset = xr.Dataset(
        coords={"epoch": epoch_time},
        attrs=cdf_manager.get_global_attributes(name),
    )

    # Create the dataset variables
    if (
        dataset.attrs["Logical_source"]
        == f"imap_ultra_l1c_{instrument_id}sensor-histogram"
    ):
        for key in data_dict.keys():
            if key in ["epoch", "sid", "row", "column"]:
                continue
            elif key == "packetdata":
                dataset[key] = xr.DataArray(
                    data_dict[key],
                    dims=["epoch", "sid", "row", "column"],
                    attrs=cdf_manager.variable_attributes[key],
                )
            elif key == "spin":
                dataset[key] = xr.DataArray(
                    data_dict[key],
                    dims=["epoch", "sid"],
                    attrs=cdf_manager.variable_attributes[key],
                )
            else:
                dataset[key] = xr.DataArray(
                    data_dict[key],
                    dims=["epoch"],
                    attrs=cdf_manager.variable_attributes[key],
                )
    else:
        for key in data_dict.keys():
            if key == "epoch":
                continue
            dataset[key] = xr.DataArray(
                data_dict[key],
                dims=["epoch"],
                attrs=cdf_manager.variable_attributes[key],
            )

    return dataset
