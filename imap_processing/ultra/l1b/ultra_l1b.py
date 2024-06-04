"""Calculates ULTRA L1b."""

from pathlib import Path

import xarray as xr

from imap_processing.cdf.cdf_attribute_manager import CdfAttributeManager
from imap_processing.ultra.l1b.badtimes import calculate_badtimes
from imap_processing.ultra.l1b.cullingmask import calculate_cullingmask
from imap_processing.ultra.l1b.de import calculate_de
from imap_processing.ultra.l1b.extendedspin import calculate_extendedspin


def create_dataset(data_dict, name):
    """
    Create xarray for L1b data.

    Parameters
    ----------
    data_dict: : dict
        L1b data dictionary.
    name: str
        Name of the dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Data in xarray format.
    """
    cdf_manager = CdfAttributeManager(Path(__file__).parents[2] / "cdf" / "config")
    cdf_manager.load_global_attributes("imap_default_global_cdf_attrs.yaml")
    cdf_manager.load_global_attributes("imap_ultra_global_cdf_attrs.yaml")
    cdf_manager.load_variable_attrs("imap_ultra_l1b_variable_attrs.yaml")

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

    for key in data_dict.keys():
        if key == "epoch":
            continue
        dataset[key] = xr.DataArray(
            data_dict[key],
            dims=["epoch"],
            attrs=cdf_manager.variable_attributes[key],
        )

    return dataset


def ultra_l1b(data_dict: dict):
    """
    Process ULTRA L1A data into L1B CDF files at output_filepath.

    Parameters
    ----------
    data_dict : dict
        The data itself and its dependent data.

    Returns
    -------
    output_datasets : list of xarray.Dataset
        List of xarray.Dataset
    """
    output_datasets = []
    instrument_id = 45 if "45" in next(iter(data_dict.keys())) else 90

    if f"imap_ultra_l1a_{instrument_id}sensor-rates" in data_dict:
        extendedspin_dict = calculate_extendedspin(data_dict)
        extendedspin_dataset = create_dataset(
            extendedspin_dict, f"imap_ultra_l1b_{instrument_id}sensor-extendedspin"
        )

        cullingmask_dict = calculate_cullingmask(extendedspin_dict)
        cullingmask_dataset = create_dataset(
            cullingmask_dict, f"imap_ultra_l1b_{instrument_id}sensor-cullingmask"
        )

        badtimes_dict = calculate_badtimes(extendedspin_dict)
        badtimes_dataset = create_dataset(
            badtimes_dict, f"imap_ultra_l1b_{instrument_id}sensor-badtimes"
        )

        output_datasets.extend(
            [extendedspin_dataset, cullingmask_dataset, badtimes_dataset]
        )
    elif (
        f"imap_ultra_l1a_{instrument_id}sensor-aux" in data_dict
        and f"imap_ultra_l1a_{instrument_id}sensor-de" in data_dict
    ):
        de_dict = calculate_de(data_dict)
        dataset = create_dataset(de_dict, f"imap_ultra_l1b_{instrument_id}sensor-de")
        output_datasets.append(dataset)
    else:
        raise ValueError("Data dictionary does not contain the expected keys.")

    return output_datasets
