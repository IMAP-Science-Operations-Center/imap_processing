"""Calculates ULTRA L1c."""

from pathlib import Path

import xarray as xr

from imap_processing.cdf.cdf_attribute_manager import CdfAttributeManager
from imap_processing.cdf.utils import load_cdf
from imap_processing.ultra.l1c.histogram import calculate_histogram
from imap_processing.ultra.l1c.pset import (
    calculate_pset_backgroundrates,
    calculate_pset_counts,
    calculate_pset_exposure,
    calculate_pset_sensitivity,
)


def create_dataset(data_dict, name):
    """
    Create xarray for L1c data.

    Parameters
    ----------
    data_dict: : dict
        L1c data dictionary.
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
    cdf_manager.load_variable_attributes("imap_ultra_l1c_variable_attrs.yaml")

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


def ultra_l1c(dependencies: list[Path]):
    """
    Process ULTRA L1A and L1B data into L1C CDF files at output_filepath.

    Parameters
    ----------
    dependencies : list[Path]
        List of dependencies.

    Returns
    -------
    output_datasets : list of xarray.Dataset
        List of xarray.Dataset
    """
    data_dict = {}
    for dependency in dependencies:
        dataset = load_cdf(dependency, to_datetime=True)
        data_dict[dataset.attrs["Logical_source"]] = dataset

    output_datasets = []
    instrument_id = 45 if "45" in next(iter(data_dict.keys())) else 90

    if (
        f"imap_ultra_l1a_{instrument_id}sensor-histogram" in data_dict
        and f"imap_ultra_l1b_{instrument_id}sensor-cullingmask" in data_dict
    ):
        histogram_dict = calculate_histogram(
            data_dict[f"imap_ultra_l1a_{instrument_id}sensor-histogram"],
            data_dict[f"imap_ultra_l1b_{instrument_id}sensor-cullingmask"],
        )
        histogram_dataset = create_dataset(
            histogram_dict, f"imap_ultra_l1c_{instrument_id}sensor-histogram"
        )
        output_datasets.append(histogram_dataset)
    elif (
        f"imap_ultra_l1b_{instrument_id}sensor-cullingmask" in data_dict
        and f"imap_ultra_l1b_{instrument_id}sensor-de" in data_dict
        and f"imap_ultra_l1b_{instrument_id}extendedspin-de" in data_dict
    ):
        pset_exposure_dict = calculate_pset_exposure(data_dict)
        pset_sensitivity_dict = calculate_pset_sensitivity(data_dict)
        pset_backgroundrates_dict = calculate_pset_backgroundrates(data_dict)
        pset_counts_dict = calculate_pset_counts(data_dict)

        pset_exposure_dataset = create_dataset(
            pset_exposure_dict, f"imap_ultra_l1c_{instrument_id}sensor-pset-exposure"
        )
        pset_sensitivity_dataset = create_dataset(
            pset_sensitivity_dict,
            f"imap_ultra_l1c_{instrument_id}sensor-pset-sensitivity",
        )
        pset_backgroundrates_dataset = create_dataset(
            pset_backgroundrates_dict,
            f"imap_ultra_l1c_{instrument_id}sensor-pset-backgroundrates",
        )
        pset_counts_dataset = create_dataset(
            pset_counts_dict, f"imap_ultra_l1c_{instrument_id}sensor-pset-counts"
        )

        output_datasets.extend(
            [
                pset_exposure_dataset,
                pset_sensitivity_dataset,
                pset_backgroundrates_dataset,
                pset_counts_dataset,
            ]
        )
    else:
        raise ValueError("Data dictionary does not contain the expected keys.")

    return output_datasets
