"""IMAP-HI L1C processing module."""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.cdf_attribute_manager import CdfAttributeManager
from imap_processing.cdf.utils import load_cdf

logger = logging.getLogger(__name__)
CDF_MANAGER = CdfAttributeManager(imap_module_directory / "cdf" / "config")
CDF_MANAGER.load_global_attributes("imap_hi_global_cdf_attrs.yaml")
CDF_MANAGER.load_variable_attributes("imap_hi_variable_attrs.yaml")


def hi_l1c(l1b_de_path: Union[Path, str], data_version: str):
    """
    High level IMAP-Hi L1C processing function.

    This function will be expanded once the L1C processing is better defined. It
    will need to add inputs such as Ephemerides, Goodtimes inputs, and
    instrument status summary and will output a Pointing Set CDF as well as a
    Goodtimes list (CDF?).

    Parameters
    ----------
    l1b_de_path : pathlib.Path
        Location of l1b annotated direct event file.

    data_version : str
        Data version to write to CDF files and the Data_version CDF attribute.
        Should be in the format Vxxx

    Returns
    -------
    processed_data : xarray.Dataset
        Processed xarray dataset
    """
    logger.info("Running Hi L1C processing")

    de_dataset = load_cdf(l1b_de_path)
    sensor_str = de_dataset.attrs["Logical_source"].split("_")[-1].split("-")[0]

    # determine the number of unique esa_stepping_numbers
    # NOTE: it is possible that multiple esa_stepping_numbers will correspond
    #     to a single esa_step. These should be bookkept as unique entries in
    #     the esa_step coordinate.
    n_esa_step = np.unique(de_dataset.esa_stepping_num.data).size

    pset_dataset = allocate_pset_dataset(n_esa_step, sensor_str)

    # TODO: Stored epoch value needs to be consistent across ENA instruments.
    #    SPDF says this should be the center of the time bin, but instrument
    #    teams may disagree.
    pset_dataset.epoch.data[0] = de_dataset.epoch.data[0]

    return pset_dataset


def allocate_pset_dataset(n_esa_steps: int, sensor_str: str):
    """
    Allocate an empty xarray.Dataset.

    Parameters
    ----------
    n_esa_steps : int
        Number of Electrostatic Analyzer steps to allocate
    sensor_str : str
        '45sensor' or '90sensor'

    Returns
    -------
    xarray.Dataset
        Empty xarray.Dataset ready to be filled with data
    """
    # preallocate coordinates xr.DataArrays
    coords = dict()
    # epoch coordinate has only 1 entry for pointing set
    attrs = CDF_MANAGER.get_variable_attributes("hi_pset_epoch").copy()
    dtype = attrs.pop("dtype")
    coords["epoch"] = xr.DataArray(
        np.empty(1, dtype=dtype),
        name="epoch",
        dims=["epoch"],
        attrs=attrs,
    )
    attrs = CDF_MANAGER.get_variable_attributes("hi_pset_esa_step").copy()
    dtype = attrs.pop("dtype")
    coords["esa_step"] = xr.DataArray(
        np.full(n_esa_steps, attrs["FILLVAL"], dtype=dtype),
        name="esa_step",
        dims=["esa_step"],
        attrs=attrs,
    )
    # spin angle bins are 0.1 degree bins for full 360 degree spin
    attrs = CDF_MANAGER.get_variable_attributes("hi_pset_spin_angle_bin").copy()
    dtype = attrs.pop("dtype")
    coords["spin_angle_bin"] = xr.DataArray(
        np.arange(int(360 / 0.1), dtype=dtype),
        name="spin_angle_bin",
        dims=["spin_angle_bin"],
        attrs=attrs,
    )

    # Allocate the variables
    data_vars = dict()
    # despun_z is a 1x3 unit vector
    data_vars["despun_z"] = full_dataarray("despun_z", coords, shape=(1, 3))
    data_vars["hae_latitude"] = full_dataarray("hae_latitude", coords)
    data_vars["hae_longitude"] = full_dataarray("hae_longitude", coords)
    data_vars["counts"] = full_dataarray("counts", coords)
    data_vars["exposure_times"] = full_dataarray("exposure_times", coords)
    data_vars["background_rates"] = full_dataarray("background_rates", coords)
    data_vars["background_rates_uncertainty"] = full_dataarray(
        "background_rates_uncertainty", coords
    )

    # Generate label variables
    data_vars["esa_step_label"] = xr.DataArray(
        coords["esa_step"].values.astype(str),
        name="esa_step_label",
        dims=["esa_step"],
        attrs=CDF_MANAGER.get_variable_attributes("hi_pset_esa_step_label"),
    )
    data_vars["spin_bin_label"] = xr.DataArray(
        coords["spin_angle_bin"].values.astype(str),
        name="spin_bin_label",
        dims=["spin_angle_bin"],
        attrs=CDF_MANAGER.get_variable_attributes("hi_pset_spin_bin_label"),
    )
    data_vars["label_vector_HAE"] = xr.DataArray(
        np.array(["x HAE", "y HAE", "z HAE"], dtype=str),
        name="label_vector_HAE",
        dims=[" "],
        attrs=CDF_MANAGER.get_variable_attributes("hi_pset_label_vector_HAE"),
    )

    pset_global_attrs = CDF_MANAGER.get_global_attributes(
        "imap_hi_l1c_pset_attrs"
    ).copy()
    pset_global_attrs["Logical_source"] = pset_global_attrs["Logical_source"].format(
        sensor=sensor_str
    )
    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=pset_global_attrs)
    return dataset


def full_dataarray(name, coords: dict, shape=None):
    """
    Generate an empty xarray.DataArray with appropriate attributes.

    Data in DataArray are filled with FILLVAL defined in attributes
    retrieved from CDF_MANAGER with shape matching coordinates defined by
    dims or overridden by optional `shape` input.

    Parameters
    ----------
    name : str
        Variable name
    coords : dict
        Coordinate variables for the Dataset.
    shape : int or tuple
        Shape of ndarray data array to instantiate in the xarray.DataArray.

    Returns
    -------
    xarray.DataArray meeting input specifications
    """
    attrs = CDF_MANAGER.get_variable_attributes(f"hi_pset_{name}").copy()
    dtype = attrs.pop("dtype")

    # extract dims keyword argument from DEPEND_i attributes
    dims = [v for k, v in attrs.items() if k.startswith("DEPEND")]
    # define shape of the ndarray to generate
    if shape is None:
        shape = [coords[k].data.size for k in dims]
    if len(shape) > len(dims):
        dims.append("")

    print(f"{name=}, {shape=}, {dims=}")

    data_array = xr.DataArray(
        np.full(shape, attrs["FILLVAL"], dtype=dtype),
        name=name,
        dims=dims,
        attrs=attrs,
    )
    return data_array
