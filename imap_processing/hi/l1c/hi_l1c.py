"""IMAP-HI l1c processing module."""

import logging

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import parse_filename_like
from imap_processing.hi.utils import full_dataarray

logger = logging.getLogger(__name__)


def hi_l1c(dependencies: list, data_version: str) -> xr.Dataset:
    """
    High level IMAP-Hi l1c processing function.

    This function will be expanded once the l1c processing is better defined. It
    will need to add inputs such as Ephemerides, Goodtimes inputs, and
    instrument status summary and will output a Pointing Set CDF as well as a
    Goodtimes list (CDF?).

    Parameters
    ----------
    dependencies : list
        Input dependencies needed for l1c processing.

    data_version : str
        Data version to write to CDF files and the Data_version CDF attribute.
        Should be in the format Vxxx.

    Returns
    -------
    l1c_dataset : xarray.Dataset
        Processed xarray dataset.
    """
    logger.info("Running Hi l1c processing")

    # TODO: I am not sure what the input for Goodtimes will be so for now,
    #    If the input is an xarray Dataset, do pset processing
    if len(dependencies) == 1 and isinstance(dependencies[0], xr.Dataset):
        l1c_dataset = generate_pset_dataset(dependencies[0])
    else:
        raise NotImplementedError(
            "Input dependencies not recognized for l1c pset processing."
        )

    # TODO: revisit this
    l1c_dataset.attrs["Data_version"] = data_version
    return l1c_dataset


def generate_pset_dataset(de_dataset: xr.Dataset) -> xr.Dataset:
    """
    Will process IMAP-Hi l1b product into a l1c pset xarray dataset.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        IMAP-Hi l1b de product.

    Returns
    -------
    pset_dataset : xarray.Dataset
        Ready to be written to CDF.
    """
    logical_source_parts = parse_filename_like(de_dataset.attrs["Logical_source"])
    n_esa_step = de_dataset.esa_step.data.size
    pset_dataset = allocate_pset_dataset(n_esa_step, logical_source_parts["sensor"])
    # TODO: Stored epoch value needs to be consistent across ENA instruments.
    #    SPDF says this should be the center of the time bin, but instrument
    #    teams may disagree.
    pset_dataset.epoch.data[0] = de_dataset.epoch.data[0]

    return pset_dataset


def allocate_pset_dataset(n_esa_steps: int, sensor_str: str) -> xr.Dataset:
    """
    Allocate an empty xarray.Dataset.

    Parameters
    ----------
    n_esa_steps : int
        Number of Electrostatic Analyzer steps to allocate.
    sensor_str : str
        '45sensor' or '90sensor'.

    Returns
    -------
    dataset : xarray.Dataset
        Empty xarray.Dataset ready to be filled with data.
    """
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)

    # preallocate coordinates xr.DataArrays
    coords = dict()
    # epoch coordinate has only 1 entry for pointing set
    coords["epoch"] = xr.DataArray(
        np.empty(1, dtype=np.int64),  # TODO: get dtype from cdf attrs?
        name="epoch",
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes("epoch"),
    )
    attrs = attr_mgr.get_variable_attributes(
        "hi_pset_esa_energy_step", check_schema=False
    ).copy()
    dtype = attrs.pop("dtype")
    coords["esa_energy_step"] = xr.DataArray(
        np.full(n_esa_steps, attrs["FILLVAL"], dtype=dtype),
        name="esa_energy_step",
        dims=["esa_energy_step"],
        attrs=attrs,
    )
    # TODO: define calibration product number to coincidence type mapping and
    #     use the number of calibration products here. I believe it will be 5
    #     0 for any, 1-4, for the number of detector hits.
    n_calibration_prod = 5
    attrs = attr_mgr.get_variable_attributes(
        "hi_pset_calibration_prod", check_schema=False
    ).copy()
    dtype = attrs.pop("dtype")
    coords["calibration_prod"] = xr.DataArray(
        np.arange(n_calibration_prod, dtype=dtype),
        name="calibration_prod",
        dims=["calibration_prod"],
        attrs=attrs,
    )
    # spin angle bins are 0.1 degree bins for full 360 degree spin
    attrs = attr_mgr.get_variable_attributes(
        "hi_pset_spin_angle_bin", check_schema=False
    ).copy()
    dtype = attrs.pop("dtype")
    coords["spin_angle_bin"] = xr.DataArray(
        np.arange(int(360 / 0.1), dtype=dtype),
        name="spin_angle_bin",
        dims=["spin_angle_bin"],
        attrs=attrs,
    )

    # Allocate the variables
    data_vars = dict()
    # despun_z is a 1x3 unit vector that does not have a DEPEND_1.
    # Define this dict to override the shape produced in full_dataarray
    var_shapes = {"despun_z": (1, 3)}
    for var_name in [
        "despun_z",
        "hae_latitude",
        "hae_longitude",
        "counts",
        "exposure_times",
        "background_rates",
        "background_rates_uncertainty",
    ]:
        data_vars[var_name] = full_dataarray(
            var_name,
            attr_mgr.get_variable_attributes(f"hi_pset_{var_name}", check_schema=False),
            coords,
            shape=var_shapes.get(var_name, None),
        )

    # Generate label variables
    data_vars["esa_energy_step_label"] = xr.DataArray(
        coords["esa_energy_step"].values.astype(str),
        name="esa_energy_step_label",
        dims=["esa_energy_step"],
        attrs=attr_mgr.get_variable_attributes(
            "hi_pset_esa_energy_step_label", check_schema=False
        ),
    )
    data_vars["calibration_prod_label"] = xr.DataArray(
        coords["calibration_prod"].values.astype(str),
        name="calibration_prod_label",
        dims=["calibration_prod"],
        attrs=attr_mgr.get_variable_attributes(
            "hi_pset_calibration_prod_label", check_schema=False
        ),
    )
    data_vars["spin_bin_label"] = xr.DataArray(
        coords["spin_angle_bin"].values.astype(str),
        name="spin_bin_label",
        dims=["spin_angle_bin"],
        attrs=attr_mgr.get_variable_attributes(
            "hi_pset_spin_bin_label", check_schema=False
        ),
    )
    data_vars["label_vector_HAE"] = xr.DataArray(
        np.array(["x HAE", "y HAE", "z HAE"], dtype=str),
        name="label_vector_HAE",
        dims=[" "],
        attrs=attr_mgr.get_variable_attributes(
            "hi_pset_label_vector_HAE", check_schema=False
        ),
    )

    pset_global_attrs = attr_mgr.get_global_attributes("imap_hi_l1c_pset_attrs").copy()
    pset_global_attrs["Logical_source"] = pset_global_attrs["Logical_source"].format(
        sensor=sensor_str
    )
    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=pset_global_attrs)
    return dataset
