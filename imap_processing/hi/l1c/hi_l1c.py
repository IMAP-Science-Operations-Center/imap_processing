"""IMAP-HI l1c processing module."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import parse_filename_like
from imap_processing.hi.utils import create_dataset_variables, full_dataarray
from imap_processing.spice.geometry import (
    SpiceFrame,
    frame_transform,
    frame_transform_az_el,
)
from imap_processing.spice.time import ttj2000ns_to_et

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
    if len(dependencies) == 2 and isinstance(dependencies[0], xr.Dataset):
        l1c_dataset = generate_pset_dataset(*dependencies)
    else:
        raise NotImplementedError(
            "Input dependencies not recognized for l1c pset processing."
        )

    # TODO: revisit this
    l1c_dataset.attrs["Data_version"] = data_version
    return l1c_dataset


def generate_pset_dataset(
    de_dataset: xr.Dataset, calibration_prod_config_path: Path
) -> xr.Dataset:
    """
    Generate IMAP-Hi l1c pset xarray dataset from l1b product.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        IMAP-Hi l1b de product.
    calibration_prod_config_path : Path
        Calibration product configuration file.

    Returns
    -------
    pset_dataset : xarray.Dataset
        Ready to be written to CDF.
    """
    logger.info(
        f"Generating IMAP-Hi l1c pset dataset for product "
        f"{de_dataset.attrs['Logical_file_id']}"
    )
    logical_source_parts = parse_filename_like(de_dataset.attrs["Logical_source"])
    n_esa_step = len(np.unique(de_dataset.esa_step.data))
    # read calibration product configuration file
    calibration_prod_config = CalibrationProductConfig.from_yaml(
        calibration_prod_config_path
    )

    pset_dataset = empty_pset_dataset(
        n_esa_step,
        calibration_prod_config.number_of_products,
        logical_source_parts["sensor"],
    )
    # For ISTP, epoch should be the center of the time bin.
    pset_dataset.epoch.data[0] = np.mean(de_dataset.epoch.data[[0, -1]]).astype(
        np.int64
    )
    pset_et = ttj2000ns_to_et(pset_dataset.epoch.data[0])
    # Calculate and add despun_z, hae_latitude, and hae_longitude variables to
    # the pset_dataset
    pset_dataset.update(pset_geometry(pset_et, logical_source_parts["sensor"]))

    # TODO: The following section will go away as PSET algorithms to populate
    #    these variables are written.
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)
    for var_name in [
        "counts",
        "exposure_times",
        "background_rates",
        "background_rates_uncertainty",
    ]:
        pset_dataset[var_name] = full_dataarray(
            var_name,
            attr_mgr.get_variable_attributes(f"hi_pset_{var_name}", check_schema=False),
            pset_dataset.coords,
        )

    return pset_dataset


def empty_pset_dataset(
    n_esa_steps: int, n_cal_prods: int, sensor_str: str
) -> xr.Dataset:
    """
    Allocate an empty xarray.Dataset with appropriate pset coordinates.

    Parameters
    ----------
    n_esa_steps : int
        Number of Electrostatic Analyzer steps to allocate.
    n_cal_prods : int
        Number of calibration products to allocate.
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
    epoch_attrs = attr_mgr.get_variable_attributes("epoch")
    epoch_attrs.update(
        attr_mgr.get_variable_attributes("hi_pset_epoch", check_schema=False)
    )
    coords["epoch"] = xr.DataArray(
        np.empty(1, dtype=np.int64),  # TODO: get dtype from cdf attrs?
        name="epoch",
        dims=["epoch"],
        attrs=epoch_attrs,
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

    attrs = attr_mgr.get_variable_attributes(
        "hi_pset_calibration_prod", check_schema=False
    ).copy()
    dtype = attrs.pop("dtype")
    coords["calibration_prod"] = xr.DataArray(
        np.arange(n_cal_prods, dtype=dtype),
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

    # Allocate the coordinate label variables
    data_vars = dict()
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


def pset_geometry(pset_et: float, sensor_str: str) -> dict[str, xr.DataArray]:
    """
    Calculate PSET geometry variables.

    Parameters
    ----------
    pset_et : float
        Pointing set ephemeris time for which to calculate PSET geometry.
    sensor_str : str
        '45sensor' or '90sensor'.

    Returns
    -------
    geometry_vars : dict[str, xarray.DataArray]
        Keys are variable names and values are data arrays.
    """
    geometry_vars = create_dataset_variables(
        ["despun_z"], (1, 3), att_manager_lookup_str="hi_pset_{0}"
    )
    despun_z = frame_transform(
        pset_et,
        np.array([0, 0, 1]),
        SpiceFrame.IMAP_DPS,
        SpiceFrame.ECLIPJ2000,
    )
    geometry_vars["despun_z"].values = despun_z[np.newaxis, :].astype(np.float32)

    # Calculate hae_latitude and hae_longitude of the spin bins
    # define the azimuth/elevation coordinates in the Pointing Frame (DPS)
    # TODO: get the sensor's true elevation using SPICE?
    el = 0 if "90" in sensor_str else -45
    dps_az_el = np.array(
        [
            np.arange(0.05, 360, 0.1),
            np.full(3600, el),
        ]
    ).T
    hae_az_el = frame_transform_az_el(
        pset_et, dps_az_el, SpiceFrame.IMAP_DPS, SpiceFrame.ECLIPJ2000, degrees=True
    )

    geometry_vars.update(
        create_dataset_variables(
            ["hae_latitude", "hae_longitude"],
            (1, 3600),
            att_manager_lookup_str="hi_pset_{0}",
        )
    )
    geometry_vars["hae_longitude"].values = hae_az_el[:, 0].astype(np.float32)[
        np.newaxis, :
    ]
    geometry_vars["hae_latitude"].values = hae_az_el[:, 1].astype(np.float32)[
        np.newaxis, :
    ]
    return geometry_vars


class CalibrationProductConfig(list):
    """
    A Class for Hi Calibration Product Configuration.

    Parameters
    ----------
    iterable : Iterable
        An iterable containing time varying calibration product configurations.
    """

    # Define required top level keys as a class attribute
    required_top_level_keys = (
        "description",
        "valid_date",
        "product_list",
    )
    # Define required calibration product configuration keys as a class attribute
    required_cal_prod_keys = (
        "index",
        "coincidence_type_bin_strings",
        "tof_ab_range",
        "tof_ac1_range",
        "tof_bc1_range",
        "tof_c1c2_range",
    )

    def __init__(self, iterable: Iterable) -> None:
        """
        Instantiate the configuration list and validate the configuration.

        Parameters
        ----------
        iterable : Iterable
            An iterable containing time varying calibration product configurations.
        """
        super().__init__(iterable)
        self._validate()

    def _validate(self) -> None:
        """
        Validate the current configuration.

        Raises
        ------
        KeyError : Raised when a required key is not present in the current
            configuration.
        """
        for entry_index, entry in enumerate(self):
            for k1 in self.required_top_level_keys:
                if k1 not in entry:
                    raise KeyError(f"Missing required key {k1} in entry {entry_index}")
            for cal_prod_index, cal_prod_def in enumerate(entry["product_list"]):
                for k2 in self.required_cal_prod_keys:
                    if k2 not in cal_prod_def:
                        raise KeyError(
                            f"Missing required key {k2} in calibration product "
                            f"definition {cal_prod_index} in entry {entry_index}"
                        )
        # TODO: Validate date strings
        # TODO: Validate same number of calibration products for all dates???
        # TODO: Validate number of ESA entries in TOF range lists

    @classmethod
    def from_yaml(cls, config_path: Path) -> CalibrationProductConfig:
        """
        Instantiate an instance of the class from a YAML file.

        Parameters
        ----------
        config_path : Path
            Location of YAML configuration file.

        Returns
        -------
        config : CalibrationProductConfig
            Matches input YAML configuration.

        Raises
        ------
        KeyError : Raised when a required key is not present in the input YAML.
        """
        logger.info(
            f"Loading calibration product configuration from {config_path.name}",
        )
        with open(config_path) as f:
            try:
                config = cls(yaml.safe_load(f))
            except yaml.YAMLError as exc:
                logger.exception(exc)
            except KeyError as exc:
                logger.exception(exc)
                raise KeyError(
                    f"Invalid configuration specified in YAML file: {config_path}"
                ) from exc
        return config

    @property
    def number_of_products(self) -> int:
        """
        Get the number of calibration products in the current configuration.

        Returns
        -------
        number_of_products : int
            The maximum number of calibration products defined in the list of
            calibration product definitions.
        """
        return max([len(entry["product_list"]) for entry in self])
