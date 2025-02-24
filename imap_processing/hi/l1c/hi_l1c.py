"""IMAP-HI l1c processing module."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

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
    # read calibration product configuration file
    config_df = CalibrationProductConfig.from_csv(calibration_prod_config_path)

    pset_dataset = empty_pset_dataset(
        de_dataset.esa_energy_step.data,
        config_df.cal_prod_config.number_of_products,
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
    l1b_energy_steps: np.ndarray, n_cal_prods: int, sensor_str: str
) -> xr.Dataset:
    """
    Allocate an empty xarray.Dataset with appropriate pset coordinates.

    Parameters
    ----------
    l1b_energy_steps : np.ndarray
        The array of esa_energy_step data from the L1B DE product.
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
    # Create the esa_energy_step coordinate
    attrs = attr_mgr.get_variable_attributes(
        "hi_pset_esa_energy_step", check_schema=False
    ).copy()
    dtype = attrs.pop("dtype")
    # Find the unique, non-zero esa_energy_steps from the L1B data
    esa_energy_steps = np.array(sorted(set(l1b_energy_steps) - {0}), dtype=dtype)
    coords["esa_energy_step"] = xr.DataArray(
        esa_energy_steps,
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


@pd.api.extensions.register_dataframe_accessor("cal_prod_config")
class CalibrationProductConfig:
    """
    Register custom accessor for calibration product configuration DataFrames.

    Parameters
    ----------
    pandas_obj : pandas.DataFrame
        Object to run validation and use accessor functions on.
    """

    index_columns = (
        "cal_prod_num",
        "esa_energy_step",
    )
    required_columns = (
        "coincidence_type_list",
        "tof_ab_low",
        "tof_ab_high",
        "tof_ac1_low",
        "tof_ac1_high",
        "tof_bc1_low",
        "tof_bc1_high",
        "tof_c1c2_low",
        "tof_c1c2_high",
    )

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _validate(self, df: pd.DataFrame) -> None:
        """
        Validate the current configuration.

        Parameters
        ----------
        df : pandas.DataFrame
            Object to validate.

        Raises
        ------
        AttributeError : If the dataframe does not pass validation.
        """
        for index_name in self.index_columns:
            if index_name in df.index:
                raise AttributeError(
                    f"Required index {index_name} not present in dataframe."
                )
        # Verify that the Dataframe has all the required columns
        for col in self.required_columns:
            if col not in df.columns:
                raise AttributeError(f"Required column {col} not present in dataframe.")
        # TODO: Verify that the same ESA energy steps exist in all unique calibration
        #   product numbers

    @classmethod
    def from_csv(cls, path: Path) -> pd.DataFrame:
        """
        Read configuration CSV file into a pandas.DataFrame.

        Parameters
        ----------
        path : Path
            Location of the Calibration Product configuration CSV file.

        Returns
        -------
        dataframe : pandas.DataFrame
            Validated calibration product configuration data frame.
        """
        return pd.read_csv(
            path,
            index_col=cls.index_columns,
            converters={"coincidence_type_list": lambda s: s.split("|")},
            comment="#",
        )

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
        return len(self._obj.index.unique(level="cal_prod_num"))
