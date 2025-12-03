"""IMAP-Hi utils functions."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numpy import typing as npt

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes


class HIAPID(IntEnum):
    """Create ENUM for apid."""

    H45_MEMDMP = 740
    H45_APP_NHK = 754
    H45_SCI_CNT = 769
    H45_SCI_DE = 770
    H45_DIAG_FEE = 772

    H90_MEMDMP = 804
    H90_APP_NHK = 818
    H90_SCI_CNT = 833
    H90_SCI_DE = 834
    H90_DIAG_FEE = 836

    @property
    def sensor(self) -> str:
        """
        Define the sensor name attribute for this class.

        Returns
        -------
        str
            "45sensor" or "90sensor".
        """
        return self.name[1:3] + "sensor"


@dataclass(frozen=True)
class HiConstants:
    """
    Constants for Hi instrument.

    Attributes
    ----------
    TOF1_TICK_DUR : int
        Duration of Time-of-Flight 1 clock tick in nanoseconds.
    TOF2_TICK_DUR : int
        Duration of Time-of-Flight 2 clock tick in nanoseconds.
    TOF3_TICK_DUR : int
        Duration of Time-of-Flight 3 clock tick in nanoseconds.
    TOF1_BAD_VALUES : tuple[int]
        Tuple of values indicating TOF1 does not contain a valid time.
    TOF2_BAD_VALUES : tuple[int]
        Tuple of values indicating TOF2 does not contain a valid time.
    TOF3_BAD_VALUES : tuple[int]
        Tuple of values indicating TOF3 does not contain a valid time.
    """

    TOF1_TICK_DUR = 1  # 1 ns
    TOF2_TICK_DUR = 1  # 1 ns
    TOF3_TICK_DUR = 0.5  # 0.5 ns

    # These values are stored in the TOF telemetry when the TOF timer
    # does not have valid data. See IMAP-Hi Algorithm Document Section
    # 2.2.5 Annotated Direct Events
    TOF1_BAD_VALUES = (511,)
    TOF2_BAD_VALUES = (511,)
    TOF3_BAD_VALUES = (1023,)


def parse_sensor_number(full_string: str) -> int:
    """
    Parse the sensor number from a string.

    This function uses regex to match any portion of the input string
    containing "(45|90)sensor".

    Parameters
    ----------
    full_string : str
        A string containing sensor number.

    Returns
    -------
    sensor_number : int
      The integer sensor number. For IMAP-Hi this is 45 or 90.
    """
    regex_str = r".*(?P<sensor_num>(45|90))sensor.*?"
    match = re.match(regex_str, full_string)
    if match is None:
        raise ValueError(
            f"String 'sensor(45|90)' not found in input string: '{full_string}'"
        )
    return int(match["sensor_num"])


def full_dataarray(
    name: str,
    attrs: dict,
    coords: dict[str, xr.DataArray] | None = None,
    shape: int | Sequence[int] | None = None,
    fill_value: float | None = None,
) -> xr.DataArray:
    """
    Generate an empty xarray.DataArray with appropriate attributes.

    Data in DataArray are filled with FILLVAL defined in attributes
    retrieved from ATTR_MGR with shape matching coordinates defined by
    dims or overridden by optional `shape` input.

    Parameters
    ----------
    name : str
        Variable name.
    attrs : dict
        CDF variable attributes. Usually retrieved from ImapCdfAttributes.
    coords : dict, Optional
        Coordinate variables for the Dataset. This function will extract the
        sizes of each dimension defined by the attributes dictionary to determine
        the size of the DataArray to be created.
    shape : int or tuple, Optional
        Shape of ndarray data array to instantiate in the xarray.DataArray. If
        shape is provided, the DataArray created will have this shape regardless
        of whether coordinates are provided or not.
    fill_value : Optional, float
        Override the fill value that the DataArray will be filled with. If not
        supplied, the "FILLVAL" value from `attrs` will be used.

    Returns
    -------
    data_array : xarray.DataArray
        Meeting input specifications.
    """
    _attrs = attrs.copy()
    dtype = _attrs.pop("dtype", None)

    # extract dims keyword argument from DEPEND_i attributes
    dims = [v for k, v in sorted(_attrs.items()) if k.startswith("DEPEND")]
    # define shape of the ndarray to generate
    if shape is None:
        shape = [coords[k].data.size for k in dims]  # type: ignore
    if hasattr(shape, "__len__") and len(shape) > len(dims):
        dims.append("")
    if fill_value is None:
        fill_value = _attrs["FILLVAL"]

    data_array = xr.DataArray(
        np.full(shape, fill_value, dtype=dtype),
        name=name,
        dims=dims,
        attrs=_attrs,
    )
    return data_array


def create_dataset_variables(
    variable_names: list[str],
    variable_shape: int | Sequence[int] | None = None,
    coords: dict[str, xr.DataArray] | None = None,
    fill_value: float | None = None,
    att_manager_lookup_str: str = "{0}",
) -> dict[str, xr.DataArray]:
    """
    Instantiate new `xarray.DataArray` variables.

    Variable attributes are retrieved from CdfAttributeManager.

    Parameters
    ----------
    variable_names : list[str]
        List of variable names to create.
    variable_shape : int or Sequence of int, Optional
        Shape of the new variables data ndarray. If not provided the shape will
        attempt to be derived from the coords dictionary.
    coords : dict, Optional
        Coordinate variables for the Dataset. If `variable_shape` is not provided
        the dataset variables created will use this dictionary along with variable
        attributes from the CdfAttributeManager to determine the shapes of the
        dataset variables created.
    fill_value : Optional, float
        Value to fill the new variables data arrays with. If not supplied,
        the fill value is pulled from the CDF variable attributes "FILLVAL"
        attribute.
    att_manager_lookup_str : str
        String defining how to build the string passed to the
        CdfAttributeManager in order to retrieve the CdfAttributes for each
        variable. The string passed to CdfAttributeManager will be the result
        of calling the `str.format()` method on this input string with the
        variable name from `variable_names` as the single argument. Defaults to
        "{0}".

    Returns
    -------
    new_variables : dict[str, xarray.DataArray]
        Dictionary of new xarray.DataArray variables.
    """
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)

    new_variables = dict()
    for var in variable_names:
        attrs = attr_mgr.get_variable_attributes(
            att_manager_lookup_str.format(var), check_schema=False
        )
        new_variables[var] = full_dataarray(
            var, attrs, shape=variable_shape, coords=coords, fill_value=fill_value
        )
    return new_variables


class CoincidenceBitmap(IntEnum):
    """IntEnum class for coincidence type bitmap values."""

    A = 2**3
    B = 2**2
    C1 = 2**1
    C2 = 2**0

    @staticmethod
    def detector_hit_str_to_int(detector_hit_str: str) -> int:
        """
        Convert a detector hit string to a coincidence type integer value.

        A detector hit string is a string containing all detectors that were hit
        for a direct event. Possible detectors include: [A, B, C1, C2]. Converting
        the detector hit string to a coincidence type integer value involves
        summing the coincidence bitmap value for each detector hit. e.g. "AC1C2"
        results in 2**3 + 2**1 + 2**0 = 11.

        Parameters
        ----------
        detector_hit_str : str
            The string containing the set of detectors hit.
            e.g. "AC1C2".

        Returns
        -------
        coincidence_type : int
            The integer value of the coincidence type.
        """
        # Join all detector names with a pipe for use with regex
        pattern = r"|".join(c.name for c in CoincidenceBitmap)
        matches = re.findall(pattern, detector_hit_str)
        # Sum the integer value assigned to the detector name for each match
        return sum(CoincidenceBitmap[m] for m in matches)


class EsaEnergyStepLookupTable:
    """Class for holding a esa_step to esa_energy lookup table."""

    def __init__(self) -> None:
        self.df = pd.DataFrame(
            columns=["start_met", "end_met", "esa_step", "esa_energy_step"]
        )
        self._indexed = False

        # Get the FILLVAL from the CDF attribute manager that will be returned
        # for queries without matches
        attr_mgr = ImapCdfAttributes()
        attr_mgr.add_instrument_global_attrs("hi")
        attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)
        var_attrs = attr_mgr.get_variable_attributes(
            "hi_de_esa_energy_step", check_schema=False
        )
        self._fillval = var_attrs["FILLVAL"]
        self._esa_energy_step_dtype = var_attrs["dtype"]

    def add_entry(
        self, start_met: float, end_met: float, esa_step: int, esa_energy_step: int
    ) -> None:
        """
        Add a single entry to the lookup table.

        Parameters
        ----------
        start_met : float
            Start mission elapsed time of the time range.
        end_met : float
            End mission elapsed time of the time range.
        esa_step : int
            ESA step value.
        esa_energy_step : int
            ESA energy step value to be stored.
        """
        new_row = pd.DataFrame(
            {
                "start_met": [start_met],
                "end_met": [end_met],
                "esa_step": [esa_step],
                "esa_energy_step": [esa_energy_step],
            }
        )
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self._indexed = False

    def _ensure_indexed(self) -> None:
        """
        Create index for faster queries if not already done.

        Notes
        -----
        This method sorts the internal DataFrame by start_met and esa_step
        for improved query performance.
        """
        if not self._indexed:
            # Sort by start_met and esa_step for better query performance
            self.df = self.df.sort_values(["start_met", "esa_step"]).reset_index(
                drop=True
            )
            self._indexed = True

    def query(
        self,
        query_met: float | Iterable[float],
        esa_step: int | Iterable[float],
    ) -> float | np.ndarray:
        """
        Query MET(s) and esa_step(s) to retrieve esa_energy_step(s).

        Parameters
        ----------
        query_met : float or array_like
            Mission elapsed time value(s) to query.
            Can be a single float or array-like of floats.
        esa_step : int or array_like
            ESA step value(s) to match. Can be a single int or array-like of ints.
            Must be same type (scalar or array-like) as query_met.

        Returns
        -------
        float or numpy.ndarray
            - If inputs are scalars: returns float (esa_energy_step)
            - If inputs are array-like: returns numpy array of esa_energy_steps
              with same length as inputs.
              Contains FILLVAL for queries with no matches.

        Raises
        ------
        ValueError
            If one input is scalar and the other is array-like, or if both are
            array-like but have different lengths.

        Notes
        -----
        If multiple entries match a query, returns the first match found.
        """
        self._ensure_indexed()

        # Check if inputs are scalars
        is_scalar_met = np.isscalar(query_met)
        is_scalar_step = np.isscalar(esa_step)

        # Check for mismatched input types
        if is_scalar_met != is_scalar_step:
            raise ValueError(
                "query_met and esa_step must both be scalars or both be array-like"
            )

        # Convert to arrays for uniform processing
        query_mets = np.atleast_1d(query_met)
        esa_steps = np.atleast_1d(esa_step)

        # Ensure both arrays have the same shape
        if query_mets.shape != esa_steps.shape:
            raise ValueError(
                "query_met and esa_step must have the same "
                "length when both are array-like"
            )

        results = np.full_like(query_mets, self._fillval)

        # Lookup esa_energy_steps for queries
        for i, (qm, es) in enumerate(zip(query_mets, esa_steps, strict=False)):
            mask = (
                (self.df["start_met"] <= qm)
                & (self.df["end_met"] >= qm)
                & (self.df["esa_step"] == es)
            )

            matches = self.df[mask]
            if not matches.empty:
                results[i] = matches["esa_energy_step"].iloc[0]

        # Return scalar for scalar inputs, array for array inputs
        if is_scalar_met and is_scalar_step:
            return results.astype(self._esa_energy_step_dtype)[0]
        else:
            return results.astype(self._esa_energy_step_dtype)


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
        "calibration_prod",
        "esa_energy_step",
    )
    tof_detector_pairs = ("ab", "ac1", "bc1", "c1c2")
    required_columns = (
        "coincidence_type_list",
        *[
            f"tof_{det_pair}_{limit}"
            for det_pair in tof_detector_pairs
            for limit in ["low", "high"]
        ],
    )

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._add_coincidence_values_column()

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

    def _add_coincidence_values_column(self) -> None:
        """Generate and add the coincidence_type_values column to the dataframe."""
        # Add a column that consists of the coincidence type strings converted
        # to integer values
        self._obj["coincidence_type_values"] = self._obj.apply(
            lambda row: tuple(
                CoincidenceBitmap.detector_hit_str_to_int(entry)
                for entry in row["coincidence_type_list"]
            ),
            axis=1,
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> pd.DataFrame:
        """
        Read configuration CSV file into a pandas.DataFrame.

        Parameters
        ----------
        path : str or pathlib.Path
            Location of the Calibration Product configuration CSV file.

        Returns
        -------
        dataframe : pandas.DataFrame
            Validated calibration product configuration data frame.
        """
        df = pd.read_csv(
            path,
            index_col=cls.index_columns,
            converters={"coincidence_type_list": lambda s: tuple(s.split("|"))},
            comment="#",
        )
        # Force the _init_ method to run by using the namespace
        _ = df.cal_prod_config.number_of_products
        return df

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
        return len(self._obj.index.unique(level="calibration_prod"))

    @property
    def calibration_product_numbers(self) -> npt.NDArray[np.int_]:
        """
        Get the calibration product numbers from the current configuration.

        Returns
        -------
        cal_prod_numbers : numpy.ndarray
            Array of calibration product numbers from the configuration.
            These are sorted in ascending order and can be arbitrary integers.
        """
        return (
            self._obj.index.get_level_values("calibration_prod")
            .unique()
            .sort_values()
            .values
        )
