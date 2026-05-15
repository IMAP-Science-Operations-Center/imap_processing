"""IMAP-Hi utils functions."""

from __future__ import annotations

import re
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy import typing as npt
from numpy.typing import NDArray

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
    DE_CLOCK_TICK_US : int
        Duration of Direct Event clock tick in microseconds. This is the time
        resolution of the Direct Event time tags. See IMAP-Hi Algorithm Document
        Section 2.2.5 Annotated Direct Events for more details.
    DE_CLOCK_TICK_S : float
        Duration of Direct Event clock tick in seconds.
        This is derived from DE_CLOCK_TICK_US.
    HALF_CLOCK_TICK_S : float
        Half of the Direct Event clock tick duration in seconds. This is derived
        from DE_CLOCK_TICK_S.
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
    STAT_FILTER_MIN_POINTINGS : int
        Minimum number of Pointings required for statistical filters.
    STAT_FILTER_0_THRESHOLD_FACTOR : float
        Multiplier for median comparison in Statistical Filter 0.
        Values exceeding threshold_factor * median are culled.
    STAT_FILTER_0_TOF_AB_LIMIT_NS : int
        Maximum abs(tof_ab) in nanoseconds for AB coincidences in Filter 0.
    STAT_FILTER_1_CONSECUTIVE_SIGMA : float
        Sigma multiplier for consecutive interval check in Filter 1.
    STAT_FILTER_1_EXTREME_SIGMA : float
        Sigma multiplier for extreme outlier check in Filter 1.
    STAT_FILTER_1_MIN_CONSECUTIVE : int
        Minimum consecutive intervals above threshold in Filter 1.
    STAT_FILTER_2_MIN_EVENTS : int
        Minimum events to form a pulse cluster in Filter 2.
    STAT_FILTER_2_MAX_TIME_DELTA : float
        Maximum time span in seconds for events to be considered clustered
        in Filter 2.
    STAT_FILTER_2_BIN_PADDING : int
        Number of bins to add on each side of pulse angle range in Filter 2.
    """

    # TODO: read DE_CLOCK_TICK_US from
    # instrument status summary later. This value
    # is rarely change but want to be able to change
    # it if needed. It stores information about how
    # fast the time was ticking. It is in microseconds.
    DE_CLOCK_TICK_US = 1999
    DE_CLOCK_TICK_S = DE_CLOCK_TICK_US / 1e6
    HALF_CLOCK_TICK_S = DE_CLOCK_TICK_S / 2

    TOF1_TICK_DUR = 1  # 1 ns
    TOF2_TICK_DUR = 1  # 1 ns
    TOF3_TICK_DUR = 0.5  # 0.5 ns

    # These values are stored in the TOF telemetry when the TOF timer
    # does not have valid data. See IMAP-Hi Algorithm Document Section
    # 2.2.5 Annotated Direct Events
    TOF1_BAD_VALUES = (511,)
    TOF2_BAD_VALUES = (511,)
    TOF3_BAD_VALUES = (1023,)

    # Statistical Filter tuning parameters
    # See IMAP-Hi Algorithm Document Section 2.3.2.3
    STAT_FILTER_MIN_POINTINGS = 4
    STAT_FILTER_0_THRESHOLD_FACTOR = 1.5
    STAT_FILTER_0_TOF_AB_LIMIT_NS = 15
    STAT_FILTER_1_CONSECUTIVE_SIGMA = 1.8
    STAT_FILTER_1_EXTREME_SIGMA = 5.0
    STAT_FILTER_1_MIN_CONSECUTIVE = 3
    STAT_FILTER_2_MIN_EVENTS = 6
    STAT_FILTER_2_MAX_TIME_DELTA = 5000 * DE_CLOCK_TICK_S
    STAT_FILTER_2_BIN_PADDING = 1


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
            {
                "start_met": pd.Series(dtype="float64"),
                "end_met": pd.Series(dtype="float64"),
                "esa_step": pd.Series(dtype="int64"),
                "esa_energy_step": pd.Series(dtype="int64"),
            }
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


class _BaseConfigAccessor:
    """
    Base class for configuration DataFrame accessors.

    Provides common functionality for validating and processing configuration
    DataFrames with coincidence types and TOF windows.

    Parameters
    ----------
    pandas_obj : pandas.DataFrame
        Object to run validation and use accessor functions on.
    """

    # Subclasses must define these
    index_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    tof_detector_pairs = ("ab", "ac1", "bc1", "c1c2")

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
            if index_name not in df.index.names:
                raise AttributeError(
                    f"Required index {index_name} not present in dataframe."
                )
        # Verify that the Dataframe has all the required columns
        for col in self.required_columns:
            if col not in df.columns:
                raise AttributeError(f"Required column {col} not present in dataframe.")

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


@pd.api.extensions.register_dataframe_accessor("cal_prod_config")
class CalibrationProductConfig(_BaseConfigAccessor):
    """Register custom accessor for calibration product configuration DataFrames."""

    index_columns = (
        "calibration_prod",
        "esa_energy_step",
    )
    required_columns = (
        "coincidence_type_list",
        *[
            f"tof_{det_pair}_{limit}"
            for det_pair in _BaseConfigAccessor.tof_detector_pairs
            for limit in ["low", "high"]
        ],
    )

    @classmethod
    def from_csv(cls, path: str | Path | IO[str]) -> pd.DataFrame:
        """
        Read calibration product configuration CSV file into a pandas.DataFrame.

        Parameters
        ----------
        path : str or pathlib.Path or file-like object
            Location of the calibration product configuration CSV file.

        Returns
        -------
        dataframe : pandas.DataFrame
            Validated calibration product configuration DataFrame with
            coincidence_type_values column added.
        """
        df = pd.read_csv(
            path,
            index_col=cls.index_columns,
            converters={"coincidence_type_list": lambda s: tuple(s.split("|"))},
            comment="#",
        )
        # Trigger the accessor to run validation and add coincidence_type_values
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


@pd.api.extensions.register_dataframe_accessor("background_config")
class BackgroundConfig(_BaseConfigAccessor):
    """Register custom accessor for background configuration DataFrames."""

    index_columns = (
        "calibration_prod",
        "background_index",
        "esa_energy_step",
    )
    # Columns that must be consistent across esa_energy_step for each
    # (calibration_prod, background_index) combination
    tof_columns = tuple(
        f"tof_{det_pair}_{limit}"
        for det_pair in _BaseConfigAccessor.tof_detector_pairs
        for limit in ["low", "high"]
    )

    required_columns = (
        "coincidence_type_list",
        *tof_columns,
        "scaling_factor",
        "uncertainty",
    )

    def _validate(self, df: pd.DataFrame) -> None:
        """
        Validate the background configuration.

        Extends base validation to verify:
        1. TOF windows and coincidence types are consistent across esa_energy_step
           for each (calibration_prod, background_index) combination.
        2. All required columns (coincidence_type_list, TOF windows, scaling_factor,
           uncertainty) are non-null for every row.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to validate.

        Raises
        ------
        AttributeError
            If required columns or index are missing.
        ValueError
            If TOF windows or coincidence types differ across ESA energy steps,
            or if any required values are null/missing.
        """
        super()._validate(df)

        # Check that all required columns have non-null values for every row
        # This catches cases where forward-fill didn't populate values
        # (e.g., missing first row in a group) or where scaling_factor/uncertainty
        # are missing for some ESA steps
        required_non_null = [
            "coincidence_type_list",
            *self.tof_columns,
            "scaling_factor",
            "uncertainty",
        ]

        for col in required_non_null:
            null_mask = df[col].isna()
            if null_mask.any():
                # Get the index values of rows with null values
                null_rows = df.index[null_mask].tolist()
                raise ValueError(
                    f"Null values found in required column '{col}' for rows: "
                    f"{null_rows}. All background configuration rows must have "
                    f"non-null values for coincidence_type_list, TOF windows, "
                    f"scaling_factor, and uncertainty."
                )

        # Columns that must be consistent across ESA steps
        consistency_columns = [*self.tof_columns, "coincidence_type_list"]

        # Group by (calibration_prod, background_index) and check consistency
        grouped = df.groupby(level=["calibration_prod", "background_index"])

        for (cal_prod, bg_idx), group in grouped:
            for col in consistency_columns:
                unique_values = group[col].unique()
                if len(unique_values) > 1:
                    raise ValueError(
                        f"Inconsistent {col} values across esa_energy_step for "
                        f"calibration_prod={cal_prod}, background_index={bg_idx}. "
                        f"Found values: {unique_values.tolist()}. "
                        f"TOF windows and coincidence types must be identical "
                        f"across all ESA energy steps."
                    )

    def get_tof_config(self) -> pd.DataFrame:
        """
        Get TOF window configuration with one row per background.

        Returns one row per (calibration_prod, background_index) combination.
        Since TOF windows are validated to be consistent across esa_energy_step,
        this returns the first row for each (calibration_prod, background_index)
        combination containing only the TOF-related columns.

        Returns
        -------
        tof_config : pandas.DataFrame
            DataFrame indexed by (calibration_prod, background_index) with
            coincidence_type_list, coincidence_type_values, and TOF window columns.
        """
        tof_cols = [
            "coincidence_type_list",
            "coincidence_type_values",
            *self.tof_columns,
        ]
        return self._obj.groupby(level=["calibration_prod", "background_index"])[
            tof_cols
        ].first()

    @classmethod
    def from_csv(cls, path: str | Path | IO[str]) -> pd.DataFrame:
        """
        Read background configuration CSV file into a pandas.DataFrame.

        TOF window columns and coincidence_type_list can be specified only on
        the first row of each (calibration_prod, background_index) group and
        will be forward-filled to subsequent rows. This reduces redundancy in
        the CSV file since these values must be identical across ESA energy
        steps.

        Parameters
        ----------
        path : str or pathlib.Path or file-like object
            Location of the background configuration CSV file.

        Returns
        -------
        dataframe : pandas.DataFrame
            Validated background configuration DataFrame with
            coincidence_type_values column added.
        """

        def parse_coincidence_list(s: str) -> tuple | None:
            """
            Parse coincidence type list, returning None for empty strings.

            Parameters
            ----------
            s : str
                Pipe-delimited string of coincidence types.

            Returns
            -------
            tuple or None
                Tuple of coincidence type strings, or None if input is empty.
            """
            if pd.isna(s) or s == "":
                return None
            return tuple(s.split("|"))

        df = pd.read_csv(
            path,
            index_col=cls.index_columns,
            converters={"coincidence_type_list": parse_coincidence_list},
            comment="#",
        )

        # Forward-fill TOF columns and coincidence_type_list within each
        # (calibration_prod, background_index) group. This allows the CSV to
        # specify these values only on the first row of each group.
        fill_columns = ["coincidence_type_list", *cls.tof_columns]
        df[fill_columns] = df.groupby(level=["calibration_prod", "background_index"])[
            fill_columns
        ].ffill()

        # Trigger the accessor to run validation and add coincidence_type_values
        _ = df.background_config.calibration_product_numbers
        return df


def get_tof_window_mask(
    de_ds: xr.Dataset,
    tof_windows: dict[str, tuple[float, float]],
    tof_fill_vals: dict[str, float],
) -> NDArray[np.bool_]:
    """
    Generate mask indicating which DEs pass TOF window checks.

    An event passes the TOF window check for a given detector pair if its TOF value
    is within the (low, high) bounds OR equals the fill value (indicating the detector
    pair was not hit).

    Parameters
    ----------
    de_ds : xarray.Dataset
        Direct Event Dataset with TOF variables (tof_ab, tof_ac1, tof_bc1, tof_c1c2).
    tof_windows : dict[str, tuple[float, float]]
        Dictionary mapping TOF field names to (low, high) tuples defining the
        acceptable window for each TOF measurement.
    tof_fill_vals : dict[str, float]
        Fill values for each TOF field - events with fill values pass the check.
        If not provided, fill value handling is disabled.

    Returns
    -------
    mask : numpy.ndarray
        Boolean mask where True = event passes all specified TOF window checks.
    """
    # Start with all True mask
    n_events = len(de_ds["event_met"]) if "event_met" in de_ds.dims else 0
    if n_events == 0:
        return np.array([], dtype=bool)

    combined_mask: np.ndarray = np.ones(n_events, dtype=bool)

    for tof_field, (low, high) in tof_windows.items():
        tof_array = de_ds[tof_field].values
        # TOF is in window if between low/high bounds OR equals fill value
        in_window = (low <= tof_array) & (tof_array <= high)
        in_window |= tof_array == tof_fill_vals[tof_field]

        combined_mask &= in_window

    return combined_mask


def filter_events_by_coincidence(
    de_ds: xr.Dataset,
    coincidence_types: Sequence[int],
) -> NDArray[np.bool_]:
    """
    Filter events by coincidence type.

    Parameters
    ----------
    de_ds : xarray.Dataset
        Direct Event Dataset with coincidence_type variable.
    coincidence_types : Sequence[int]
        Sequence of coincidence type integers to match.

    Returns
    -------
    mask : np.ndarray
        Boolean mask where True = event's coincidence_type is in the provided list.
    """
    if "coincidence_type" not in de_ds:
        raise ValueError("Dataset must have 'coincidence_type' variable")

    coincidence_array = de_ds["coincidence_type"].values
    return np.isin(coincidence_array, list(coincidence_types))


def get_bin_range_with_wrap(
    first_bin: int, last_bin: int, n_bins: int, extend_by: int
) -> np.ndarray:
    """
    Get bin range with wraparound and optional extension.

    Computes a range of bin indices from first_bin to last_bin, optionally
    extending by a padding amount on each side, with proper wraparound
    handling for circular bin structures (e.g., spin bins).

    Parameters
    ----------
    first_bin : int
        First bin index in the range.
    last_bin : int
        Last bin index in the range (may be less than first_bin if wrapping).
    n_bins : int
        Total number of bins (bins are 0 to n_bins-1).
    extend_by : int
        Number of bins to add on each side of the range.

    Returns
    -------
    bins : np.ndarray
        Array of bin indices in the range, with wrapping handled.
    """
    # Apply extension
    bot = (first_bin - extend_by) % n_bins
    top = (last_bin + extend_by) % n_bins

    # Check if we need to wrap
    if top >= bot:
        # No wrap needed
        return np.arange(bot, top + 1)
    else:
        # Wrap around: bins from bot to n_bins-1, then 0 to top
        return np.concatenate([np.arange(bot, n_bins), np.arange(0, top + 1)])


def _build_tof_fill_vals(de_ds: xr.Dataset) -> dict[str, float]:
    """
    Build TOF fill values dictionary from dataset attributes.

    Parameters
    ----------
    de_ds : xarray.Dataset
        Direct Event dataset with TOF variables containing FILLVAL attributes.

    Returns
    -------
    dict[str, float]
        Dictionary mapping TOF variable names to their fill values.
    """
    tof_fill_vals = {}
    for pair in CalibrationProductConfig.tof_detector_pairs:
        tof_var = f"tof_{pair}"
        tof_fill_vals[tof_var] = de_ds[tof_var].attrs.get("FILLVAL", np.nan)
    return tof_fill_vals


def iter_qualified_events_by_config(
    de_ds: xr.Dataset,
    cal_product_config: pd.DataFrame,
    esa_energy_steps: NDArray[np.int_],
) -> Generator[tuple[Any, Any, NDArray[np.bool_]], None, None]:
    """
    Iterate over calibration config, yielding masks for qualified events.

    For each (esa_energy_step, calibration_prod) combination in the config,
    yields a mask indicating which events qualify based on BOTH coincidence_type
    AND TOF window checks.

    Parameters
    ----------
    de_ds : xarray.Dataset
        Direct Event dataset with coincidence_type and TOF variables.
        TOF variables must have FILLVAL attribute for fill value handling.
    cal_product_config : pandas.DataFrame
        Config DataFrame with multi-index (calibration_prod, esa_energy_step).
        Must have coincidence_type_values column and TOF window columns.
    esa_energy_steps : np.ndarray
        ESA energy step for each event in de_ds.

    Yields
    ------
    esa_energy : Any
        The ESA energy step value.
    config_row : namedtuple
        The config row from itertuples() containing calibration product settings.
    qualified_mask : np.ndarray
        Boolean mask where True = event qualifies for this (esa, cal_prod).
    """
    n_events = len(de_ds["event_met"]) if "event_met" in de_ds.dims else 0

    # Build TOF fill values from dataset attributes
    tof_fill_vals = _build_tof_fill_vals(de_ds)

    for esa_energy, esa_df in cal_product_config.groupby(level="esa_energy_step"):
        # Mask for events at this ESA energy step
        esa_mask = esa_energy_steps == esa_energy if n_events > 0 else np.array([])

        for config_row in esa_df.itertuples():
            if n_events == 0 or not np.any(esa_mask):
                yield esa_energy, config_row, np.zeros(n_events, dtype=bool)
                continue

            # Apply common filtering logic
            filter_mask = _filter_events_by_config_row(de_ds, config_row, tof_fill_vals)

            yield esa_energy, config_row, esa_mask & filter_mask


def _filter_events_by_config_row(
    de_ds: xr.Dataset,
    config_row: Any,
    tof_fill_vals: dict[str, float],
) -> NDArray[np.bool_]:
    """
    Filter events by coincidence type and TOF windows for a single config row.

    Helper function to apply common filtering logic used by both
    iter_qualified_events_by_config and iter_background_events_by_config.

    Parameters
    ----------
    de_ds : xarray.Dataset
        Direct Event dataset with coincidence_type and TOF variables.
    config_row : namedtuple
        Config row from DataFrame.itertuples() containing:
        - coincidence_type_values: tuple of int coincidence types
        - tof_<pair>_low, tof_<pair>_high: TOF window bounds
    tof_fill_vals : dict[str, float]
        Dictionary mapping TOF variable names to their fill values.

    Returns
    -------
    filter_mask : numpy.ndarray
        Boolean mask where True = event matches the filter criteria.
    """
    # Check coincidence type
    coin_mask = filter_events_by_coincidence(de_ds, config_row.coincidence_type_values)

    # Build TOF windows dict from config row
    tof_windows = {
        f"tof_{pair}": (
            getattr(config_row, f"tof_{pair}_low"),
            getattr(config_row, f"tof_{pair}_high"),
        )
        for pair in CalibrationProductConfig.tof_detector_pairs
    }

    # Check TOF windows
    tof_mask = get_tof_window_mask(de_ds, tof_windows, tof_fill_vals)

    return coin_mask & tof_mask


def iter_background_events_by_config(
    de_ds: xr.Dataset,
    background_config: pd.DataFrame,
) -> Generator[tuple[Any, xr.Dataset], None, None]:
    """
    Iterate over background config, yielding filtered event datasets.

    For each (calibration_prod, background_index) combination in the config,
    yields the filtered dataset containing only events that match BOTH
    coincidence_type AND TOF window checks.

    Unlike iter_qualified_events_by_config, this does NOT filter by ESA energy
    step, as background counts are accumulated across all ESA steps.

    Parameters
    ----------
    de_ds : xarray.Dataset
        Direct Event dataset with coincidence_type and TOF variables.
        TOF variables must have FILLVAL attribute for fill value handling.
    background_config : pandas.DataFrame
        Config DataFrame with multi-index (calibration_prod, background_index).
        Must have coincidence_type_values column and TOF window columns.

    Yields
    ------
    config_row : namedtuple
        The config row from itertuples() containing background settings.
    filtered_ds : xarray.Dataset
        Filtered dataset containing only events matching the criteria.
    """
    n_events = len(de_ds["event_met"])

    # Build TOF fill values from dataset attributes
    tof_fill_vals = _build_tof_fill_vals(de_ds)

    for config_row in background_config.itertuples():
        if n_events == 0:
            # Return empty dataset
            yield config_row, de_ds.isel(event_met=slice(0, 0))
            continue

        # Apply common filtering logic
        filter_mask = _filter_events_by_config_row(de_ds, config_row, tof_fill_vals)

        # Return filtered dataset (no ESA energy filtering)
        filtered_ds = de_ds.isel(event_met=filter_mask)
        yield config_row, filtered_ds


def compute_qualified_event_mask(
    de_ds: xr.Dataset,
    cal_product_config: pd.DataFrame,
    esa_energy_steps: NDArray[np.int_],
) -> NDArray[np.bool_]:
    """
    Compute mask of events qualifying for ANY calibration product.

    An event qualifies if it passes BOTH coincidence_type AND TOF window
    checks for ANY (calibration_prod, esa_energy_step) combination in the
    configuration.

    Parameters
    ----------
    de_ds : xarray.Dataset
        Direct Event dataset with coincidence_type and TOF variables.
        TOF variables must have FILLVAL attribute for fill value handling.
    cal_product_config : pandas.DataFrame
        Config DataFrame with multi-index (calibration_prod, esa_energy_step).
        Must have coincidence_type_values column and TOF window columns.
    esa_energy_steps : np.ndarray
        ESA energy step for each event in de_ds.

    Returns
    -------
    qualified_mask : np.ndarray
        Boolean mask - True if event qualifies for at least one cal product.
    """
    n_events = len(de_ds["event_met"])
    if n_events == 0:
        return np.array([], dtype=bool)

    qualified_mask: np.ndarray = np.zeros(n_events, dtype=bool)

    for _, _, mask in iter_qualified_events_by_config(
        de_ds, cal_product_config, esa_energy_steps
    ):
        qualified_mask |= mask

    return qualified_mask
