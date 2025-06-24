"""IMAP-Hi utils functions."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Union

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes


class HIAPID(IntEnum):
    """Create ENUM for apid."""

    H45_APP_NHK = 754
    H45_SCI_CNT = 769
    H45_SCI_DE = 770
    H45_DIAG_FEE = 772

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
    coords: Optional[dict[str, xr.DataArray]] = None,
    shape: Optional[Union[int, Sequence[int]]] = None,
    fill_value: Optional[float] = None,
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
    variable_shape: Optional[Union[int, Sequence[int]]] = None,
    coords: Optional[dict[str, xr.DataArray]] = None,
    fill_value: Optional[float] = None,
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
