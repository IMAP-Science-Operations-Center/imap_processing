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

    H90_APP_NHK = 818
    H90_SCI_CNT = 833
    H90_SCI_DE = 834

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
    # does not have valid data.
    TOF1_BAD_VALUES = (511, 1023)
    TOF2_BAD_VALUES = (1023,)
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
    coords : dict
        Coordinate variables for the Dataset.
    shape : int or tuple
        Shape of ndarray data array to instantiate in the xarray.DataArray.

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

    data_array = xr.DataArray(
        np.full(shape, _attrs["FILLVAL"], dtype=dtype),
        name=name,
        dims=dims,
        attrs=_attrs,
    )
    return data_array


def create_dataset_variables(
    variable_names: list[str],
    variable_shape: Union[int, Sequence[int]],
    att_manager_lookup_str: str = "{0}",
) -> dict[str, xr.DataArray]:
    """
    Instantiate new `xarray.DataArray` variables.

    Variable attributes are retrieved from CdfAttributeManager.

    Parameters
    ----------
    variable_names : list[str]
        List of variable names to create.
    variable_shape : tuple[int]
        Shape of the new variables data ndarray.
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
        new_variables[var] = full_dataarray(var, attrs, shape=variable_shape)
    return new_variables
