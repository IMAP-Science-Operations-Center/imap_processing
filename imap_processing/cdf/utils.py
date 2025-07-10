"""Various utility functions to support creation of CDF files."""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path

import imap_data_access
import numpy as np
import xarray as xr
from cdflib.xarray import cdf_to_xarray, xarray_to_cdf
from cdflib.xarray.cdf_to_xarray import ISTP_TO_XARRAY_ATTRS

import imap_processing
from imap_processing._version import __version__, __version_tuple__  # noqa: F401
from imap_processing.spice.time import TTJ2000_EPOCH

logger = logging.getLogger(__name__)


def load_cdf(
    file_path: Path | str, remove_xarray_attrs: bool = True, **kwargs: dict
) -> xr.Dataset:
    """
    Load the contents of a CDF file into an ``xarray`` dataset.

    Parameters
    ----------
    file_path : Path or ImapFilePath or str
        The path to the CDF file or ImapFilePath object.
    remove_xarray_attrs : bool
        Whether to remove the xarray attributes that get injected by the
        cdf_to_xarray function from the output xarray.Dataset. Default is True.
    **kwargs : dict, optional
        Keyword arguments for ``cdf_to_xarray``.

    Returns
    -------
    dataset : xarray.Dataset
        The ``xarray`` dataset for the CDF file.
    """
    if isinstance(file_path, imap_data_access.ImapFilePath):
        file_path = file_path.construct_path()

    # By default, do not convert epoch to datetime64. This ensures that the
    # round-trip of writing and then loading a cdf keeps the dataset the same.
    if "to_datetime" not in kwargs:
        kwargs["to_datetime"] = False  # type: ignore
    dataset = cdf_to_xarray(file_path, **kwargs)

    # cdf_to_xarray converts single-value attributes to lists
    # convert these back to single values where applicable
    for attribute in dataset.attrs:
        value = dataset.attrs[attribute]
        if isinstance(value, list) and len(value) == 1:
            dataset.attrs[attribute] = value[0]

    # Remove attributes specific to xarray plotting from vars and coords
    # TODO: This can be removed if/when feature is added to cdf_to_xarray to
    #      make adding these attributes optional
    if remove_xarray_attrs:
        for key in dataset.variables.keys():
            for xarray_key in ISTP_TO_XARRAY_ATTRS.values():
                dataset[key].attrs.pop(xarray_key, None)

    return dataset


def write_cdf(
    dataset: xr.Dataset,
    **extra_cdf_kwargs: dict,
) -> Path:
    """
    Write the contents of "data" to a CDF file using cdflib.xarray_to_cdf.

    This function determines the file name to use from the global attributes,
    fills in the final attributes, and converts the whole dataset to a CDF.
    The date in the file name is determined by the time of the first epoch in the
    xarray Dataset.  The first 3 file name fields (mission, instrument, level) are
    determined by the "Logical_source" attribute.  The version is determined from
    "Data_version".

    The start_date and repointing attributes in the dataset are used to override the
    computed values.

    If these are not included, start_date is generated from the first epoch in the
    dataset and repointing is not included if the attribute is not present or None.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset object to convert to a CDF.
    **extra_cdf_kwargs : dict
        Additional keyword arguments to pass to the ``xarray_to_cdf`` function.

    Returns
    -------
    file_path : Path
        Path to the file created.
    """
    # Create the filename from the global attributes
    # Logical_source looks like "imap_swe_l2_counts-1min"
    instrument, data_level, descriptor = dataset.attrs["Logical_source"].split("_")[1:]
    # Convert J2000 epoch referenced data to datetime64
    # TODO: This implementation of epoch to time string results in an error of
    #       5 seconds due to 5 leap-second occurrences since the J2000 epoch.
    # TODO: Create a ttj2000_to_datetime function to handle this conversion
    start_date = dataset.attrs.get("Start_date", None)
    if start_date is None:
        # If no start time is included, then use the first epoch in the dataset
        dt64 = TTJ2000_EPOCH + dataset["epoch"].values[0].astype("timedelta64[ns]")
        start_date = np.datetime_as_string(dt64, unit="D").replace("-", "")

    version = dataset.attrs.get("Data_version", None)
    if version is None:
        warnings.warn(
            "No Data_version attribute found in dataset. Using default v999.",
            stacklevel=2,
        )
        version = "999"
        dataset.attrs["Data_version"] = version
    elif not re.match(r"\d{3}", version):
        raise ValueError(
            f"The Data_version attribute {version} does not match expected format XXX."
        )

    repointing = dataset.attrs.get("Repointing", None)

    repointing_int = int(repointing[-5:]) if repointing else None
    science_file = imap_data_access.ScienceFilePath.generate_from_inputs(
        instrument=instrument,
        data_level=data_level,
        descriptor=descriptor,
        start_time=start_date,
        version=f"v{version}",  # Ensure version is prefixed with 'v'
        repointing=repointing_int,
    )
    file_path = Path(science_file.construct_path())
    if not file_path.parent.exists():
        logger.info(
            "The directory does not exist, creating directory %s", file_path.parent
        )
        file_path.parent.mkdir(parents=True)
    # Insert the final attribute:
    # The Logical_file_id is always the name of the file without the extension
    dataset.attrs["Logical_file_id"] = file_path.stem
    # Add the processing version to the dataset attributes
    dataset.attrs["ground_software_version"] = imap_processing._version.__version__

    # Convert the xarray object to a CDF
    if "l1" in data_level:
        if "terminate_on_warning" not in extra_cdf_kwargs:
            extra_cdf_kwargs["terminate_on_warning"] = False  # type: ignore
    else:
        if "terminate_on_warning" not in extra_cdf_kwargs:
            extra_cdf_kwargs["terminate_on_warning"] = True  # type: ignore
        if "istp" not in extra_cdf_kwargs:
            extra_cdf_kwargs["istp"] = True  # type: ignore
    if "compression" not in extra_cdf_kwargs:
        extra_cdf_kwargs["compression"] = 6  # type: ignore

    xarray_to_cdf(dataset, str(file_path), **extra_cdf_kwargs)
    return file_path


def parse_filename_like(filename_like: str) -> re.Match:
    """
    Parse a filename like string.

    This function is based off of the more strict regex parsing of IMAP science
    product filenames found in the `imap_data_access` package `ScienceFilePath`
    class. This function implements a more relaxed regex that can be used on
    `Logical_source` or `Logical_file_id` found in the CDF file. The required
    components in the input string are `mission`, `instrument`, `data_level`,
    and `descriptor`.

    Parameters
    ----------
    filename_like : str
        A filename like string. This includes `Logical_source` or `Logical_file_id`
        strings.

    Returns
    -------
    match : re.Match
        A dictionary like re.Match object resulting from parsing the input string.

    Raises
    ------
    ValueError if the regex fails to match the input string.
    """
    regex_str = (
        r"^(?P<mission>imap)_"  # Required mission
        r"(?P<instrument>[^_]+)_"  # Required instrument
        r"(?P<data_level>[^_]+)_"  # Required data level
        r"((?P<sensor>\d{2}sensor)?-)?"  # Optional sensor number
        r"(?P<descriptor>[^_]+)"  # Required descriptor
        r"(_(?P<start_date>\d{8}))?"  # Optional start date
        r"(-repoint(?P<repointing>\d{5}))?"  # Optional repointing field
        r"(?:_(?P<version>v\d{3}))?"  # Optional version
        r"(?:\.(?P<extension>cdf|pkts))?$"  # Optional extension
    )
    match = re.match(regex_str, filename_like)
    if match is None:
        raise ValueError(
            "Filename like string did not contain required fields"
            "including mission, instrument, data_level, and descriptor."
        )
    return match
