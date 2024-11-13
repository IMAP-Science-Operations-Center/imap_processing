"""Various utility functions to support creation of CDF files."""

import logging
import re
from pathlib import Path
from typing import Optional

import imap_data_access
import numpy as np
import xarray as xr
from cdflib.xarray import cdf_to_xarray, xarray_to_cdf
from cdflib.xarray.cdf_to_xarray import ISTP_TO_XARRAY_ATTRS

import imap_processing
from imap_processing._version import __version__, __version_tuple__  # noqa: F401

logger = logging.getLogger(__name__)


# Reference start time (launch time or epoch)
# DEFAULT_EPOCH = np.datetime64("2010-01-01T00:01:06.184", "ns")
J2000_EPOCH = np.datetime64("2000-01-01T11:58:55.816", "ns")


def load_cdf(
    file_path: Path, remove_xarray_attrs: bool = True, **kwargs: dict
) -> xr.Dataset:
    """
    Load the contents of a CDF file into an ``xarray`` dataset.

    Parameters
    ----------
    file_path : Path
        The path to the CDF file.
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
    dataset = cdf_to_xarray(file_path, kwargs)

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
    dataset: xr.Dataset, parent_files: Optional[list] = None, **extra_cdf_kwargs: dict
) -> Path:
    """
    Write the contents of "data" to a CDF file using cdflib.xarray_to_cdf.

    This function determines the file name to use from the global attributes,
    fills in the final attributes, and converts the whole dataset to a CDF.
    The date in the file name is determined by the time of the first epoch in the
    xarray Dataset.  The first 3 file name fields (mission, instrument, level) are
    determined by the "Logical_source" attribute.  The version is determiend from
    "Data_version".

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset object to convert to a CDF.
    parent_files : list of Path, optional
        List of parent files that were used to make this file. These get added to
        the ``Parents`` global attribute:
        https://spdf.gsfc.nasa.gov/istp_guide/gattributes.html.
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
    dt64 = J2000_EPOCH + dataset["epoch"].values[0].astype("timedelta64[ns]")
    start_time = np.datetime_as_string(dt64, unit="D").replace("-", "")

    # Will now accept vXXX or XXX formats, as batch starter sends versions as vXXX.
    r = re.compile(r"v\d{3}")
    if (
        not isinstance(dataset.attrs["Data_version"], str)
        or r.match(dataset.attrs["Data_version"]) is None
    ):
        version = f"v{int(dataset.attrs['Data_version']):03d}"  # vXXX
    else:
        version = dataset.attrs["Data_version"]
    repointing = dataset.attrs.get("Repointing", None)
    science_file = imap_data_access.ScienceFilePath.generate_from_inputs(
        instrument=instrument,
        data_level=data_level,
        descriptor=descriptor,
        start_time=start_time,
        version=version,
        repointing=repointing,
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
    # Add any parent files to the dataset attributes
    if parent_files:
        # Include the current files if there are any and include just the filename
        # [file1.txt, file2.cdf, ...]
        dataset.attrs["Parents"] = dataset.attrs.get("Parents", []) + [
            parent_file.name for parent_file in parent_files
        ]

    # Convert the xarray object to a CDF
    xarray_to_cdf(
        dataset,
        str(file_path),
        terminate_on_warning=True,
        **extra_cdf_kwargs,
    )  # Terminate if not ISTP compliant

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
        r"(?:_v(?P<version>\d{3}))?"  # Optional version
        r"(?:\.(?P<extension>cdf|pkts))?$"  # Optional extension
    )
    match = re.match(regex_str, filename_like)
    if match is None:
        raise ValueError(
            "Filename like string did not contain required fields"
            "including mission, instrument, data_level, and descriptor."
        )
    return match
