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

logger = logging.getLogger(__name__)


def calc_start_time(
    shcoarse_time: float,
    launch_time: Optional[np.datetime64] = imap_processing.launch_time,
) -> np.datetime64:
    """Calculate the datetime64 from the CCSDS secondary header information.

    Since all instrument has SHCOARSE or MET seconds, we need convert it to
    UTC. Took this from IDEX code.

    Parameters
    ----------
    shcoarse_time: float
        Number of seconds since epoch (nominally the launch time)
    launch_time : np.datetime64
        The time of launch to use as the baseline

    Returns
    -------
    np.datetime64
        The time of the event

    TODO - move this into imap-data-access? How should it be used?
    -----
    This conversion is temporary for now, and will need SPICE in the future.
    Nick Dutton mentioned that s/c clock start epoch is
        jan-1-2010-00:01:06.184 ET
    We will use this for now.
    """
    # Get the datetime of Jan 1 2010 as the start date
    time_delta = np.timedelta64(int(shcoarse_time * 1e9), "ns")
    return launch_time + time_delta


def load_cdf(
    file_path: Path, remove_xarray_attrs: bool = True, **kwargs: dict
) -> xr.Dataset:
    """Load the contents of a CDF file into an ``xarray`` dataset.

    Parameters
    ----------
    file_path : Path
        The path to the CDF file
    remove_xarray_attrs: bool
        Whether to remove the xarray attributes that get injected by the
        cdf_to_xarray function from the output xarray.Dataset. Default is True.
    **kwargs : dict, optional
        Keyword arguments for ``cdf_to_xarray``. This function overrides the
        ``cdf_to_xarray`` default keyword value `to_datetime=False` with
        ``to_datetime=True`.

    Returns
    -------
    dataset : xarray.Dataset
        The ``xarray`` dataset for the CDF file
    """
    # TODO: remove this when cdflib is updated to version >1.3.0
    if "to_datetime" not in kwargs:
        kwargs["to_datetime"] = True
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


def write_cdf(dataset: xr.Dataset):
    """Write the contents of "data" to a CDF file using cdflib.xarray_to_cdf.

    This function determines the file name to use from the global attributes,
    fills in the final attributes, and converts the whole dataset to a CDF.
    The date in the file name is determined by the time of the first epoch in the
    xarray Dataset.  The first 3 file name fields (mission, instrument, level) are
    determined by the "Logical_source" attribute.  The version is determiend from
    "Data_version".

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset object to convert to a CDF

    Returns
    -------
    file_path: pathlib.Path
        Path to the file created
    """
    # Create the filename from the global attributes
    # Logical_source looks like "imap_swe_l2_counts-1min"
    instrument, data_level, descriptor = dataset.attrs["Logical_source"].split("_")[1:]
    start_time = np.datetime_as_string(dataset["epoch"].values[0], unit="D").replace(
        "-", ""
    )

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
    file_path = science_file.construct_path()
    if not file_path.parent.exists():
        logger.info(
            "The directory does not exist, creating directory %s", file_path.parent
        )
        file_path.parent.mkdir(parents=True)
    # Insert the final attribute:
    # The Logical_file_id is always the name of the file without the extension
    dataset.attrs["Logical_file_id"] = file_path.stem
    # Add the processing version to the dataset attributes
    dataset.attrs["ground_software_version"] = imap_processing.__version__

    # Convert the xarray object to a CDF
    xarray_to_cdf(
        dataset,
        str(file_path),
        datetime64_to_cdftt2000=True,
        terminate_on_warning=True,
    )  # Terminate if not ISTP compliant

    return file_path
