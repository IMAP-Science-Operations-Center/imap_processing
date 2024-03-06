"""Various utility functions to support creation of CDF files."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from cdflib.xarray import xarray_to_cdf


def calc_start_time(shcoarse_time: int):
    """Calculate the datetime64 from the CCSDS secondary header information.

    Since all instrument has SHCOARSE or MET seconds, we need convert it to
    UTC. Took this from IDEX code.

    Parameters
    ----------
    shcoarse_time: int
        Number of seconds since epoch (nominally the launch time)

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
    launch_time = np.datetime64("2010-01-01T00:01:06.184")
    return launch_time + np.timedelta64(shcoarse_time, "s")


def write_cdf(data: xr.Dataset, filepath: Path):
    """Write the contents of "data" to a CDF file using cdflib.xarray_to_cdf.

    This function determines the file name to use from the global attributes,
    fills in the final attributes, and converts the whole dataset to a CDF.
    The date in the file name is determined by the time of the first Epoch in the
    xarray Dataset.  The first 3 file name fields (mission, instrument, level) are
    determined by the "Logical_source" attribute.  The version is determiend from
    "Data_version".

    Parameters
    ----------
        data : xarray.Dataset
            The dataset object to convert to a CDF
        filepath: Path
            The output path, including filename, to write the CDF to.

    Returns
    -------
        pathlib.Path
            Path to the file created
    """
    if not filepath.parent.exists():
        logging.info("The directory does not exist, creating directory %s", filepath)
        filepath.parent.mkdir(parents=True)

    # Insert the final attribute:
    # The Logical_file_id is always the name of the file without the extension
    data.attrs["Logical_file_id"] = filepath.stem

    # Convert the xarray object to a CDF
    xarray_to_cdf(
        data,
        str(filepath),
        datetime64_to_cdftt2000=True,
        terminate_on_warning=True,
    )  # Terminate if not ISTP compliant

    return filepath
