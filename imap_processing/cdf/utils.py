"""Various utility functions to support creation of CDF files."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from cdflib.xarray import xarray_to_cdf

import imap_processing


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

    TODO
    -----
    This conversion is temporary for now, and will need SPICE in the future.
    Nick Dutton mentioned that s/c clock start epoch is
        jan-1-2010-00:01:06.184 ET
    We will use this for now.
    """
    # Get the datetime of Jan 1 2010 as the start date
    launch_time = np.datetime64("2010-01-01T00:01:06.184")
    return launch_time + np.timedelta64(shcoarse_time, "s")


def write_cdf(
    data: xr.Dataset,
    descriptor: str,
    directory: Optional[Path] = None,
):
    """Write the contents of "data" to a CDF file using cdflib.xarray_to_cdf.

    This function determines the file name to use from the global attributes,
    fills in the the final attributes, and converts the whole dataset to a CDF.
    The date in the file name is determined by the time of the first Epoch in the
    xarray Dataset.  The first 3 file name fields (mission, instrument, level) are
    determined by the "Logical_source" attribute.  The version is determiend from
    "Data_version".

    Parameters
    ----------
        data : xarray.Dataset
            The dataset object to convert to a CDF
        descriptor : str
            The descriptor to insert into the file name after the
            orbit, before the SPICE field.  No underscores allowed.
        directory : pathlib.Path, optional
            The directory to write the file to. The default is obtained
            from the global imap_processing.config["DATA_DIR"].

    Returns
    -------
        pathlib.Path
            Path to the file created
    """
    # Determine the start date of the data in the file,
    # based on the time of the first dust impact
    file_start_date = None
    if "idex" in data.attrs["Logical_source"]:
        file_start_date = data["Epoch"][0].data
    else:
        start_time = data["Epoch"].data[0]
        file_start_date = calc_start_time(start_time)
    if file_start_date is None:
        raise ValueError(
            "Unable to determine file start date. Check Logical_source value"
        )

    date_string = np.datetime_as_string(file_start_date, unit="D").replace("-", "")

    # Determine the file name based on the attributes in the xarray
    # Set file name based on this convention:
    # imap_<instrument>_<datalevel>_<descriptor>_<startdate>_
    # <version>.cdf
    # data.attrs["Logical_source"] has the mission, instrument, and level
    # like this:
    #   imap_idex_l1
    filename = (
        f"{data.attrs['Logical_source']}"
        f"_{descriptor}"
        f"_{date_string}"
        f"_v{data.attrs['Data_version']}.cdf"
    )

    if directory is None:
        # Storage directory
        # mission/instrument/data_level/year/month/filename
        # /<directory | DATA_DIR>/<instrument>/<data_level>/<year>/<month>
        _, instrument, data_level = data.attrs["Logical_source"].split("_")
        directory = imap_processing.config["DATA_DIR"] / instrument / data_level
        directory /= date_string[:4]
        directory /= date_string[4:6]
    filename_and_path = Path(directory)
    if not filename_and_path.exists():
        logging.info(
            "The directory does not exist, creating directory %s", filename_and_path
        )
        filename_and_path.mkdir(parents=True)
    filename_and_path /= filename

    # Insert the final attribute:
    # The Logical_file_id is always the name of the file without the extension
    data.attrs["Logical_file_id"] = filename.split(".")[0]

    # Convert the xarray object to a CDF
    xarray_to_cdf(
        data,
        filename_and_path,
        datetime64_to_cdftt2000=True,
        terminate_on_warning=True,
    )  # Terminate if not ISTP compliant

    return filename_and_path
