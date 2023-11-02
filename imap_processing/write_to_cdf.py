import os

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


def write_to_cdf(
    data: xr.Dataset,
    instrument: str,
    level: str,
    version: str,
    directory: str,
    mode: str = "",
    description: str = "",
):
    """Write the contents of data to a CDF file.

    The date in the file name is determined by the start time of the first event
    in the data

    Parameters
    ----------
    data (xarray.Dataset): The data to write
    instrument (str): The instrument name
    level (str): The data level
    version (str): The version number to append to the file
    directory (str): The directory to write the file to
    mode (str): Instrument mode
    description (str): The description to insert into the file name after the
                        orbit, before the SPICE field.  No underscores allowed.

    Returns
    -------
    str
        The name of the file created
    """
    # Get the start date of the data
    # using Epoch data.
    file_start_date = calc_start_time(data["Epoch"].data[0])
    date_string = np.datetime_as_string(file_start_date, unit="D").replace("-", "")

    # Set file name based on this convention:
    # imap_<instrument>_<datalevel>_<startdate>_<mode>_<descriptor>_<version>.cdf

    description = (
        description
        if (description.startswith("_") or not description)
        else f"_{description}"
    )
    mode = mode if (mode.startswith("_") or not mode) else f"_{mode}"

    filename = (
        f"imap_{instrument}_{level}_{date_string}{mode}{description}_v{version}.cdf"
    )
    filename_and_path = os.path.join(directory, filename)

    # The Logical_file_id is always the name of the file without the extension
    data.attrs["Logical_file_id"] = filename.split(".")[0]

    # Convert the xarray object to a CDF!
    xarray_to_cdf(
        data,
        filename_and_path,
        datetime64_to_cdftt2000=True,
        terminate_on_warning=True,
    )  # Terminate if not ISTP compliant

    return filename_and_path
