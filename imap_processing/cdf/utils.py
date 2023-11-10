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


def write_cdf(
    data: xr.Dataset, description: str = "", mode: str = "", directory: str = ""
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
        data (xarray.Dataset): The dataset object to convert to a CDF
        description (str): The description to insert into the file name after the
                            orbit, before the SPICE field.  No underscores allowed.
        mode (str): Instrument mode
        directory (str): The directory to write the file to

    Returns
    -------
        str
            The name of the file created
    """
    # Determine the start date of the data in the file,
    # based on the time of the first dust impact
    file_start_date = None
    if "idex" in data.attrs["Logical_source"]:
        file_start_date = data["Epoch"][0].data
    elif "swe" in data.attrs["Logical_source"]:
        start_time = data["Epoch"].data[0]
        file_start_date = calc_start_time(start_time)
    if file_start_date is None:
        raise ValueError(
            "Unable to determine file start date. Check Logical_source value"
        )

    date_string = np.datetime_as_string(file_start_date, unit="D").replace("-", "")

    # Determine the optional "description" field
    description = (
        description
        if (description.startswith("_") or not description)
        else f"_{description}"
    )
    mode = mode if (mode.startswith("_") or not mode) else f"_{mode}"

    # Determine the file name based on the attributes in the xarray
    # Set file name based on this convention:
    # imap_<instrument>_<datalevel>_<mode>_<descriptor>_<startdate>_
    # <version>.cdf
    # data.attrs["Logical_source"] has the mission, instrument, and level
    # like this:
    #   imap_idex_l1
    filename = (
        data.attrs["Logical_source"]
        + mode
        + description
        + "_"
        + date_string
        + f"_v{data.attrs['Data_version']}.cdf"
    )
    filename_and_path = os.path.join(directory, filename)

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
