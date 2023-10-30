import os

import numpy as np
import xarray as xr
from cdflib.xarray import xarray_to_cdf

# Recommended FILLVAL for all integers
INT_FILLVAL = np.iinfo(np.int64).min
# Recommended FILLVALL for all floats
DOUBLE_FILLVAL = np.float64(-1.0e31)
# Recommended min/max Epoch based on MMS approved values
MIN_EPOCH = -315575942816000000
MAX_EPOCH = 946728069183000000

global_base = {
    "Project": "STP>Solar-Terrestrial Physics",
    "Source_name": "IMAP>Interstellar Mapping and Acceleration Probe",
    "Discipline": "Solar Physics>Heliospheric Physics",
    "PI_name": "Dr. David J. McComas",
    "PI_affiliation": [
        "Princeton Plasma Physics Laboratory",
        "100 Stellarator Road, Princeton, NJ 08540",
    ],
    "Instrument_type": "Particles (space)",
    "Mission_group": "IMAP>Interstellar Mapping and Acceleration Probe",
}

epoch_attrs = {
    "CATDESC": "Default time",
    "FIELDNAM": "Epoch",
    "FILLVAL": INT_FILLVAL,
    "FORMAT": "a2",
    "LABLAXIS": "Epoch",
    "UNITS": "ns",
    "VALIDMIN": MIN_EPOCH,
    "VALIDMAX": MAX_EPOCH,
    "VAR_TYPE": "support_data",
    "SCALETYP": "linear",
    "MONOTON": "INCREASE",
    "TIME_BASE": "J2000",
    "TIME_SCALE": "Terrestrial Time",
    "REFERENCE_POSITION": "Rotating Earth Geoid",
}


def write_cdf(data: xr.Dataset, description: str = "", directory: str = ""):
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
        directory (str): The directory to write the file to

    Returns
    -------
        str
            The name of the file created
    """
    # Determine the start date of the data in the file,
    # based on the time of the first dust impact
    file_start_date = data["Epoch"][0].data
    date_string = np.datetime_as_string(file_start_date, unit="D").replace("-", "")

    # Determine the optional "description" field
    description = (
        description
        if (description.startswith("_") or not description)
        else f"_{description}"
    )

    # Determine the file name based on the attributes in the xarray
    filename = (
        data.attrs["Logical_source"]
        + "_"
        + date_string
        + description
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
