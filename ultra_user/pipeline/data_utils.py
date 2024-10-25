import numpy as np
import xarray as xr
import re
import imap_data_access
from pathlib import Path

# Reference start time (launch time or epoch)
# DEFAULT_EPOCH = np.datetime64("2010-01-01T00:01:06.184", "ns")
J2000_EPOCH = np.datetime64("2000-01-01T11:58:55.816", "ns")


def get_cdf_path(dataset: xr.Dataset):
    # Create the filename from the global attributes
    # Logical_source looks like "imap_swe_l2_counts-1min"
    instrument, data_level, descriptor = dataset.attrs["Logical_source"].split("_")[1:]
    dt64 = J2000_EPOCH + dataset["epoch"].values[0].astype("timedelta64[ns]")
    start_time = np.datetime_as_string(dt64, unit="D").replace("-", "")
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

    return file_path

