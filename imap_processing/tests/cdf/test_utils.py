import numpy as np
import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import calc_start_time, write_cdf
from imap_processing.swe.swe_cdf_attrs import swe_l1a_global_attrs


def test_calc_start_time():
    # TODO: Update this when launch time is updated
    launch_time = np.datetime64("2010-01-01T00:01:06.184")
    assert calc_start_time(0) == launch_time
    assert calc_start_time(1) == launch_time + np.timedelta64(1, "s")


def test_write_cdf(tmp_path):
    # Set up a fake dataset
    # lots of requirements on attributes, so depend on SWE for now
    dataset = xr.Dataset(
        {"Epoch": ("Epoch", [1, 2, 3])},
        attrs=swe_l1a_global_attrs.output()
        | {"Logical_source": "imap_test_l1", "Data_version": "01"},
    )
    dataset["Epoch"].attrs = ConstantCoordinates.EPOCH

    fname = write_cdf(dataset, descriptor="test-descriptor")
    assert fname.exists()
    assert fname.name == "imap_test_l1_test-descriptor_20100101_v01.cdf"
    # Created automatically for us
    dir_structure = fname.parts[-5:-1]
    # instrument, level, year, month
    assert dir_structure == ("test", "l1", "2010", "01")

    # Test an explicit directory doesn't create that structure
    filename = write_cdf(dataset, descriptor="test-descriptor", directory=tmp_path)
    assert filename.exists()
    assert filename.name == "imap_test_l1_test-descriptor_20100101_v01.cdf"
    # Created automatically for us
    dir_structure = filename.parts[-5:-1]
    # It should be the same as the tmp_path structure (minus the file name)
    assert dir_structure == tmp_path.parts[-4:]
