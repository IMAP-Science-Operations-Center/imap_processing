import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.lo.l1a.lo_l1a import lo_l1a


@pytest.fixture
def housekeeping_datasets():
    dependency = (
        imap_module_directory / "tests/lo/test_pkts/imap_lo_l0_raw_20240803_v002.pkts"
    )
    # Two datasets, l1a and l1b. Only test the l1b dataset.
    datasets = lo_l1a(dependency)
    return datasets


def test_housekeeping(housekeeping_datasets):
    validation_l1a_file = (
        imap_module_directory
        / "tests/lo/validation_data"
        / "Instrument_FM1_T104_R129_20240803_ILO_APP_NHK_DN_trimmed.csv"
    )
    validation_data_l1a = pd.read_csv(validation_l1a_file)
    validation_l1b_file = (
        imap_module_directory
        / "tests/lo/validation_data"
        / "Instrument_FM1_T104_R129_20240803_ILO_APP_NHK_EU_trimmed.csv"
    )
    validation_data_l1b = pd.read_csv(validation_l1b_file)

    ## Assert
    # Get the l1a dataset
    ds = next(
        x
        for x in housekeeping_datasets
        if x.attrs["Logical_source"] == "imap_lo_l1a_nhk"
    )
    # We are only spot checking a few values from the validation file
    # the first 3 and the final value.
    small_ds = ds.isel(epoch=[0, 1, 2, -1])
    for var in validation_data_l1a.columns:
        print(var)
        if var == "PPM_UPPER_BOUND":
            # This is 65535 in the validation data, but only 4095 in the dataset.
            # This is because it is defined as a 12-bit quantity in the packet
            # definition, so it can't feasibly be 65535 in the unpacked data.
            # Ignore for now as this field is inconsequential.
            continue
        np.testing.assert_array_equal(
            small_ds[var.lower()], validation_data_l1a[var.upper()]
        )

    # Get the l1b dataset
    ds = next(
        x
        for x in housekeeping_datasets
        if x.attrs["Logical_source"] == "imap_lo_l1b_nhk"
    )
    small_ds = ds.isel(epoch=[0, 1, 2, -1])
    # Some of the fields are decoded states in the output dataset, but left as
    # raw integers in the validation data. So we just spot check a few important
    # fields here to ensure they match and were converted correctly.
    for var in [
        "shcoarse",
        "pcc_cumulative_cnt_pri",
        "pcc_cumulative_cnt_sec",
        "lvps_5v",
    ]:
        np.testing.assert_array_equal(small_ds[var], validation_data_l1b[var.upper()])
