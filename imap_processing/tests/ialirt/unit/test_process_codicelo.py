import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.ialirt.l0.process_codicelo import (
    append_cod_lo_data,
    find_groups,
    process_codicelo,
)
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def xtce_codicelo_path():
    """Returns the xtce directory."""
    return (
        imap_module_directory / "ialirt" / "packet_definitions" / "ialirt_codicelo.xml"
    )


@pytest.fixture(scope="session")
def binary_packet_path():
    """Returns the xtce auxiliary directory."""
    return (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "apid01152.tlm"
    )


@pytest.fixture(scope="session")
def codicelo_test_data():
    """Returns the xtce auxiliary directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "imap_codice_l1a_lo-ialirt_20241110193700_v0.0.0.cdf"
    )
    data = load_cdf(data_path)

    return data


@pytest.fixture()
def xarray_data(binary_packet_path, xtce_codicelo_path):
    """Create xarray data"""
    apid = 1152
    xarray_data = packet_file_to_datasets(binary_packet_path, xtce_codicelo_path)[apid]

    return xarray_data


def test_find_groups(xarray_data):
    """Tests find_groups"""

    grouped_data = find_groups(xarray_data)
    group_1_data = grouped_data["src_seq_ctr"].values[grouped_data["group"] == 1]

    np.testing.assert_array_equal(group_1_data, np.arange(240))


def test_append_cod_lo_data(xarray_data):
    """Tests append_cod_lo_data"""

    grouped_data = find_groups(xarray_data)
    unique_groups = np.unique(grouped_data["group"])
    for group in unique_groups:
        mask = grouped_data["group"] == group
        group_data = grouped_data.sel(group=mask)
        expected_src_seq_ctr = np.repeat(group_data["src_seq_ctr"].values, 15)
        appended_data = append_cod_lo_data(group_data)
        assert np.array_equal(
            appended_data["src_seq_ctr"].values, expected_src_seq_ctr.astype(int)
        )


def test_process_codicelo(xarray_data, codicelo_test_data, caplog):
    """Tests process_codicelo."""
    codicelo_product = process_codicelo(xarray_data)
    assert codicelo_product == {}

    indices = (xarray_data["cod_lo_acq"] != 0).values.nonzero()[0]
    xarray_data["src_seq_ctr"].values[indices[0] : indices[0] + 240] = (
        np.random.permutation(240)
    )

    with caplog.at_level("WARNING"):
        process_codicelo(xarray_data)

    assert any(
        "does not contain all values from 0 to 239 without duplicates" in message
        for message in caplog.text.splitlines()
    )
