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
    """Returns the xtce directory."""
    return (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "apid01152.tlm"
    )


@pytest.fixture(scope="session")
def codicelo_validation_data():
    """Returns the test data directory."""
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
def codicelo_test_data(binary_packet_path, xtce_codicelo_path):
    """Create xarray data"""
    apid = 1152
    codicelo_test_data = packet_file_to_datasets(
        binary_packet_path, xtce_codicelo_path
    )[apid]

    return codicelo_test_data


def test_find_groups(codicelo_test_data):
    """Tests find_groups"""

    grouped_data = find_groups(codicelo_test_data)
    unique_groups = np.unique(grouped_data["group"])
    for group in unique_groups:
        group_data = grouped_data["cod_lo_counter"].values[
            grouped_data["group"] == group
        ]
        np.testing.assert_array_equal(group_data, np.arange(233))


def test_append_cod_lo_data(codicelo_test_data):
    """Tests append_cod_lo_data"""

    grouped_data = find_groups(codicelo_test_data)
    unique_groups = np.unique(grouped_data["group"])
    for group in unique_groups:
        mask = grouped_data["group"] == group
        filtered_indices = np.where(mask)[0]
        group_data = grouped_data.isel(epoch=filtered_indices)
        expected_cod_lo_counter = np.repeat(group_data["cod_lo_counter"].values, 15)
        appended_data = append_cod_lo_data(group_data)
        assert np.array_equal(
            appended_data["cod_lo_counter"].values, expected_cod_lo_counter.astype(int)
        )


def test_process_codicelo(codicelo_test_data, codicelo_validation_data, caplog):
    """Tests process_codicelo."""
    codicelo_product = process_codicelo(codicelo_test_data)
    assert codicelo_product == [{}]

    indices = (codicelo_test_data["cod_lo_acq"] != 0).values.nonzero()[0]
    codicelo_test_data["cod_lo_counter"].values[indices[0] : indices[0] + 233] = (
        np.random.permutation(233)
    )

    with caplog.at_level("WARNING"):
        process_codicelo(codicelo_test_data)

    assert any(
        "does not contain all values from 0 to 232 without duplicates" in message
        for message in caplog.text.splitlines()
    )
