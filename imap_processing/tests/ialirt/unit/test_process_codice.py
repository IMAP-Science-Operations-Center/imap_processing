"""Tests for the ``process_codice`` module.

See tests.codice.test_codice_l[1a|1b|2] for more unit tests related to this
code.
"""

import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice import constants
from imap_processing.codice.codice_l1a import process_ialirt_data_streams
from imap_processing.codice.codice_l1a_lo_species import l1a_lo_species
from imap_processing.codice.codice_l1b import convert_to_rates
from imap_processing.codice.decompress import decompress
from imap_processing.ialirt.l0.process_codice import (
    COD_HI_COUNTER,
    COD_LO_COUNTER,
    FILLVAL_UINT8,
    concatenate_bytes,
    create_xarray_dataset,
    process_codice,
)
from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.tests.codice.conftest import (
    VALIDATION_FILE_DATE,
    VALIDATION_FILE_VERSION,
)
from imap_processing.utils import packet_file_to_datasets

pytestmark = pytest.mark.external_test_data


@pytest.fixture(scope="session")
def l0_test_file():
    return Path(
        imap_module_directory / "tests" / "ialirt" / "data" / "l0" / "apid_478.bin"
    )


@pytest.fixture(scope="session")
def test_datasets(l0_test_file):
    xtce_packet_definition = Path(
        imap_module_directory / "ialirt" / "packet_definitions" / "ialirt.xml"
    )

    datasets = packet_file_to_datasets(l0_test_file, xtce_packet_definition)

    return datasets


@pytest.fixture(scope="session")
def cod_lo_test_file():
    return Path(
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_input"
        / f"imap_codice_l0_lo-ialirt_{VALIDATION_FILE_DATE}_v001.pkts"
    )


@pytest.fixture(scope="session")
def cod_lo_test_dataset(cod_lo_test_file):
    xtce_packet_definition = Path(
        imap_module_directory / "ialirt" / "packet_definitions" / "ialirt_codicelo.xml"
    )

    datasets = packet_file_to_datasets(
        cod_lo_test_file, xtce_packet_definition, use_derived_value=True
    )[1152]

    return datasets


@pytest.fixture(scope="session")
def cod_lo_l1a_test_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_validation"
        / (
            f"imap_codice_l1a_lo-ialirt_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )

    data = load_cdf(data_path)

    return data


@pytest.fixture(scope="session")
def cod_hi_test_file():
    return Path(
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_input"
        / f"imap_codice_l0_hi-ialirt_{VALIDATION_FILE_DATE}_v001.pkts"
    )


@pytest.fixture(scope="session")
def cod_hi_test_dataset(cod_hi_test_file):
    xtce_packet_definition = Path(
        imap_module_directory / "ialirt" / "packet_definitions" / "ialirt_codicehi.xml"
    )

    datasets = packet_file_to_datasets(
        cod_hi_test_file, xtce_packet_definition, use_derived_value=True
    )[1168]

    return datasets


@pytest.fixture
def codice_test_data(test_datasets):
    return test_datasets[478]


@pytest.fixture(scope="session")
def cod_lo_decom_test_file():
    return Path(
        imap_module_directory
        / "tests"
        / "ialirt"
        / "data"
        / "l0"
        / "imap_codice_l1a_lo-ialirt.pickle"
    )


@pytest.fixture(scope="session")
def cod_hi_decom_test_file():
    return Path(
        imap_module_directory
        / "tests"
        / "ialirt"
        / "data"
        / "l0"
        / "imap_codice_l1a_hi-ialirt.pickle"
    )


@pytest.fixture(scope="session")
def cod_lo_l1b_test_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / (
            f"imap_codice_l1b_lo-ialirt_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )

    data = load_cdf(data_path)

    return data


def make_codice_lo_ialirt_dataset(cod_lo_l1a_test_data, descriptor):
    coords = {
        "epoch": cod_lo_l1a_test_data["epoch"],
        "esa_step": cod_lo_l1a_test_data["esa_step"],
        "spin_sector": cod_lo_l1a_test_data["spin_sector"],
    }

    data_vars = {
        "k_factor": ("dim0", cod_lo_l1a_test_data["k_factor"].data),
        "voltage_table": ("esa_step", cod_lo_l1a_test_data["voltage_table"].data),
        "data_quality": ("epoch", cod_lo_l1a_test_data["data_quality"].data),
        "acquisition_time_per_step": (
            "esa_step",
            cod_lo_l1a_test_data["acquisition_time_per_step"].data,
        ),
        "epoch_delta_minus": ("epoch", cod_lo_l1a_test_data["epoch_delta_minus"].data),
        "epoch_delta_plus": ("epoch", cod_lo_l1a_test_data["epoch_delta_plus"].data),
    }

    variables_to_convert = getattr(
        constants, f"{descriptor.upper().replace('-', '_')}_VARIABLE_NAMES"
    )

    for variable in variables_to_convert:
        data_vars[variable] = (
            ("epoch", "esa_step", "spin_sector"),
            cod_lo_l1a_test_data[variable].data,
        )
        data_vars[f"unc_{variable}"] = (
            ("epoch", "esa_step", "spin_sector"),
            cod_lo_l1a_test_data[f"unc_{variable}"].data,
        )

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    return ds


@patch("xarray.Dataset.drop_vars", new=lambda self, *args, **kwargs: self)
@pytest.mark.external_test_data
def test_l1b_ialirt_cod_lo(cod_lo_l1a_test_data, cod_lo_l1b_test_data):
    "Test I-ALiRT CoDICE-Lo l1b data."
    descriptor = "lo-ialirt"
    dataset = make_codice_lo_ialirt_dataset(cod_lo_l1a_test_data, descriptor)
    l1b = convert_to_rates(
        dataset,
        descriptor,
    )
    variables_to_convert = getattr(
        constants, f"{descriptor.upper().replace('-', '_')}_VARIABLE_NAMES"
    )
    for variable in variables_to_convert:
        actual = l1b[variable].data
        expected = cod_lo_l1b_test_data[variable].data

        np.testing.assert_allclose(actual, expected, rtol=1e-5)


@pytest.fixture(scope="session")
def cod_hi_l1a_test_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_validation"
        / (
            f"imap_codice_l1a_hi-ialirt_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )

    data = load_cdf(data_path)

    return data


@pytest.fixture(scope="session")
def cod_hi_l1b_test_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / (
            f"imap_codice_l1b_hi-ialirt_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )

    data = load_cdf(data_path)

    return data


@patch("xarray.Dataset.drop_vars", new=lambda self, *args, **kwargs: self)
@pytest.mark.external_test_data
def test_l1b_ialirt_cod_hi(cod_hi_l1a_test_data, cod_hi_l1b_test_data):
    "Test I-ALiRT CoDICE-Hi l1b data."
    descriptor = "hi-ialirt"
    l1b = convert_to_rates(
        cod_hi_l1a_test_data,
        descriptor,
    )
    variables_to_convert = getattr(
        constants, f"{descriptor.upper().replace('-', '_')}_VARIABLE_NAMES"
    )
    for variable in variables_to_convert:
        actual = l1b[variable].data
        expected = cod_hi_l1b_test_data[variable].data

        np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.fixture
def lut_path():
    """Returns the calibration data."""
    lut_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_lut"
        / "imap_codice_l1a-sci-lut_20251007_v001.json"
    )

    return lut_path


def test_create_xarray_dataset_basic(lut_path):
    """Test create_xarray_dataset function."""

    science_values = ["0000000100100011"]
    metadata_values = {
        "VIEW_ID": np.array([0]),
        "TABLE_ID": np.array([3952862729]),
        "ACQ_START_SECONDS": np.array([1625078400]),
        "ACQ_START_SUBSECONDS": np.array([0]),
        "SPIN_PERIOD": np.array([24]),
    }

    ds = create_xarray_dataset(science_values, metadata_values, "lo", lut_path)

    for key in metadata_values:
        assert key.lower() in ds.variables

    assert ds["pkt_apid"].item() == 1152

    combined_bytes = b"".join(
        int(val, 2).to_bytes(len(val) // 8, byteorder="big") for val in science_values
    )
    assert ds["data"].item() == combined_bytes


@pytest.mark.external_test_data
def test_group_and_decompress_ialirt_cod_lo(
    cod_lo_test_dataset, cod_lo_decom_test_file, lut_path, cod_lo_l1a_test_data
):
    "Test that I-ALiRT CoDICE-Lo data can be grouped and decompressed properly."

    grouped_cod_lo_data = find_groups(
        cod_lo_test_dataset, (0, COD_LO_COUNTER), "cod_lo_counter", "cod_lo_acq"
    )

    # Verify that we grouped the values properly.
    counter_values = cod_lo_test_dataset["cod_lo_counter"].data
    valid_values = counter_values[counter_values != FILLVAL_UINT8]
    resets = np.where(valid_values == COD_LO_COUNTER)

    count = increment = 0
    for reset in resets[0]:
        group = valid_values[increment : reset + 1]
        np.testing.assert_array_equal(
            group, np.arange(0, COD_LO_COUNTER + 1, dtype=np.uint8)
        )
        increment = reset + 1
        count = count + 1

    assert count == int(grouped_cod_lo_data.group.max())

    unique_groups = np.unique(grouped_cod_lo_data["group"])

    # Test data.
    with open(cod_lo_decom_test_file, "rb") as handle:
        data = pickle.load(handle)  # noqa: S301
    test_grouped_data = data["grouped_lo_ialirt"][0]
    test_decom_data = data["decompressed_lo_ialirt"][0]

    header_len = 6  # Test data header at start of block
    checksum_len = 2  # Test data checksum at end of block
    data_len = 3484  # Data length in decompressed packet
    block_size = header_len + data_len + checksum_len

    test_grouped_data_array = []

    for i, group in enumerate(unique_groups):
        compressed_data = concatenate_bytes(grouped_cod_lo_data, group, "lo")

        start = header_len + i * block_size
        end = start + data_len
        expected_slice = test_grouped_data[start:end]

        test_grouped_data_array.append(expected_slice)

        assert expected_slice == compressed_data[:data_len]

    science_values, metadata_values = process_ialirt_data_streams(
        test_grouped_data_array
    )

    for i in range(len(science_values)):
        values = int(science_values[i], 2).to_bytes(
            len(science_values[i]) // 8, byteorder="big"
        )

        decompressed_values = decompress(values, metadata_values["VIEW_ID"][0])
        test_decom_data_array = test_decom_data[i]

        np.testing.assert_array_equal(decompressed_values, test_decom_data_array)

    dataset = create_xarray_dataset(science_values, metadata_values, "lo", lut_path)
    result = l1a_lo_species(dataset, lut_path)

    expected_species = [
        "heplusplus",
        "cplus5",
        "cplus6",
        "oplus6",
        "oplus7",
        "oplus8",
        "mg",
        "fe_loq",
        "fe_hiq",
    ]

    # Returns data for all expected species at 128 esa steps.
    for species in expected_species:
        np.array_equal(result[species].values, cod_lo_l1a_test_data["heplusplus"].data)


@pytest.mark.external_test_data
def test_group_and_decompress_ialirt_cod_hi(
    cod_hi_test_dataset, cod_hi_decom_test_file, lut_path
):
    "Test that I-ALiRT CoDICE-Hi data can be grouped and decompressed properly."

    grouped_cod_hi_data = find_groups(
        cod_hi_test_dataset, (0, COD_HI_COUNTER), "cod_hi_counter", "cod_hi_acq"
    )

    # Verify that we grouped the values properly.
    counter_values = cod_hi_test_dataset["cod_hi_counter"].data
    valid_values = counter_values[counter_values != FILLVAL_UINT8]
    resets = np.where(valid_values == COD_HI_COUNTER)

    count = increment = 0
    for reset in resets[0]:
        group = valid_values[increment : reset + 1]
        np.testing.assert_array_equal(
            group, np.arange(0, COD_HI_COUNTER + 1, dtype=np.uint8)
        )
        increment = reset + 1
        count = count + 1

    assert count == int(grouped_cod_hi_data.group.max())

    unique_groups = np.unique(grouped_cod_hi_data["group"])

    # Test data.
    with open(cod_hi_decom_test_file, "rb") as handle:
        data = pickle.load(handle)  # noqa: S301
    test_grouped_data = data["grouped_hi_ialirt"][0]
    test_decom_data = data["decompressed_hi_ialirt"][0]

    header_len = 6  # Test data header at start of block
    checksum_len = 2  # Test data checksum at end of block
    data_len = 988  # Data length in decompressed packet
    block_size = header_len + data_len + checksum_len

    test_grouped_data_array = []

    for i, group in enumerate(unique_groups):
        compressed_data = concatenate_bytes(grouped_cod_hi_data, group, "hi")

        start = header_len + i * block_size
        end = start + data_len
        expected_slice = test_grouped_data[start:end]

        test_grouped_data_array.append(expected_slice)

        assert expected_slice == compressed_data[:data_len]

    science_values, metadata_values = process_ialirt_data_streams(
        test_grouped_data_array
    )

    for i in range(len(science_values)):
        values = int(science_values[i], 2).to_bytes(
            len(science_values[i]) // 8, byteorder="big"
        )

        decompressed_values = decompress(values, metadata_values["VIEW_ID"][0])

        np.testing.assert_array_equal(decompressed_values, test_decom_data[i])

    dataset = create_xarray_dataset(science_values, metadata_values, "hi", lut_path)  # noqa
    # TODO: add function l1a_hi_species


@pytest.mark.external_test_data
def test_process_codice(codice_test_data, caplog, lut_path):
    """Ensure that the ``process_codice`` function creates a dataset

    Here we just need to make sure the function is returning the expected data.
    CoDICE I-ALiRT data products are being validated separately in the
    ``codice.test_codice_l[1a|1b|2]`` modules.
    """

    with caplog.at_level("WARNING"):
        cod_lo_data, cod_hi_data = process_codice(codice_test_data, lut_path)

    assert isinstance(cod_lo_data, list)
    assert all(isinstance(item, dict) for item in cod_lo_data)
    assert isinstance(cod_hi_data, list)
    assert all(isinstance(item, dict) for item in cod_hi_data)
