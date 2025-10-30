"""Tests for the ``process_codice`` module.

See tests.codice.test_codice_l[1a|1b|2] for more unit tests related to this
code.
"""

from pathlib import Path

import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice.codice_l1b import convert_to_rates
from imap_processing.ialirt.l0.process_codice import (
    COD_HI_COUNTER,
    COD_HI_RANGE,
    COD_LO_COUNTER,
    COD_LO_RANGE,
    FILLVAL_UINT8,
    concatenate_bytes,
    process_codice,
)
from imap_processing.ialirt.utils.grouping import find_groups
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
        / "imap_codice_lo-ialirt_20250814_v001.pkts"
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
def cod_hi_test_file():
    return Path(
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_input"
        / "imap_codice_hi-ialirt_20250814_v001.pkts"
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
def cod_lo_l1a_test_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_validation"
        / "imap_codice_l1a_lo-ialirt_20250814211100_v0.0.5.cdf"
    )

    data = load_cdf(data_path)

    return data

import numpy as np
import xarray as xr

def make_codice_lo_ialirt_dataset(cod_lo_l1a_test_data):
    n_epoch = 9
    n_esa = 128
    n_sector = 1

    coords = {
        "epoch": cod_lo_l1a_test_data["epoch"],
        "esa_step": cod_lo_l1a_test_data["spin_sector_index"],
        "esa_step_label": [str(i) for i in range(n_esa)],
        "k_factor": cod_lo_l1a_test_data["k_factor"],
        "spin_sector": [0],
        "spin_sector_label": ["0"],
    }

    # --- Data variables ---
    data_vars = {
        # simple 1D variables
        "spin_period": ("epoch",
                        cod_lo_l1a_test_data["spin_period"].data),
        "voltage_table": ("esa_step",
                          cod_lo_l1a_test_data["voltage_table"].data),
        "data_quality": ("epoch",
                         cod_lo_l1a_test_data["data_quality"].data),
        "acquisition_time_per_step": ("esa_step",
                                      cod_lo_l1a_test_data["acquisition_time_per_step"].data),
        "sw_bias_gain_mode": ("epoch",
                              cod_lo_l1a_test_data["sw_bias_gain_mode"].data),
        "st_bias_gain_mode": ("epoch",
                              cod_lo_l1a_test_data["st_bias_gain_mode"].data),
        "epoch_delta_minus": ("epoch",
                              cod_lo_l1a_test_data["epoch_delta_minus"].data),
        "epoch_delta_plus": ("epoch",
                             cod_lo_l1a_test_data["epoch_delta_plus"].data),
        "heplusplus": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["heplusplus"].data),
        "unc_heplusplus": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["unc_heplusplus"].data),
        "cplus5": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["cplus5"].data),
        "unc_cplus5": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["unc_cplus5"].data),
        "cplus6": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["cplus6"].data),
        "unc_cplus6": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["unc_cplus6"].data),
        "oplus6": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["oplus6"].data),
        "unc_oplus6": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["unc_oplus6"].data),
        "oplus7": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["oplus7"].data),
        "unc_oplus7": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["unc_oplus7"].data),
        "oplus8": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["oplus8"].data),
        "unc_oplus8": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["unc_oplus8"].data),
        "mg": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["mg"].data),
        "unc_mg": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["unc_mg"].data),
        "fe_loq": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["fe_loq"].data),
        "unc_fe_loq": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["unc_fe_loq"].data),
        "fe_hiq": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["fe_hiq"].data),
        "unc_fe_hiq": (("epoch", "esa_step", "spin_sector"),
                       cod_lo_l1a_test_data["unc_fe_hiq"].data),
        "nso_half_spin": (("epoch"),
                        cod_lo_l1a_test_data["nso_half_spin"].data),
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    return ds

@pytest.mark.external_test_data
def test_l1b_ialirt_cod_lo(cod_lo_l1a_test_data):

    dataset = load_cdf("/Users/lasa6858/Desktop/imap_codice_l1a_lo-sw-species_20250814_v999.cdf")
    ds = make_codice_lo_ialirt_dataset(cod_lo_l1a_test_data)
    l1b = convert_to_rates(
        ds,
        "lo-ialirt",
    )
    print("hi")


@pytest.mark.external_test_data
def test_group_and_decompress_ialirt_cod_lo(cod_lo_test_dataset):
    "Test that I-ALiRT CoDICE-Lo data can be grouped properly."

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

    for group in unique_groups:
        compressed_data = concatenate_bytes(grouped_cod_lo_data, group, "lo")
        byte_data = np.frombuffer(compressed_data, dtype=np.uint8)
        num_bits = byte_data.size * 8
        assert num_bits == (COD_LO_COUNTER + 1) * len(COD_LO_RANGE) * 8
        # TODO: left off here. Need to validate decompression with test data.
        # decompressed_data = decompress._apply_pack_24_bit(compressed_data)


@pytest.mark.external_test_data
def test_group_and_decompress_ialirt_cod_hi(cod_hi_test_dataset):
    "Test that I-ALiRT CoDICE-Hi data can be grouped properly."

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

    for group in unique_groups:
        compressed_data = concatenate_bytes(grouped_cod_hi_data, group, "hi")
        byte_data = np.frombuffer(compressed_data, dtype=np.uint8)
        num_bits = byte_data.size * 8
        assert num_bits == (COD_HI_COUNTER + 1) * len(COD_HI_RANGE) * 8
        # TODO: left off here. Need to validate decompression with test data.
        # decompressed_data = decompress._apply_loggy_a(compressed_data)


def test_process_codice(codice_test_data, caplog):
    """Ensure that the ``process_codice`` function creates a dataset

    Here we just need to make sure the function is returning the expected data.
    CoDICE I-ALiRT data products are being validated separately in the
    ``codice.test_codice_l[1a|1b|2]`` modules.
    """

    with caplog.at_level("WARNING"):
        cod_lo_data, cod_hi_data = process_codice(codice_test_data)

    assert isinstance(cod_lo_data, list)
    assert all(isinstance(item, dict) for item in cod_lo_data)
    assert isinstance(cod_hi_data, list)
    assert all(isinstance(item, dict) for item in cod_hi_data)
