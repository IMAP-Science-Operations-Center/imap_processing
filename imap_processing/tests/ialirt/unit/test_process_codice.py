"""Tests for the ``process_codice`` module.

See tests.codice.test_codice_l[1a|1b|2] for more unit tests related to this
code.
"""

import pickle
from pathlib import Path
from unittest.mock import patch

import cdflib
import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice import constants
from imap_processing.codice.codice_l1a_ialirt_hi import l1a_ialirt_hi
from imap_processing.codice.codice_l1a_lo_species import l1a_lo_species
from imap_processing.codice.codice_l1b import convert_to_rates
from imap_processing.codice.codice_l2 import (
    compute_geometric_factors,
    get_efficiency_lut,
    get_geometric_factor_lut,
    process_lo_species_intensity,
)
from imap_processing.codice.decompress import decompress
from imap_processing.ialirt.l0.process_codice import (
    COD_HI_COUNTER,
    COD_LO_COUNTER,
    concatenate_bytes,
    convert_to_intensities,
    create_xarray_dataset,
    process_codice,
    process_ialirt_data_streams,
)
from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.tests.codice.conftest import (
    VALIDATION_FILE_DATE,
    VALIDATION_FILE_VERSION,
)
from imap_processing.utils import packet_file_to_datasets

pytestmark = pytest.mark.external_test_data

OLD_IAL_BIT_STRUCTURE = {
    "SHCOARSE": 32,
    "PACKET_VERSION": 16,
    "SPIN_PERIOD": 16,
    "ACQ_START_SECONDS": 32,
    "ACQ_START_SUBSECONDS": 20,
    "SPARE_00": 8,
    "ST_BIAS_GAIN_MODE": 2,
    "SW_BIAS_GAIN_MODE": 2,
    "TABLE_ID": 32,
    "PLAN_ID": 16,
    "PLAN_STEP": 4,
    "VIEW_ID": 4,
    "RGFO_HALF_SPIN": 6,
    "NSO_HALF_SPIN": 6,
    "SPARE_01": 1,
    "SUSPECT": 1,
    "COMPRESSION": 3,
    "BYTE_COUNT": 23,
}


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


@pytest.fixture(scope="session")
def cod_hi_l1a_test_data_transposed():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_validation"
        / "imap_codice_l1a_hi-ialirt_20260331_v0.0.22.cdf"
    )

    data = load_cdf(data_path)

    return data


@pytest.fixture(scope="session")
def postlaunch_packet_path():
    """Returns the paths to the binary packets."""
    directory = imap_module_directory / "tests" / "ialirt" / "data" / "l0"
    filenames = [
        "iois_1_packets_2026_090_05_03_05",
        "iois_1_packets_2026_090_05_04_06",
        "iois_1_packets_2026_090_05_05_07",
        "iois_1_packets_2026_090_05_06_08",
        "iois_1_packets_2026_090_05_07_09",
    ]
    return tuple(directory / fname for fname in filenames)


@pytest.fixture
def postlaunch_xarray_data(postlaunch_packet_path, sc_packet_path):
    """Create xarray data for multiple packets."""
    apid = 478
    _, xtce_ialirt_path = sc_packet_path

    xarray_data = tuple(
        packet_file_to_datasets(packet, xtce_ialirt_path, use_derived_value=False)[apid]
        for packet in postlaunch_packet_path
    )

    merged_xarray_data = xr.concat(xarray_data, dim="epoch")
    return merged_xarray_data


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
        "acquisition_time_per_esa_step": (
            (
                "epoch",
                "esa_step",
            ),
            cod_lo_l1a_test_data["acquisition_time_per_esa_step"].data,
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


@pytest.fixture(scope="session")
def cod_lo_l2_test_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l2_validation"
        / (
            f"imap_codice_l2_lo-ialirt_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )
    # TODO: fix error in cdf file and change to:
    # data = load_cdf(data_path)
    cdf_file = cdflib.CDF(data_path)

    return cdf_file


@pytest.fixture(scope="session")
def cod_hi_l2_test_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l2_validation"
        / (
            f"imap_codice_l2_hi-ialirt_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )
    # TODO: fix error in cdf file and change to:
    # data = load_cdf(data_path)
    cdf_file = cdflib.CDF(data_path)

    return cdf_file


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
def l1a_lut_path():
    """Returns the calibration data."""
    lut_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_lut"
        / "imap_codice_l1a-sci-lut_20251007_v005.json"
    )

    return lut_path


@pytest.fixture
def l2_lut_path():
    """Returns the calibration data."""
    lut_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l2_lut"
        / "imap_codice_l2-hi-ialirt-efficiency_20251212_v003.csv"
    )

    return lut_path


@pytest.fixture
def l2_processing_dependencies():
    eff_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l2_lut"
        / "imap_codice_l2-lo-efficiency_20251212_v003.csv"
    )
    gf_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l2_lut"
        / "imap_codice_l2-lo-gfactor_20251212_v003.csv"
    )

    return eff_path, gf_path


def test_create_xarray_dataset_basic(l1a_lut_path):
    """Test create_xarray_dataset function."""

    science_values = ["0000000100100011"]
    metadata_values = {
        "VIEW_ID": np.array([0]),
        "TABLE_ID": np.array([3952862729]),
        "ACQ_START_SECONDS": np.array([1625078400]),
        "ACQ_START_SUBSECONDS": np.array([0]),
        "SPIN_PERIOD": np.array([24]),
    }

    ds = create_xarray_dataset(science_values, metadata_values, "lo")

    for key in metadata_values:
        assert key.lower() in ds.variables

    assert ds["pkt_apid"].item() == 1152

    combined_bytes = b"".join(
        int(val, 2).to_bytes(len(val) // 8, byteorder="big") for val in science_values
    )
    assert ds["data"].item() == combined_bytes


@pytest.mark.external_test_data
@patch(
    "imap_processing.codice.constants.IAL_BIT_STRUCTURE",
    OLD_IAL_BIT_STRUCTURE,
)
def test_group_and_decompress_ialirt_cod_lo(
    cod_lo_test_dataset, cod_lo_decom_test_file, l1a_lut_path, cod_lo_l1a_test_data
):
    "Test that I-ALiRT CoDICE-Lo data can be grouped and decompressed properly."

    grouped_cod_lo_data = find_groups(
        cod_lo_test_dataset, (0, COD_LO_COUNTER), "cod_lo_counter", "cod_lo_acq"
    )

    # Verify that we grouped the values properly.
    counter_values = cod_lo_test_dataset["cod_lo_counter"].data
    valid_values = counter_values[counter_values != 255]
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

    dataset = create_xarray_dataset(science_values, metadata_values, "lo")
    result = l1a_lo_species(dataset, l1a_lut_path)

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
@patch(
    "imap_processing.codice.constants.IAL_BIT_STRUCTURE",
    OLD_IAL_BIT_STRUCTURE,
)
def test_group_and_decompress_ialirt_cod_hi(
    cod_hi_test_dataset, cod_hi_decom_test_file, l1a_lut_path, cod_hi_l1a_test_data
):
    "Test that I-ALiRT CoDICE-Hi data can be grouped and decompressed properly."

    codice_hi_counter = 197
    grouped_cod_hi_data = find_groups(
        cod_hi_test_dataset, (0, codice_hi_counter), "cod_hi_counter", "cod_hi_acq"
    )

    # Verify that we grouped the values properly.
    counter_values = cod_hi_test_dataset["cod_hi_counter"].data
    valid_values = counter_values[counter_values != 255]
    resets = np.where(valid_values == codice_hi_counter)

    count = increment = 0
    for reset in resets[0]:
        group = valid_values[increment : reset + 1]
        np.testing.assert_array_equal(
            group, np.arange(0, codice_hi_counter + 1, dtype=np.uint8)
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

    dataset = create_xarray_dataset(science_values, metadata_values, "hi")
    result = l1a_ialirt_hi(dataset, l1a_lut_path)

    expected_species = [
        "h",
    ]

    # Returns data for all expected species at 15 energy steps.
    for species in expected_species:
        np.array_equal(result[species].values, cod_hi_l1a_test_data["h"].data)
    assert np.array_equal(result["data_quality"], cod_hi_l1a_test_data["data_quality"])
    assert np.allclose(
        result["spin_period"].values,
        cod_hi_l1a_test_data["spin_period"].values,
        atol=1e-6,
    )


@pytest.mark.external_test_data
def test_l2_ialirt_cod_hi(cod_hi_l1b_test_data, l2_lut_path, cod_hi_l2_test_data):
    "Test that I-ALiRT CoDICE-Hi L2 data."

    # Read efficiency lookup table
    intensity = convert_to_intensities(cod_hi_l1b_test_data, l2_lut_path, "h")

    # test data
    test_data = cod_hi_l2_test_data["h"]

    np.testing.assert_allclose(
        intensity,
        test_data,
        atol=1e-6,
    )


@pytest.mark.external_test_data
def test_l2_ialirt_cod_lo(
    cod_lo_l1b_test_data, l1a_lut_path, cod_lo_l2_test_data, l2_processing_dependencies
):
    """Test process_codice for hi."""
    eff_path, gf_path = l2_processing_dependencies

    geometric_factor_lookup = get_geometric_factor_lut(None, gf_path)
    geometric_factors = compute_geometric_factors(
        cod_lo_l1b_test_data, geometric_factor_lookup
    )

    efficiency_lookup = get_efficiency_lut(None, eff_path)
    efficiencies = efficiency_lookup[efficiency_lookup["product"] == "sw"]

    # Temporarily store energy_per_charge values from energy_table variable.
    energy_per_charge_values = cod_lo_l1b_test_data["energy_table"].values

    # L1B validation data is missing esa_step coordinate. Create esa_step coordinate.
    # Also, all variables in l1b validation data is using energy_table as coordinate.
    # Update both to match the processing code expectations with rename().
    cod_lo_l1b_test_data = cod_lo_l1b_test_data.rename({"energy_table": "esa_step"})
    # Now, create variable in data_vars with name energy_per_charge and values from
    # energy_table variable.
    cod_lo_l1b_test_data["energy_per_charge"] = xr.DataArray(
        energy_per_charge_values, dims=["esa_step"]
    )

    intensity = process_lo_species_intensity(
        cod_lo_l1b_test_data,
        constants.LO_IALIRT_VARIABLE_NAMES,
        geometric_factors,
        efficiencies,
        constants.SOLAR_WIND_POSITIONS,
    )

    pseudo_density_dict = {}

    for species in constants.LO_IALIRT_VARIABLE_NAMES:
        pseudo_density = (
            intensity[species]
            * np.sqrt(cod_lo_l1b_test_data["energy_per_charge"])
            * np.sqrt(constants.LO_IALIRT_M_OVER_Q[species])
        )  # (epoch, esa_step, spin_sector)

        summed_pseudo_density = pseudo_density.sum(dim="esa_step").squeeze(
            "spin_sector"
        )  # (epoch,)
        pseudo_density_dict[species] = summed_pseudo_density.values

    species = constants.LO_IALIRT_VARIABLE_NAMES

    # Denominator.
    # Note that outside of this test a zero value denominator
    # will lead to a null value.
    # The use of zeros here is only to match the test data as
    # confirmed by the instrument team.
    o_abundance_ratio = (
        pseudo_density_dict[species[3]]
        + pseudo_density_dict[species[4]]
        + pseudo_density_dict[species[5]]
    )

    c_over_o_abundance_ratio = np.divide(
        pseudo_density_dict[species[1]] + pseudo_density_dict[species[2]],
        o_abundance_ratio,
        out=np.full(o_abundance_ratio.shape, np.nan),  # fill with nans by default
        where=o_abundance_ratio != 0,
    )
    mg_over_o_abundance_ratio = np.divide(
        pseudo_density_dict[species[6]],
        o_abundance_ratio,
        out=np.full(o_abundance_ratio.shape, np.nan),
        where=o_abundance_ratio != 0,
    )
    fe_over_o_abundance_ratio = np.divide(
        pseudo_density_dict[species[7]] + pseudo_density_dict[species[8]],
        o_abundance_ratio,
        out=np.full(o_abundance_ratio.shape, np.nan),
        where=o_abundance_ratio != 0,
    )

    c_plus_6_over_c_plus_5_ratio = np.divide(
        pseudo_density_dict[species[2]],
        pseudo_density_dict[species[1]],
        out=np.full(pseudo_density_dict[species[1]].shape, np.nan),
        where=o_abundance_ratio != 0,
    )
    o_plus_7_over_o_plus_6_ratio = np.divide(
        pseudo_density_dict[species[4]],
        pseudo_density_dict[species[3]],
        out=np.full(pseudo_density_dict[species[1]].shape, np.nan),
        where=o_abundance_ratio != 0,
    )
    fe_low_over_fe_high_ratio = np.divide(
        pseudo_density_dict[species[7]],
        pseudo_density_dict[species[8]],
        out=np.full(pseudo_density_dict[species[1]].shape, np.nan),
        where=o_abundance_ratio != 0,
    )

    np.testing.assert_array_equal(
        c_over_o_abundance_ratio, cod_lo_l2_test_data["c_over_o_abundance_ratio"]
    )
    np.testing.assert_array_equal(
        mg_over_o_abundance_ratio, cod_lo_l2_test_data["mg_over_o_abundance_ratio"]
    )
    np.testing.assert_array_equal(
        fe_over_o_abundance_ratio, cod_lo_l2_test_data["fe_over_o_abundance_ratio"]
    )
    np.testing.assert_array_equal(
        c_plus_6_over_c_plus_5_ratio,
        cod_lo_l2_test_data["c_plus_6_over_c_plus_5_ratio"],
    )
    np.testing.assert_array_equal(
        o_plus_7_over_o_plus_6_ratio,
        cod_lo_l2_test_data["o_plus_7_over_o_plus_6_ratio"],
    )
    np.testing.assert_array_equal(
        fe_low_over_fe_high_ratio, cod_lo_l2_test_data["fe_low_over_fe_high_ratio"]
    )


@pytest.mark.external_test_data
@patch(
    "imap_processing.codice.constants.IAL_BIT_STRUCTURE",
    OLD_IAL_BIT_STRUCTURE,
)
def test_process_codice_lo(
    cod_lo_test_dataset,
    l1a_lut_path,
    l2_lut_path,
    cod_lo_l2_test_data,
    l2_processing_dependencies,
    furnish_kernels,
):
    """Test process_codice for hi."""
    eff_path, gf_path = l2_processing_dependencies

    n = cod_lo_test_dataset.dims["epoch"]
    cod_lo_test_dataset = cod_lo_test_dataset.assign(
        sc_sclk_sec=("epoch", np.zeros(n, dtype=np.int64)),
        sc_sclk_sub_sec=("epoch", np.zeros(n, dtype=np.int64)),
    )
    kernels = [
        "naif0012.tls",
        "imap_sclk_0036.tsc",
    ]
    with furnish_kernels(kernels):
        cod_lo_data, _ = process_codice(
            cod_lo_test_dataset, l1a_lut_path, eff_path, "codice_lo", gf_path
        )

    l2_products = [
        "codice_lo_c_over_o_abundance",
        "codice_lo_mg_over_o_abundance",
        "codice_lo_fe_over_o_abundance",
        "codice_lo_c_plus_6_over_c_plus_5",
        "codice_lo_o_plus_7_over_o_plus_6",
        "codice_lo_fe_low_over_fe_high",
    ]

    assert len(cod_lo_data) == 9

    for product in l2_products:
        assert cod_lo_data[0][product] is None


@pytest.mark.external_test_data
def test_process_codice_hi(postlaunch_xarray_data, cod_hi_l1a_test_data_transposed):
    """Test process_codice for hi."""
    grouped_cod_hi_data = find_groups(
        postlaunch_xarray_data, (0, COD_HI_COUNTER), "cod_hi_counter", "cod_hi_acq"
    )
    unique_cod_hi_groups = np.unique(grouped_cod_hi_data["group"])

    for group in unique_cod_hi_groups:
        cod_hi_data_stream = concatenate_bytes(grouped_cod_hi_data, group, "hi")
        cod_hi_science_values, cod_hi_metadata_values = process_ialirt_data_streams(
            [cod_hi_data_stream]
        )
        if not cod_hi_science_values:
            continue
        cod_hi_dataset = create_xarray_dataset(
            cod_hi_science_values, cod_hi_metadata_values, "hi"
        )
        l1a_lut_path = (
            imap_module_directory
            / "tests"
            / "codice"
            / "data"
            / "l1a_lut"
            / "imap_codice_l1a-sci-lut_20260129_v002.json"
        )
        l1a_hi = l1a_ialirt_hi(cod_hi_dataset, l1a_lut_path)

        expected = cod_hi_l1a_test_data_transposed.sel(
            epoch=l1a_hi["epoch"], method="nearest"
        )

        np.testing.assert_array_equal(
            l1a_hi["h"].values,
            expected["h"].data,
        )
