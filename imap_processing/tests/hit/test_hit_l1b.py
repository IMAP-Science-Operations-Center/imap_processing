import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.l1a import hit_l1a
from imap_processing.hit.l1b.hit_l1b import (
    PARTICLE_ENERGY_RANGE_MAPPING,
    SummedCounts,
    add_energy_variables,
    add_rates_to_dataset,
    calculate_summed_counts,
    create_particle_data_arrays,
    hit_l1b,
    process_standard_rates_data,
    process_summed_rates_data,
)
from imap_processing.tests.hit.helpers.l1_validation import (
    prepare_standard_rates_validation_data,
)

# TODO: Packet files are per apid at the moment so the tests currently
#  reflect this. Eventually, HIT will provide a packet file with all apids
#  and the tests will need to be updated.


@pytest.fixture(scope="module")
def packet_filepath():
    """Set path to test data file"""
    # TODO: Update this path when HIT provides a packet file with all apids.
    #  Current test file only has the housekeeping apid is available.
    return (
        imap_module_directory / "tests/hit/test_data/imap_hit_l0_raw_20100105_v001.pkts"
    )


@pytest.fixture(scope="module")
def sci_packet_filepath():
    """Set path to test data file"""
    return imap_module_directory / "tests/hit/test_data/sci_sample.ccsds"


@pytest.fixture()
def dependencies(packet_filepath, sci_packet_filepath):
    """Get dependencies for L1B processing"""
    # Create dictionary of dependencies and add CCSDS packet file
    data_dict = {"imap_hit_l0_raw": packet_filepath}
    # Add L1A datasets
    l1a_datasets = hit_l1a.hit_l1a(packet_filepath, "001")
    # TODO: Remove this when HIT provides a packet file with all apids.
    l1a_datasets.extend(hit_l1a.hit_l1a(sci_packet_filepath, "001"))
    for dataset in l1a_datasets:
        data_dict[dataset.attrs["Logical_source"]] = dataset
    return data_dict


@pytest.fixture()
def l1b_hk_dataset(dependencies):
    """Get the housekeeping dataset"""
    datasets = hit_l1b(dependencies, "001")
    for dataset in datasets:
        if dataset.attrs["Logical_source"] == "imap_hit_l1b_hk":
            return dataset


@pytest.fixture()
def l1b_standard_rates_dataset(dependencies):
    """Get the standard rates dataset"""
    # TODO: use this fixture in future unit test to validate the standard rates dataset
    datasets = hit_l1b(dependencies, "001")
    for dataset in datasets:
        if dataset.attrs["Logical_source"] == "imap_hit_l1b_standard-rates":
            return dataset


@pytest.fixture()
def l1a_counts_dataset(sci_packet_filepath):
    """Get L1A counts dataset to test l1b processing functions"""
    l1a_datasets = hit_l1a.hit_l1a(sci_packet_filepath, "001")
    for dataset in l1a_datasets:
        if dataset.attrs["Logical_source"] == "imap_hit_l1a_counts":
            return dataset


@pytest.fixture()
def livetime(l1a_counts_dataset):
    """Calculate livetime for L1A counts dataset"""
    return l1a_counts_dataset["livetime_counter"] / 270


def test_calculate_summed_counts():
    # Create a mock raw_counts_dataset
    data = {
        "l2fgrates": (
            ("epoch", "index"),
            np.array([[1, 2, 3, 4, 5]] * 5, dtype=np.int64),
        ),
        "l3fgrates": (
            ("epoch", "index"),
            np.array([[6, 7, 8, 9, 10]] * 5, dtype=np.int64),
        ),
        "penfgrates": (
            ("epoch", "index"),
            np.array([[11, 12, 13, 14, 15]] * 5, dtype=np.int64),
        ),
        "l2fgrates_delta_minus": (
            ("epoch", "index"),
            np.zeros((5, 5), dtype=np.float32),
        ),
        "l3fgrates_delta_minus": (
            ("epoch", "index"),
            np.full((5, 5), 0.01, dtype=np.float32),
        ),
        "penfgrates_delta_minus": (
            ("epoch", "index"),
            np.full((5, 5), 0.001, dtype=np.float32),
        ),
        "l2fgrates_delta_plus": (
            ("epoch", "index"),
            np.full((5, 5), 0.02, dtype=np.float32),
        ),
        "l3fgrates_delta_plus": (
            ("epoch", "index"),
            np.full((5, 5), 0.002, dtype=np.float32),
        ),
        "penfgrates_delta_plus": (
            ("epoch", "index"),
            np.full((5, 5), 0.003, dtype=np.float32),
        ),
    }
    coords = {"epoch": np.arange(5), "index": np.arange(5)}
    raw_counts_dataset = xr.Dataset(data, coords=coords)

    # Define count_indices
    count_indices = {
        "R2": [0, 1],
        "R3": [2, 3],
        "R4": [4],
    }

    # Call the function
    summed_counts, summed_counts_delta_minus, summed_counts_delta_plus = (
        calculate_summed_counts(raw_counts_dataset, count_indices)
    )

    # Expected values based on `count_indices`
    expected_summed_counts = np.array([35, 35, 35, 35, 35])
    expected_summed_counts_delta_minus = np.array([0.021, 0.021, 0.021, 0.021, 0.021])
    expected_summed_counts_delta_plus = np.array([0.047, 0.047, 0.047, 0.047, 0.047])

    # Assertions
    assert summed_counts.shape == (5,)
    assert summed_counts_delta_minus.shape == (5,)
    assert summed_counts_delta_plus.shape == (5,)

    np.testing.assert_array_almost_equal(summed_counts.values, expected_summed_counts)
    np.testing.assert_array_almost_equal(
        summed_counts_delta_minus.values, expected_summed_counts_delta_minus
    )
    np.testing.assert_array_almost_equal(
        summed_counts_delta_plus.values, expected_summed_counts_delta_plus
    )

    # Check dtype consistency
    assert summed_counts.dtype == np.int64, f"Unexpected dtype: {summed_counts.dtype}"
    assert (
        summed_counts_delta_minus.dtype == np.float32
    ), f"Unexpected dtype: {summed_counts_delta_minus.dtype}"
    assert (
        summed_counts_delta_plus.dtype == np.float32
    ), f"Unexpected dtype: {summed_counts_delta_plus.dtype}"


def test_add_rates_to_dataset():
    # Create a sample dataset
    dataset = xr.Dataset(
        {
            "epoch": ("epoch", np.arange(10)),
            "livetime": ("epoch", np.random.rand(10) + 1),  # Avoid division by zero
        }
    )

    # Add empty data arrays for a sample particle
    particle = "test_particle"
    dataset[particle] = xr.DataArray(
        data=np.zeros((10, 5), dtype=np.float32),
        dims=["epoch", f"{particle}_energy_index"],
    )
    dataset[f"{particle}_delta_minus"] = xr.DataArray(
        data=np.zeros((10, 5), dtype=np.float32),
        dims=["epoch", f"{particle}_energy_index"],
    )
    dataset[f"{particle}_delta_plus"] = xr.DataArray(
        data=np.zeros((10, 5), dtype=np.float32),
        dims=["epoch", f"{particle}_energy_index"],
    )

    # Set the random seed for reproducibility
    np.random.seed(42)

    # Define the summed counts with random values in a namedtuple
    summed_counts = SummedCounts(
        xr.DataArray(np.random.rand(10), dims=["epoch"]),
        xr.DataArray(np.random.rand(10), dims=["epoch"]),
        xr.DataArray(np.random.rand(10), dims=["epoch"]),
    )

    # Call the function
    updated_dataset = add_rates_to_dataset(
        dataset, particle, 0, summed_counts, dataset["livetime"]
    )

    # Check the results
    np.testing.assert_array_almost_equal(
        updated_dataset[particle][:, 0].values,
        summed_counts.summed_counts / dataset["livetime"].values,
    )
    np.testing.assert_array_almost_equal(
        updated_dataset[f"{particle}_delta_minus"][:, 0].values,
        summed_counts.summed_counts_delta_minus / dataset["livetime"].values,
    )
    np.testing.assert_array_almost_equal(
        updated_dataset[f"{particle}_delta_plus"][:, 0].values,
        summed_counts.summed_counts_delta_plus / dataset["livetime"].values,
    )


def test_add_energy_variables():
    dataset = xr.Dataset()
    particle = "test_particle"
    energy_min = np.array([1.8, 4.0, 6.0], dtype=np.float32)
    energy_max = np.array([2.2, 6.0, 10.0], dtype=np.float32)
    result = add_energy_variables(dataset, particle, energy_min, energy_max)
    assert f"{particle}_energy_min" in result.data_vars
    assert f"{particle}_energy_max" in result.data_vars
    assert np.all(result[f"{particle}_energy_min"].values == energy_min)
    assert np.all(result[f"{particle}_energy_max"].values == energy_max)


def test_create_particle_data_arrays():
    dataset = xr.Dataset()
    particle = "test_particle"
    result = create_particle_data_arrays(
        dataset, particle, num_energy_ranges=3, epoch_size=10
    )

    assert f"{particle}" in result.data_vars
    assert f"{particle}_delta_minus" in result.data_vars
    assert f"{particle}_delta_plus" in result.data_vars
    assert f"{particle}_energy_index" in result.coords

    for var in result.data_vars:
        assert result[var].shape == (10, 3)

    assert result[f"{particle}_energy_index"].shape == (3,)


def test_process_summed_rates_data(l1a_counts_dataset, livetime):
    """Test the variables in the summed rates dataset"""

    l1b_summed_rates_dataset = process_summed_rates_data(l1a_counts_dataset, livetime)

    # Check that a xarray dataset is returned
    assert isinstance(l1b_summed_rates_dataset, xr.Dataset)

    valid_coords = {
        "epoch",
        "h_energy_index",
        "he3_energy_index",
        "he4_energy_index",
        "he_energy_index",
        "c_energy_index",
        "o_energy_index",
        "fe_energy_index",
        "n_energy_index",
        "si_energy_index",
        "mg_energy_index",
        "s_energy_index",
        "ar_energy_index",
        "ca_energy_index",
        "na_energy_index",
        "al_energy_index",
        "ne_energy_index",
        "ni_energy_index",
    }

    # Check that the dataset has the correct coords and variables
    assert valid_coords == set(l1b_summed_rates_dataset.coords), "Coordinates mismatch"

    assert "dynamic_threshold_state" in l1b_summed_rates_dataset.data_vars

    for particle in PARTICLE_ENERGY_RANGE_MAPPING.keys():
        assert f"{particle}" in l1b_summed_rates_dataset.data_vars
        assert f"{particle}_delta_minus" in l1b_summed_rates_dataset.data_vars
        assert f"{particle}_delta_plus" in l1b_summed_rates_dataset.data_vars
        assert f"{particle}_energy_min" in l1b_summed_rates_dataset.data_vars
        assert f"{particle}_energy_max" in l1b_summed_rates_dataset.data_vars


def test_process_standard_rates_data(l1a_counts_dataset, livetime):
    """Test the variables in the standard rates dataset"""
    l1b_standard_rates_dataset = process_standard_rates_data(
        l1a_counts_dataset, livetime
    )

    # Check that a xarray dataset is returned
    assert isinstance(l1b_standard_rates_dataset, xr.Dataset)

    # Define the data variables that should be present in the dataset
    valid_data_vars = {
        "sngrates",
        "coinrates",
        "pbufrates",
        "l2fgrates",
        "l2bgrates",
        "l3fgrates",
        "l3bgrates",
        "penfgrates",
        "penbgrates",
        "ialirtrates",
        "l4fgrates",
        "l4bgrates",
        "sngrates_delta_plus",
        "coinrates_delta_plus",
        "pbufrates_delta_plus",
        "l2fgrates_delta_plus",
        "l2bgrates_delta_plus",
        "l3fgrates_delta_plus",
        "l3bgrates_delta_plus",
        "penfgrates_delta_plus",
        "penbgrates_delta_plus",
        "ialirtrates_delta_plus",
        "l4fgrates_delta_plus",
        "l4bgrates_delta_plus",
        "sngrates_delta_minus",
        "coinrates_delta_minus",
        "pbufrates_delta_minus",
        "l2fgrates_delta_minus",
        "l2bgrates_delta_minus",
        "l3fgrates_delta_minus",
        "l3bgrates_delta_minus",
        "penfgrates_delta_minus",
        "penbgrates_delta_minus",
        "ialirtrates_delta_minus",
        "l4fgrates_delta_minus",
        "l4bgrates_delta_minus",
        "dynamic_threshold_state",
    }

    valid_coords = [
        "epoch",
        "gain",
        "sngrates_index",
        "coinrates_index",
        "pbufrates_index",
        "l2fgrates_index",
        "l2bgrates_index",
        "l3fgrates_index",
        "l3bgrates_index",
        "penfgrates_index",
        "penbgrates_index",
        "ialirtrates_index",
        "l4fgrates_index",
        "l4bgrates_index",
    ]

    # Check that the dataset has the correct variables
    assert valid_data_vars == set(
        l1b_standard_rates_dataset.data_vars.keys()
    ), "Data variables mismatch"
    assert valid_coords == list(
        l1b_standard_rates_dataset.coords
    ), "Coordinates mismatch"


def test_hit_l1b_hk_dataset_variables(l1b_hk_dataset):
    """Test the variables in the housekeeping dataset"""
    # Define the keys that should have dropped from the housekeeping dataset
    dropped_keys = {
        "pkt_apid",
        "version",
        "type",
        "sec_hdr_flg",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "hskp_spare1",
        "hskp_spare2",
        "hskp_spare3",
        "hskp_spare4",
        "hskp_spare5",
    }
    # Define the keys that should be present in the housekeeping dataset
    valid_keys = {
        "sc_tick",
        "heater_on",
        "fsw_version_b",
        "ebox_m12va",
        "phasic_stat",
        "ebox_3d4vd",
        "ebox_p2d0vd",
        "temp1",
        "last_bad_seq_num",
        "ebox_m5d7va",
        "ebox_p12va",
        "table_status",
        "enable_50khz",
        "mram_disabled",
        "temp3",
        "preamp_l1a",
        "l2ab_bias",
        "l34b_bias",
        "fsw_version_c",
        "num_evnt_last_hk",
        "dac1_enable",
        "preamp_l234b",
        "analog_temp",
        "fee_running",
        "fsw_version_a",
        "num_errors",
        "test_pulser_on",
        "dac0_enable",
        "preamp_l1b",
        "l1ab_bias",
        "l34a_bias",
        "leak_i",
        "last_good_cmd",
        "lvps_temp",
        "idpu_temp",
        "temp2",
        "preamp_l234a",
        "last_good_seq_num",
        "num_good_cmds",
        "heater_control",
        "hvps_temp",
        "ebox_p5d7va",
        "spin_period_long",
        "enable_hvps",
        "temp0",
        "spin_period_short",
        "dyn_thresh_lvl",
        "num_bad_cmds",
        "adc_mode",
        "ebox_5d1vd",
        "active_heater",
        "last_error_num",
        "last_bad_cmd",
        "ref_p5v",
        "code_checksum",
        "mode",
    }
    # Check that the dataset has the correct variables
    assert valid_keys == set(l1b_hk_dataset.data_vars.keys())
    assert set(dropped_keys).isdisjoint(set(l1b_hk_dataset.data_vars.keys()))

    # Define the coordinates and dimensions. Both have equivalent values
    dataset_coords_dims = {"epoch", "adc_channels", "adc_channels_label"}

    # Check that the dataset has the correct coordinates, and dimensions
    assert l1b_hk_dataset.coords.keys() == dataset_coords_dims


def test_validate_l1b_hk_data(l1b_hk_dataset):
    """Test to validate the housekeeping dataset created by the L1B processing.

    Parameters
    ----------
    l1b_hk_dataset : xr.Dataset
        Housekeeping dataset created by L1B processing.
    """

    # Load the validation data
    validation_file = (
        imap_module_directory / "tests/hit/validation_data/hskp_sample_eu_3_6_2025.csv"
    )
    validation_data = pd.read_csv(validation_file)
    validation_data.columns = validation_data.columns.str.lower().str.strip()

    # Get a list of leak columns in ascending order
    # (LEAK_I_00, LEAK_I_01, ..., LEAK_I_63)
    # and group values into a single column
    leak_columns = [
        col for col in validation_data.columns if col.startswith("leak_i_")
    ][::-1]
    validation_data["leak_i"] = validation_data[leak_columns].apply(
        lambda row: row.values.astype(np.float64), axis=1
    )
    validation_data.drop(columns=leak_columns, inplace=True)

    # Define the keys that should have dropped from the housekeeping dataset
    dropped_fields = {
        "pkt_apid",
        "version",
        "type",
        "sec_hdr_flg",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "hskp_spare1",
        "hskp_spare2",
        "hskp_spare3",
        "hskp_spare4",
        "hskp_spare5",
    }

    # Check that dropped variables are not in the dataset
    assert set(dropped_fields).isdisjoint(set(l1b_hk_dataset.data_vars.keys()))

    # Define the keys that should be ignored in the validation
    # like ccsds headers
    ignore_validation_fields = {
        "ccsds_version",
        "ccsds_type",
        "ccsds_sec_hdr_flag",
        "ccsds_appid",
        "ccsds_grp_flag",
        "ccsds_seq_cnt",
        "ccsds_length",
        "sc_tick",
    }

    for field in validation_data.columns:
        if field not in ignore_validation_fields:
            assert field in l1b_hk_dataset.data_vars.keys()
            if field == "leak_i":
                # Compare leak_i arrays
                # Reshape validation_data to match the shape of l1b_hk_dataset
                reshaped_validation_data = np.vstack(validation_data[field].values)
                # Compare leak_i arrays
                np.testing.assert_allclose(
                    l1b_hk_dataset[field].values.astype(np.float64),
                    reshaped_validation_data,
                    atol=1e-2,
                    err_msg=f"Mismatch in {field}",
                )
            elif l1b_hk_dataset[field].dtype.kind == "U":
                np.testing.assert_array_equal(
                    l1b_hk_dataset[field].values,
                    validation_data[field].str.strip().values,
                    err_msg=f"Mismatch in {field}",
                )
            else:
                # Compare float values
                np.testing.assert_allclose(
                    l1b_hk_dataset[field].values.astype(np.float64),
                    validation_data[field].values,
                    atol=1e-2,
                    err_msg=f"Mismatch in {field}",
                )


def test_validate_l1b_standard_rates_data(l1b_standard_rates_dataset):
    """A test to validate the standard rates dataset created by the L1B processing."""

    validation_data = pd.read_csv(
        imap_module_directory
        / "tests/hit/validation_data/hit_l1b_standard_sample2_nsrl_v4_3decimals.csv"
    )

    validation_data = prepare_standard_rates_validation_data(validation_data)

    for field in validation_data.columns:
        assert (
            field in l1b_standard_rates_dataset.data_vars.keys()
        ), f"Field {field} not found in actual data variables"
        for frame in range(validation_data.shape[0]):
            np.testing.assert_allclose(
                l1b_standard_rates_dataset[field][frame].data,
                validation_data[field][frame],
                rtol=1e-7,
                atol=1e-3,
                err_msg=f"Mismatch in {field} at frame {frame}",
            )


def test_hit_l1b_missing_apid(sci_packet_filepath):
    """Test missing housekeeping apid from packet file

    Check that no L1B datasets are created when the housekeeping
    apid is missing from the L0 file path dependency since APIDs
    are currently in separate packet files.

    In the future, all HIT APIDs will be included in the same packet file.

    Parameters
    ----------
    sci_packet_filepath : str
        Science CCSDS packet file path. Only contains science APID and is
        missing the housekeeping APID.
    """
    # Create a dependency dictionary with a science CCSDS packet file
    # excluding the housekeeping apid
    dependency = {"imap_hit_l0_raw": sci_packet_filepath}
    datasets = hit_l1b(dependency, "001")
    assert len(datasets) == 0


def test_hit_l1b(dependencies):
    """Test creating L1B CDF files

    Creates a list of xarray datasets for each L1B product

    Parameters
    ----------
    dependencies : dict
        Dictionary of L1A datasets and CCSDS packet file path
    """
    # TODO: update assertions after science data processing is completed
    datasets = hit_l1b(dependencies, "001")

    assert len(datasets) == 3
    for dataset in datasets:
        assert isinstance(dataset, xr.Dataset)
    assert datasets[0].attrs["Logical_source"] == "imap_hit_l1b_hk"
    assert datasets[1].attrs["Logical_source"] == "imap_hit_l1b_standard-rates"
    assert datasets[2].attrs["Logical_source"] == "imap_hit_l1b_summed-rates"
