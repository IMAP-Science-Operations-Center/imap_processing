import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.l1a import hit_l1a
from imap_processing.hit.l1b.hit_l1b import (
    SUMMED_PARTICLE_ENERGY_RANGE_MAPPING,
    calculate_rates,
    hit_l1b,
    process_sectored_rates_data,
    process_standard_rates_data,
    process_summed_rates_data,
    subset_data_for_sectored_counts,
    sum_livetime_10min,
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


@pytest.fixture
def dependencies(packet_filepath, sci_packet_filepath):
    """Get dependencies for L1B processing"""
    # Create dictionary of dependencies and add CCSDS packet file
    data_dict = {"imap_hit_l0_raw": packet_filepath}
    # Add L1A datasets
    l1a_datasets = hit_l1a.hit_l1a(packet_filepath)
    # TODO: Remove this when HIT provides a packet file with all apids.
    l1a_datasets.extend(hit_l1a.hit_l1a(sci_packet_filepath))
    for dataset in l1a_datasets:
        data_dict[dataset.attrs["Logical_source"]] = dataset
    return data_dict


@pytest.fixture
def l1b_hk_dataset(dependencies):
    """Get the housekeeping dataset"""
    datasets = hit_l1b(dependencies)
    for dataset in datasets:
        if dataset.attrs["Logical_source"] == "imap_hit_l1b_hk":
            return dataset


@pytest.fixture
def l1b_standard_rates_dataset(dependencies):
    """Get the standard rates dataset"""
    datasets = hit_l1b(dependencies)
    for dataset in datasets:
        if dataset.attrs["Logical_source"] == "imap_hit_l1b_standard-rates":
            return dataset


@pytest.fixture
def l1a_counts_dataset(sci_packet_filepath):
    """Get L1A counts dataset to test l1b processing functions"""
    l1a_datasets = hit_l1a.hit_l1a(sci_packet_filepath)
    for dataset in l1a_datasets:
        if dataset.attrs["Logical_source"] == "imap_hit_l1a_counts":
            return dataset


@pytest.fixture
def livetime(l1a_counts_dataset: xr.Dataset) -> xr.DataArray:
    """Calculate livetime for L1A counts dataset"""
    return xr.DataArray(l1a_counts_dataset["livetime_counter"] / 270)


def test_calculate_rates():
    """Test the calculate_rates function"""

    # Create a sample dataset
    data = {
        "counts": (("epoch",), np.array([100, 200, 300], dtype=np.float32)),
        "counts_stat_uncert_minus": (
            ("epoch",),
            np.array([10, 20, 30], dtype=np.float32),
        ),
        "counts_stat_uncert_plus": (
            ("epoch",),
            np.array([15, 25, 35], dtype=np.float32),
        ),
    }
    coords = {"epoch": np.array([0, 1, 2], dtype=np.float32)}
    dataset = xr.Dataset(data, coords=coords)

    # Create a sample livetime array
    livetime = xr.DataArray(np.array([10, 20, 30], dtype=np.float32), dims="epoch")

    # Call the function
    result = calculate_rates(dataset, "counts", livetime)

    # Check the results
    expected_counts = np.array([10, 10, 10], dtype=np.float32)
    expected_counts_uncert_minus = np.array([1, 1, 1], dtype=np.float32)
    expected_counts_uncert_plus = np.array([1.5, 1.25, 1.1666666], dtype=np.float32)

    np.testing.assert_allclose(result["counts"].values, expected_counts)
    np.testing.assert_allclose(
        result["counts_stat_uncert_minus"].values, expected_counts_uncert_minus
    )
    np.testing.assert_allclose(
        result["counts_stat_uncert_plus"].values, expected_counts_uncert_plus
    )


def test_sum_livetime_10min():
    """Test the sum_livetime_10min function."""
    # Create a sample livetime DataArray
    livetime_values = np.arange(1, 31)  # 30 epochs with values 1 to 30
    livetime = xr.DataArray(
        livetime_values, dims=["epoch"], coords={"epoch": np.arange(30)}
    )

    # Expected result: sum of every 10 values repeated 10 times
    expected_values = np.repeat(
        [sum(livetime_values[i : i + 10]) for i in range(0, 30, 10)], 10
    )
    expected_livetime = xr.DataArray(
        expected_values, dims=["epoch"], coords={"epoch": np.arange(30)}
    )

    # Call the function
    result = sum_livetime_10min(livetime)

    # Assert the result is as expected
    xr.testing.assert_equal(result, expected_livetime)


def test_subset_data_for_sectored_counts():
    """Test the subset_data_for_sectored_counts function."""
    # Create a sample L1A counts dataset
    l1a_counts_dataset = xr.Dataset(
        {
            "hdr_minute_cnt": ("epoch", np.arange(105, 135)),
            "h_sectored_counts": ("epoch", np.arange(0, 30)),
            "he4_sectored_counts": ("epoch", np.arange(0, 30)),
        },
    )

    # Create a sample livetime data array
    livetime = xr.DataArray(np.arange(1.0, 31.0, dtype=np.float32), dims=["epoch"])

    # Call the function
    subset_dataset, subset_livetime = subset_data_for_sectored_counts(
        l1a_counts_dataset, livetime
    )

    # Check the results
    assert subset_dataset.dims["epoch"] == 10
    assert len(subset_livetime["epoch"]) == 10
    assert np.all(subset_dataset["hdr_minute_cnt"].values % 10 == np.arange(10))


def test_process_summed_rates_data(l1a_counts_dataset, livetime):
    """Test the variables in the summed rates dataset"""

    l1b_summed_rates_dataset = process_summed_rates_data(l1a_counts_dataset, livetime)

    # Check that a xarray dataset is returned
    assert isinstance(l1b_summed_rates_dataset, xr.Dataset)

    valid_coords = {
        "epoch",
        "h_energy_mean",
        "he3_energy_mean",
        "he4_energy_mean",
        "he_energy_mean",
        "c_energy_mean",
        "o_energy_mean",
        "fe_energy_mean",
        "n_energy_mean",
        "si_energy_mean",
        "mg_energy_mean",
        "s_energy_mean",
        "ar_energy_mean",
        "ca_energy_mean",
        "na_energy_mean",
        "al_energy_mean",
        "ne_energy_mean",
        "ni_energy_mean",
    }

    # Check that the dataset has the correct coords and variables
    assert valid_coords == set(l1b_summed_rates_dataset.coords), "Coordinates mismatch"

    assert "dynamic_threshold_state" in l1b_summed_rates_dataset.data_vars

    for particle in SUMMED_PARTICLE_ENERGY_RANGE_MAPPING.keys():
        assert f"{particle}" in l1b_summed_rates_dataset.data_vars
        assert f"{particle}_stat_uncert_minus" in l1b_summed_rates_dataset.data_vars
        assert f"{particle}_stat_uncert_plus" in l1b_summed_rates_dataset.data_vars
        assert f"{particle}_energy_delta_minus" in l1b_summed_rates_dataset.data_vars
        assert f"{particle}_energy_delta_plus" in l1b_summed_rates_dataset.data_vars


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
        "sngrates_stat_uncert_plus",
        "coinrates_stat_uncert_plus",
        "pbufrates_stat_uncert_plus",
        "l2fgrates_stat_uncert_plus",
        "l2bgrates_stat_uncert_plus",
        "l3fgrates_stat_uncert_plus",
        "l3bgrates_stat_uncert_plus",
        "penfgrates_stat_uncert_plus",
        "penbgrates_stat_uncert_plus",
        "ialirtrates_stat_uncert_plus",
        "l4fgrates_stat_uncert_plus",
        "l4bgrates_stat_uncert_plus",
        "sngrates_stat_uncert_minus",
        "coinrates_stat_uncert_minus",
        "pbufrates_stat_uncert_minus",
        "l2fgrates_stat_uncert_minus",
        "l2bgrates_stat_uncert_minus",
        "l3fgrates_stat_uncert_minus",
        "l3bgrates_stat_uncert_minus",
        "penfgrates_stat_uncert_minus",
        "penbgrates_stat_uncert_minus",
        "ialirtrates_stat_uncert_minus",
        "l4fgrates_stat_uncert_minus",
        "l4bgrates_stat_uncert_minus",
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
    assert valid_data_vars == set(l1b_standard_rates_dataset.data_vars.keys()), (
        "Data variables mismatch"
    )
    assert valid_coords == list(l1b_standard_rates_dataset.coords), (
        "Coordinates mismatch"
    )


def test_process_sectored_rates_data(l1a_counts_dataset, livetime):
    """Test the variables in the sectored rates dataset"""

    l1b_sectored_rates_dataset = process_sectored_rates_data(
        l1a_counts_dataset, livetime
    )

    # Check that a xarray dataset is returned
    assert isinstance(l1b_sectored_rates_dataset, xr.Dataset)

    valid_coords = {
        "epoch",
        "declination",
        "azimuth",
        "h_energy_mean",
        "he4_energy_mean",
        "cno_energy_mean",
        "nemgsi_energy_mean",
        "fe_energy_mean",
    }

    # Check that the dataset has the correct coords and variables
    assert valid_coords == set(l1b_sectored_rates_dataset.coords), (
        "Coordinates mismatch"
    )

    assert "dynamic_threshold_state" in l1b_sectored_rates_dataset.data_vars

    particles = ["h", "he4", "cno", "nemgsi", "fe"]
    for particle in particles:
        assert f"{particle}" in l1b_sectored_rates_dataset.data_vars
        assert f"{particle}_stat_uncert_minus" in l1b_sectored_rates_dataset.data_vars
        assert f"{particle}_stat_uncert_plus" in l1b_sectored_rates_dataset.data_vars
        assert f"{particle}_energy_delta_minus" in l1b_sectored_rates_dataset.data_vars
        assert f"{particle}_energy_delta_plus" in l1b_sectored_rates_dataset.data_vars


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
        assert field in l1b_standard_rates_dataset.data_vars.keys(), (
            f"Field {field} not found in actual data variables"
        )
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
    datasets = hit_l1b(dependency)
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
    datasets = hit_l1b(dependencies)

    assert len(datasets) == 4
    for dataset in datasets:
        assert isinstance(dataset, xr.Dataset)
    assert datasets[0].attrs["Logical_source"] == "imap_hit_l1b_hk"
    assert datasets[1].attrs["Logical_source"] == "imap_hit_l1b_standard-rates"
    assert datasets[2].attrs["Logical_source"] == "imap_hit_l1b_summed-rates"
    assert datasets[3].attrs["Logical_source"] == "imap_hit_l1b_sectored-rates"
