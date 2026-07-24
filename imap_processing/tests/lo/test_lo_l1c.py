from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo.l1c.lo_l1c import (
    N_ESA_ENERGY_STEPS,
    N_OFF_ANGLE_BINS,
    N_SPIN_ANGLE_BINS,
    OFF_ANGLE_BIN_CENTERS,
    PSET_SHAPE,
    FilterType,
    calculate_exposure_times,
    compute_pointing_directions,
    create_pset_counts,
    filter_goodtimes,
    lo_l1c,
    set_background_rates,
    set_pointing_directions,
)
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.time import met_to_ttj2000ns


@pytest.fixture
def l1b_de():
    l1b_de = xr.Dataset(
        {
            "spin_bin": ("epoch", [20, 0, 20, 2000, 3500]),
            "off_angle_bin": ("epoch", [20, 20, 20, 20, 20]),
            "esa_step": ("epoch", [1, 2, 1, 4, 5]),
            "coincidence_type": (
                "epoch",
                [
                    "111111",  # golden triple - H
                    "111111",  # golden triple - O (based on TOF)
                    "111000",  # triple
                    "110100",  # double
                    "110000",  # double
                ],
            ),
            # TOF data for species identification
            # Event 0: H (tof0_s=45, tof1_s=30, tof2=25)
            "tof0": ("epoch", [40, 185, 40, 40, 40]),
            "tof1": ("epoch", [40, 105, 40, 40, 40]),
            "tof2": ("epoch", [25, 105, 25, 25, 25]),
            "tof3": ("epoch", [10, 10, 10, 10, 10]),
            "spin_cycle": ("epoch", [1, 2, 3, 4, 5]),
            "avg_spin_durations": ("epoch", [15.2, 15.2, 14.9, 15, 14.9]),
        },
        coords={
            "epoch": [
                7.9794907049e17,
                7.9794907153e17,
                7.9794907254e17,
                7.9794907354e17,
                7.9794907454e17,
            ],
        },
    )
    return l1b_de


@pytest.fixture
def repoint_met():
    met = np.arange(511000000, 511000000 + 86400 * 5, 86400)
    return met


@pytest.fixture
def l1b_de_spin():
    l1b_de = xr.Dataset(
        {
            "spin_bin": ("epoch", [20, 0, 20, 2000, 3500]),
            "off_angle_bin": ("epoch", [20, 20, 20, 20, 20]),
            "esa_step": ("epoch", [1, 2, 1, 4, 5]),
            "coincidence_type": (
                "epoch",
                [
                    "111111",  # golden triple - H
                    "111111",  # golden triple - O (based on TOF)
                    "111000",  # triple
                    "110100",  # double
                    "110000",  # double
                ],
            ),
            # TOF data for species identification
            # Event 0: H (tof0_s=45, tof1_s=30, tof2=25)
            # Event 1: O (tof0_s=185, tof1_s=105, tof2=105)
            "tof0": ("epoch", [40, 185, 40, 40, 40]),
            "tof1": ("epoch", [40, 105, 40, 40, 40]),
            "tof2": ("epoch", [25, 105, 25, 25, 25]),
            "tof3": ("epoch", [10, 10, 10, 10, 10]),
            "spin_cycle": ("epoch", [1, 2, 3, 4, 5]),
            "avg_spin_durations": ("epoch", [15.2, 15.2, 14.9, 15, 14.9]),
            "pivot_angle": ([45.0]),
        },
        coords={
            "epoch": met_to_ttj2000ns(np.arange(511000000, 511000000 + 200, 40) + 902),
        },
        attrs={"Repointing": "repoint00000"},
    )
    return l1b_de


@pytest.fixture
def anc_dependencies():
    anc_dependencies_path = [
        str(
            imap_module_directory
            / "tests/lo/test_anc/imap_lo_good-times-small_20250101_20270101_v001.csv"
        ),
        str(
            imap_module_directory
            / "tests/lo/test_anc/"
            / "imap_lo_hydrogen-background-small_20250101_20270101_v001.csv"
        ),
        str(
            imap_module_directory
            / "tests/lo/test_anc/"
            / "imap_lo_oxygen-background-small_20250101_20270101_v001.csv"
        ),
    ]
    return anc_dependencies_path


@pytest.fixture
def attr_mgr():
    attr_mgr_l1b = ImapCdfAttributes()
    attr_mgr_l1b.add_instrument_global_attrs(instrument="lo")
    attr_mgr_l1b.add_instrument_variable_attrs(instrument="lo", level="l1c")
    return attr_mgr_l1b


@pytest.fixture
def counts():
    """Fixture for initial counts."""
    return np.zeros(PSET_SHAPE)


# ESA Indices are ESA step - 1
@pytest.fixture
def h_counts(counts):
    h = counts.copy()
    # Only event 0 is H (golden triple with H TOF peaks)
    h[0, 0, 20, 20] = 1
    return h


@pytest.fixture
def o_counts(counts):
    o = counts.copy()
    # Only event 1 is O (golden triple with O TOF peaks)
    o[0, 1, 0, 20] = 1
    return o


@pytest.fixture
def triples_counts(counts):
    triples = counts.copy()
    # Events 0, 1 are golden triples (111111), event 2 is regular triple (111000)
    triples[0, 0, 20, 20] = 2  # events 0 and 2 (both esa_step=1)
    triples[0, 1, 0, 20] = 1  # event 1 (esa_step=2)
    return triples


@pytest.fixture
def doubles_counts(counts):
    doubles = counts.copy()
    doubles[0, 3, 2000, 20] = 1
    doubles[0, 4, 3500, 20] = 1
    return doubles


@pytest.fixture
def l1b_bgrates_ds():
    h_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], dtype=np.float32)
    h_var = np.array(
        [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007], dtype=np.float32
    )
    o_rates = np.array(
        [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007], dtype=np.float32
    )
    o_var = np.array(
        [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007], dtype=np.float32
    )
    return xr.Dataset(
        {
            "h_background_rates": (["epoch", "esa_step"], h_rates[np.newaxis, :]),
            "h_background_variance": (["epoch", "esa_step"], h_var[np.newaxis, :]),
            "o_background_rates": (["epoch", "esa_step"], o_rates[np.newaxis, :]),
            "o_background_variance": (["epoch", "esa_step"], o_var[np.newaxis, :]),
        }
    )


@patch("imap_processing.lo.l1c.lo_l1c.calculate_exposure_times")
@patch("imap_processing.lo.l1c.lo_l1c.set_background_rates")
@patch("imap_processing.lo.l1c.lo_l1c.filter_goodtimes")
@patch("imap_processing.lo.l1c.lo_l1c.set_pointing_directions")
@patch("imap_processing.lo.l1c.lo_l1c.add_spacecraft_position_and_velocity_to_pset")
def test_lo_l1c(
    mock_add_spacecraft_position_and_velocity_to_pset,
    mock_set_pointing_directions,
    mock_filter_goodtimes,
    mock_set_background_rates,
    mock_calculate_exposure_times,
    l1b_de_spin,
    anc_dependencies,
    use_fake_repoint_data_for_time,
    use_fake_spin_data_for_time,
    repoint_met,
):
    # Arrange
    repoint_start_met = 511000000.0
    repoint_stride_seconds = 86400  # 1 day stride
    data = {
        "imap_lo_l1b_de": l1b_de_spin,
        "imap_lo_l1b_goodtimes": xr.Dataset(
            {
                "gt_start_met": ("epoch", [repoint_start_met]),
                "gt_end_met": ("epoch", [repoint_start_met]),
                "pivot": ([45.0]),
                "pivot_de": ([45.0]),
            },
            coords={"epoch": met_to_ttj2000ns([repoint_start_met])},
        ),
        "imap_lo_l1b_histrates": xr.Dataset(
            {
                "exposure_time_6deg": (
                    ["epoch", "esa_step", "spin_bin_6"],
                    np.zeros((1, N_ESA_ENERGY_STEPS, 60), dtype=np.float32),
                )
            },
            coords={"epoch": met_to_ttj2000ns([repoint_start_met])},
        ),
    }
    use_fake_spin_data_for_time(repoint_start_met)
    use_fake_repoint_data_for_time(
        np.arange(
            repoint_start_met,
            repoint_start_met + repoint_stride_seconds * 5,
            repoint_stride_seconds,
        )
    )
    mock_set_background_rates.return_value = (None, None, None)
    mock_filter_goodtimes.return_value = l1b_de_spin
    mock_set_pointing_directions.return_value = (
        xr.DataArray(np.zeros((3600, 40)), dims=("spin_angle", "off_angle")),
        xr.DataArray(np.zeros((3600, 40)), dims=("spin_angle", "off_angle")),
    )
    # Mock exposure time calculation to avoid SPICE calls
    mock_calculate_exposure_times.return_value = xr.DataArray(
        np.ones(PSET_SHAPE, dtype=np.float32),
        dims=["epoch", "esa_energy_step", "spin_angle", "off_angle"],
    )

    # Pass through the pset with sc_position and sc_velocity added
    def mock_add_sc_pos_vel(pset):
        pset["sc_position"] = xr.DataArray(np.zeros(3), dims=["x_y_z"])
        pset["sc_velocity"] = xr.DataArray(np.zeros(3), dims=["x_y_z"])
        return pset

    mock_add_spacecraft_position_and_velocity_to_pset.side_effect = mock_add_sc_pos_vel
    expected_logical_source = "imap_lo_l1c_pset"

    # Act
    output_dataset = lo_l1c(data, anc_dependencies)[0]

    # Assert
    assert expected_logical_source == output_dataset.attrs["Logical_source"]
    # Verify that pivot_angle is passed through from l1b_goodtimes
    assert "pivot_angle" in output_dataset
    assert output_dataset["pivot_angle"].values[0] == 45.0
    mock_add_spacecraft_position_and_velocity_to_pset.assert_called_once()
    assert "sc_position" in output_dataset
    assert "sc_velocity" in output_dataset
    # Verify that set_pointing_directions uses pointing midpoint
    # Repoint table starts at 511000000, has a 15-minute repoint followed by
    # a 24-hour - 15-minute pointing. So, pointing midpoint is:
    expected_pointing_midpoint = (
        repoint_start_met + 15 * 60 + repoint_start_met + repoint_stride_seconds
    ) / 2
    np.testing.assert_almost_equal(
        mock_set_pointing_directions.call_args[0][0],
        met_to_ttj2000ns(expected_pointing_midpoint),
    )


def test_filter_goodtimes():
    # Arrange
    event_mets = [473389199, 473389200, 473389201, 473389202, 473389203, 473407619]
    l1b_de_all = xr.Dataset(
        {
            "esa_step": ("epoch", [1, 2, 1, 4, 5, 2]),
            "spin_bin": ("epoch", [1900, 2000, 3000, 3000, 3000, 3000]),
        },
        coords={"epoch": met_to_ttj2000ns(event_mets)},
    )

    # Two goodtime windows: [473389201, 473389203] and [473407619, 473407620]
    gt_starts = np.array([473389201.0, 473407619.0])
    goodtimes_ds = xr.Dataset(
        {
            "gt_start_met": ("epoch", gt_starts),
            "gt_end_met": ("epoch", [473389203.0, 473407620.0]),
        },
        coords={"epoch": met_to_ttj2000ns(gt_starts)},
    )

    # Events at MET 473389201-473389203 and 473407619 fall inside goodtime windows;
    # 473389199 and 473389200 are before the first window.
    expected_mask = [False, False, True, True, True, True]
    expected = l1b_de_all.isel(epoch=expected_mask)

    # Act
    result = filter_goodtimes(l1b_de_all, goodtimes_ds)

    # Assert
    xr.testing.assert_equal(result, expected)


@patch("imap_processing.lo.l1c.lo_l1c.calculate_exposure_times")
@patch("imap_processing.lo.l1c.lo_l1c.set_pointing_directions")
@patch("imap_processing.lo.l1c.lo_l1c.add_spacecraft_position_and_velocity_to_pset")
def test_lo_l1c_no_goodtimes(
    mock_add_spacecraft_position_and_velocity_to_pset,
    mock_set_pointing_directions,
    mock_calculate_exposure_times,
    l1b_de_spin,
    anc_dependencies,
    use_fake_repoint_data_for_time,
    use_fake_spin_data_for_time,
    repoint_met,
):
    # Arrange
    # Goodtime window [511000000, 511000900] is within the repoint period
    # but before all events (which start at 511000902)
    goodtime_start = 511000000.0
    goodtime_end = 511000900.0
    data = {
        "imap_lo_l1b_de": l1b_de_spin,
        "imap_lo_l1b_goodtimes": xr.Dataset(
            {
                "gt_start_met": ("epoch", [goodtime_start]),
                "gt_end_met": ("epoch", [goodtime_end]),
                "pivot": ([45.0]),
                "pivot_de": ([45.0]),
            },
            coords={"epoch": met_to_ttj2000ns([goodtime_start])},
        ),
        "imap_lo_l1b_histrates": xr.Dataset(
            {
                "exposure_time_6deg": (
                    ["epoch", "esa_step", "spin_bin_6"],
                    np.zeros((1, N_ESA_ENERGY_STEPS, 60), dtype=np.float32),
                )
            },
            coords={"epoch": met_to_ttj2000ns([goodtime_start])},
        ),
    }
    use_fake_spin_data_for_time(511000000)
    use_fake_repoint_data_for_time(np.arange(511000000, 511000000 + 86400 * 5, 86400))
    expected_logical_source = "imap_lo_l1c_pset"

    # Mock exposure time calculation to return zeros (no events in goodtimes)
    mock_calculate_exposure_times.return_value = xr.DataArray(
        np.zeros(PSET_SHAPE, dtype=np.float32),
        dims=["epoch", "esa_energy_step", "spin_angle", "off_angle"],
    )

    # Mock pointing directions to return valid non-zero values
    mock_set_pointing_directions.return_value = (
        xr.DataArray(
            np.ones((1, 3600, 40)) * 180.0,
            dims=["epoch", "spin_angle", "off_angle"],
        ),
        xr.DataArray(
            np.ones((1, 3600, 40)) * 45.0,
            dims=["epoch", "spin_angle", "off_angle"],
        ),
    )

    # Mock spacecraft position/velocity
    def mock_add_sc_pos_vel(pset):
        pset["sc_position"] = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=["x_y_z"])
        pset["sc_velocity"] = xr.DataArray(np.array([0.1, 0.2, 0.3]), dims=["x_y_z"])
        return pset

    mock_add_spacecraft_position_and_velocity_to_pset.side_effect = mock_add_sc_pos_vel

    # Act
    output_dataset = lo_l1c(data, anc_dependencies)[0]

    # Assert
    assert expected_logical_source == output_dataset.attrs["Logical_source"]
    # Verify that pivot_angle is passed through from l1b_goodtimes
    assert "pivot_angle" in output_dataset
    assert output_dataset["pivot_angle"].values[0] == 45.0

    # Verify that times are valid (not zeros/junk)
    # Pointing start is repoint start (511000000) + 15 minutes (900 seconds)
    assert output_dataset["pointing_start_met"].values[0] == 511000900.0
    assert (
        output_dataset["pointing_end_met"].values[0]
        > output_dataset["pointing_start_met"].values[0]
    )

    # Verify counts are zeros (no events in goodtimes window)
    expected_counts = np.zeros((1, 7, 3600, 40))
    np.testing.assert_array_equal(output_dataset["h_counts"], expected_counts)
    np.testing.assert_array_equal(output_dataset["o_counts"], expected_counts)
    np.testing.assert_array_equal(output_dataset["doubles_counts"], expected_counts)
    np.testing.assert_array_equal(output_dataset["triples_counts"], expected_counts)

    # Verify exposure times are zeros (mocked)
    np.testing.assert_array_equal(output_dataset["exposure_time"], expected_counts)

    # Verify background rates are zeros (no bgrates dependency provided)
    np.testing.assert_array_equal(output_dataset["h_background_rates"], expected_counts)
    np.testing.assert_array_equal(output_dataset["o_background_rates"], expected_counts)

    # Verify geometry is computed (not zeros) - mocked to return valid values
    assert "hae_latitude" in output_dataset
    assert "hae_longitude" in output_dataset
    # HAE values should be the mocked non-zero values
    np.testing.assert_array_equal(
        output_dataset["hae_longitude"].values, np.ones((1, 3600, 40)) * 180.0
    )
    np.testing.assert_array_equal(
        output_dataset["hae_latitude"].values, np.ones((1, 3600, 40)) * 45.0
    )

    # Verify spacecraft position/velocity are valid (not zeros)
    assert "sc_position" in output_dataset
    assert "sc_velocity" in output_dataset
    np.testing.assert_array_equal(output_dataset["sc_position"].values, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(output_dataset["sc_velocity"].values, [0.1, 0.2, 0.3])


def test_create_pset_counts(l1b_de):
    # Arrange
    expected_counts = np.zeros((1, 7, 3600, 40))
    # ESA Indices are ESA step - 1
    # Events 0 and 2 have esa_step=1, bin 20
    expected_counts[0, 0, 20, 20] = 2
    # Event 1 has esa_step=2, bin 0
    expected_counts[0, 1, 0, 20] = 1
    # Event 3 has esa_step=4, bin 2000
    expected_counts[0, 3, 2000, 20] = 1
    # Event 4 has esa_step=5, bin 3500
    expected_counts[0, 4, 3500, 20] = 1

    # Act
    counts = create_pset_counts(l1b_de)

    # Assert
    np.testing.assert_array_equal(counts, expected_counts)


def test_create_h_pset_counts(l1b_de, h_counts):
    # Act
    counts = create_pset_counts(l1b_de, FilterType.HYDROGEN)

    # Assert
    np.testing.assert_array_equal(counts, h_counts)


def test_create_o_pset_counts(l1b_de, o_counts):
    # Act
    counts = create_pset_counts(l1b_de, FilterType.OXYGEN)

    # Assert
    np.testing.assert_array_equal(counts, o_counts)


def test_create_triples_pset_counts(l1b_de, triples_counts):
    # Act
    counts = create_pset_counts(l1b_de, FilterType.TRIPLES)

    # Assert
    np.testing.assert_array_equal(counts, triples_counts)


def test_create_doubles_pset_counts(l1b_de, doubles_counts):
    # Act
    counts = create_pset_counts(l1b_de, FilterType.DOUBLES)

    # Assert
    np.testing.assert_array_equal(counts, doubles_counts)


def test_calculate_exposure_times():
    """Test the histrates-based exposure time calculation."""
    gt_start_met = 500000000.0
    gt_end_met = 500001000.0
    n_epochs = 3

    epoch_mets = np.linspace(gt_start_met, gt_end_met, n_epochs)
    exposure_values = np.ones((n_epochs, N_ESA_ENERGY_STEPS, 60), dtype=np.float32)

    histrates_ds = xr.Dataset(
        {"exposure_time_6deg": (["epoch", "esa_step", "spin_bin_6"], exposure_values)},
        coords={"epoch": met_to_ttj2000ns(epoch_mets)},
    )
    goodtimes_ds = xr.Dataset(
        {
            "gt_start_met": ("epoch", [gt_start_met]),
            "gt_end_met": ("epoch", [gt_end_met]),
        },
        coords={"epoch": met_to_ttj2000ns([gt_start_met])},
    )

    exposure_times = calculate_exposure_times(histrates_ds, goodtimes_ds)

    assert exposure_times.shape == PSET_SHAPE
    # Each of the 3 epochs contributes 1.0 per (esa, spin_6) cell.
    # Expanding 60 → 3600: divide by 60 per cell; distributing across 40 off-angle bins:
    # divide by N_OFF_ANGLE_BINS.
    expected_per_cell = float(n_epochs) / 60 / N_OFF_ANGLE_BINS
    np.testing.assert_allclose(exposure_times.values, expected_per_cell, rtol=1e-5)
    # All ESA steps should have equal exposure (uniform input)
    for i in range(1, N_ESA_ENERGY_STEPS):
        np.testing.assert_array_equal(
            exposure_times.values[0, 0, :, :], exposure_times.values[0, i, :, :]
        )


def test_calculate_exposure_times_no_goodtimes_overlap():
    """Test that exposure times are zero when goodtimes don't overlap with histrates."""
    epoch_met = 500000000.0
    exposure_values = np.ones((1, N_ESA_ENERGY_STEPS, 60), dtype=np.float32)

    histrates_ds = xr.Dataset(
        {"exposure_time_6deg": (["epoch", "esa_step", "spin_bin_6"], exposure_values)},
        coords={"epoch": met_to_ttj2000ns([epoch_met])},
    )
    goodtimes_ds = xr.Dataset(
        {
            "gt_start_met": ("epoch", [400000000.0]),
            "gt_end_met": ("epoch", [400001000.0]),
        },
        coords={"epoch": met_to_ttj2000ns([400000000.0])},
    )

    exposure_times = calculate_exposure_times(histrates_ds, goodtimes_ds)

    assert exposure_times.shape == PSET_SHAPE
    np.testing.assert_array_equal(exposure_times.values, 0.0)


@pytest.mark.parametrize("species", [FilterType.HYDROGEN, FilterType.OXYGEN])
def test_set_background_rates(l1b_bgrates_ds, attr_mgr, species):
    # Arrange
    sci_deps = {"imap_lo_l1b_bgrates": l1b_bgrates_ds}
    species_key = species.value
    expected_rates = l1b_bgrates_ds[f"{species_key}_background_rates"].values[0]
    expected_var = l1b_bgrates_ds[f"{species_key}_background_variance"].values[0]

    # Act
    rates, uncert, err = set_background_rates(species, sci_deps, attr_mgr)

    # Assert shape
    assert rates.shape == (1, N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS)
    assert uncert.shape == (1, N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS)
    assert err.shape == (1, N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS)

    # Rates and uncertainties must be uniform across spatial bins for each ESA step
    for i in range(N_ESA_ENERGY_STEPS):
        np.testing.assert_array_equal(
            rates.values[0, i, :, :],
            np.full(
                (N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS),
                expected_rates[i],
                dtype=np.float16,
            ),
        )
        np.testing.assert_array_equal(
            uncert.values[0, i, :, :],
            np.full(
                (N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS), expected_var[i], dtype=np.float16
            ),
        )

    # Systematic error is always zero
    np.testing.assert_array_equal(err.values, 0)


def test_set_background_rates_no_bgrates(attr_mgr):
    """Returns zeros when imap_lo_l1b_bgrates is absent from sci_dependencies."""
    rates, uncert, err = set_background_rates(FilterType.HYDROGEN, {}, attr_mgr)

    np.testing.assert_array_equal(rates.values, 0)
    np.testing.assert_array_equal(uncert.values, 0)
    np.testing.assert_array_equal(err.values, 0)


def test_set_background_rates_species_error(attr_mgr):
    # Act
    with pytest.raises(
        ValueError, match="Species must be 'h' or 'o', but got doubles."
    ):
        set_background_rates(FilterType.DOUBLES, {}, attr_mgr)


def test_set_pointing_directions(attr_mgr):
    """Test the set_pointing_directions function."""
    # Mock the external dependencies
    mock_et = 123456789.0
    mock_hae_az_el = np.stack(
        np.meshgrid(np.arange(3600), np.arange(40), indexing="ij"), axis=-1
    )  # spin_angle x off_angle x 2
    with (
        patch("imap_processing.lo.l1c.lo_l1c.ttj2000ns_to_et") as mock_ttj2000ns_to_et,
        patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el"
        ) as mock_frame_transform,
    ):
        # Set up mocks
        mock_ttj2000ns_to_et.return_value = mock_et
        mock_frame_transform.return_value = mock_hae_az_el

        # Test input
        test_epoch = 1000000000.0

        # Call the function
        hae_longitude, hae_latitude = set_pointing_directions(test_epoch, attr_mgr, 90)

        # Verify ttj2000ns_to_et was called correctly
        mock_ttj2000ns_to_et.assert_called_once_with(test_epoch)

        # Verify frame_transform_az_el was called correctly
        mock_frame_transform.assert_called_once()
        call_args = mock_frame_transform.call_args
        assert call_args[0][0] == mock_et  # et parameter
        assert call_args[1]["degrees"] is True
        # Verify the shape of dps_az_el
        dps_az_el = call_args[0][1]
        assert dps_az_el.shape == (3600, 40, 2)  # spin_angle x off_angle x 2

        # Verify the returned DataArrays
        assert isinstance(hae_longitude, xr.DataArray)
        assert isinstance(hae_latitude, xr.DataArray)

        # Check dimensions
        assert hae_longitude.dims == ("epoch", "spin_angle", "off_angle")
        assert hae_latitude.dims == ("epoch", "spin_angle", "off_angle")

        # Check shapes
        assert hae_longitude.shape == (1, 3600, 40)
        assert hae_latitude.shape == (1, 3600, 40)

        # Check data types
        assert hae_longitude.dtype == np.float64
        assert hae_latitude.dtype == np.float64

        # Check that longitude uses first component (index 0)
        # and latitude uses second (index 1)
        # Note: Compare with the added epoch dimension [0]
        np.testing.assert_array_equal(hae_longitude.values[0], mock_hae_az_el[:, :, 0])
        np.testing.assert_array_equal(hae_latitude.values[0], mock_hae_az_el[:, :, 1])


def test_set_pointing_directions_meshgrid(attr_mgr):
    """Test that the meshgrid is created correctly."""
    with (
        patch("imap_processing.lo.l1c.lo_l1c.ttj2000ns_to_et") as mock_ttj2000ns_to_et,
        patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el"
        ) as mock_frame_transform,
    ):
        mock_ttj2000ns_to_et.return_value = 123456789.0
        mock_hae_az_el = np.stack(
            np.meshgrid(np.arange(3600), np.arange(40), indexing="ij"), axis=-1
        )  # spin_angle x off_angle x 2
        mock_frame_transform.return_value = mock_hae_az_el

        set_pointing_directions(1000000000.0, attr_mgr, 90)

        # Get the dps_az_el array that was passed to frame_transform_az_el
        call_args = mock_frame_transform.call_args
        dps_az_el = call_args[0][1]

        # Verify the meshgrid was created correctly
        # The first component should be spin angles repeated for each off angle
        expected_spin_shape = (3600, 40)
        assert dps_az_el[:, :, 0].shape == expected_spin_shape

        # The second component should be off angles repeated for each spin angle
        assert dps_az_el[:, :, 1].shape == expected_spin_shape

        # Check that spin angles vary along the first dimension
        assert not np.allclose(dps_az_el[0, 0, 0], dps_az_el[1, 0, 0])

        # Check that off angles vary along the second dimension
        assert not np.allclose(dps_az_el[0, 0, 1], dps_az_el[0, 1, 1])


@pytest.mark.parametrize("pivot_angle", [75, 90, 105])
def test_set_pointing_directions_pivot_angle(attr_mgr, pivot_angle):
    """Test that pivot_angle correctly adjusts off_angles before transformation."""
    with (
        patch("imap_processing.lo.l1c.lo_l1c.ttj2000ns_to_et") as mock_ttj2000ns_to_et,
        patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el"
        ) as mock_frame_transform,
    ):
        mock_ttj2000ns_to_et.return_value = 123456789.0
        mock_hae_az_el = np.stack(
            np.meshgrid(np.arange(3600), np.arange(40), indexing="ij"), axis=-1
        )
        mock_frame_transform.return_value = mock_hae_az_el

        set_pointing_directions(1000000000.0, attr_mgr, pivot_angle=pivot_angle)

        # Get the dps_az_el array that was passed to frame_transform_az_el
        call_args = mock_frame_transform.call_args
        dps_az_el = call_args[0][1]

        # Calculate expected offset: off_angles should be adjusted by (90 - pivot_angle)
        offset = 90 - pivot_angle

        # OFF_ANGLE_BIN_CENTERS range from -1.95 to 1.95 (40 bins from -2 to 2)
        # After offset, they should be shifted by the offset amount
        expected_off_angles = OFF_ANGLE_BIN_CENTERS + offset

        # Check that the off_angle component (index 1) was adjusted correctly
        # dps_az_el[:, :, 1] should have the adjusted off angles repeated across spin
        actual_off_angles = dps_az_el[0, :, 1]  # Take first spin angle
        np.testing.assert_allclose(actual_off_angles, expected_off_angles, rtol=1e-10)


def test_compute_pointing_directions_defaults():
    """Default grid/frame reproduce the PSET (3600 x 40) IMAP_DPS->IMAP_HAE case."""
    mock_et = 123456789.0
    mock_az_el = np.stack(
        np.meshgrid(np.arange(3600), np.arange(40), indexing="ij"), axis=-1
    )
    with (
        patch("imap_processing.lo.l1c.lo_l1c.ttj2000ns_to_et") as mock_ttj2000ns_to_et,
        patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el"
        ) as mock_frame_transform,
    ):
        mock_ttj2000ns_to_et.return_value = mock_et
        mock_frame_transform.return_value = mock_az_el

        result = compute_pointing_directions(1000000000.0, 90)

        # Returns the raw (n_spin, n_off, 2) array, not a DataArray.
        assert result.shape == (3600, 40, 2)
        call_args = mock_frame_transform.call_args
        assert call_args[0][1].shape == (3600, 40, 2)  # dps_az_el grid
        assert call_args[0][2] == SpiceFrame.IMAP_DPS  # from_frame
        assert call_args[0][3] == SpiceFrame.IMAP_HAE  # default to_frame


def test_compute_pointing_directions_custom_grid_and_frame():
    """Custom spin/off angles and destination frame are honored."""
    spin_angles = np.arange(3.0, 360.0, 6.0)  # 60 bins
    off_angles = np.array([0.0])  # single pivot-cone off-angle
    pivot_angle = 75.0
    with (
        patch("imap_processing.lo.l1c.lo_l1c.ttj2000ns_to_et") as mock_ttj2000ns_to_et,
        patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el"
        ) as mock_frame_transform,
    ):
        mock_ttj2000ns_to_et.return_value = 123456789.0
        mock_frame_transform.side_effect = lambda et, az_el, *a, **k: az_el

        result = compute_pointing_directions(
            1000000000.0,
            pivot_angle,
            spin_angles=spin_angles,
            off_angles=off_angles,
            to_frame=SpiceFrame.ECLIPJ2000,
        )

        assert result.shape == (60, 1, 2)
        # Spin component matches the requested spin angles.
        np.testing.assert_allclose(result[:, 0, 0], spin_angles)
        # Off component is the single off-angle offset by (90 - pivot_angle).
        np.testing.assert_allclose(result[:, 0, 1], 90 - pivot_angle)
        # Destination frame is forwarded.
        assert mock_frame_transform.call_args[0][3] == SpiceFrame.ECLIPJ2000


def test_set_pointing_directions_delegates(attr_mgr):
    """set_pointing_directions wraps compute_pointing_directions output unchanged."""
    mock_az_el = np.stack(
        np.meshgrid(np.arange(3600), np.arange(40), indexing="ij"), axis=-1
    ).astype(float)
    with patch(
        "imap_processing.lo.l1c.lo_l1c.compute_pointing_directions"
    ) as mock_compute:
        mock_compute.return_value = mock_az_el

        hae_longitude, hae_latitude = set_pointing_directions(
            1000000000.0, attr_mgr, 90
        )

        mock_compute.assert_called_once_with(1000000000.0, 90)
        assert hae_longitude.dims == ("epoch", "spin_angle", "off_angle")
        assert hae_longitude.shape == (1, 3600, 40)
        np.testing.assert_array_equal(hae_longitude.values[0], mock_az_el[:, :, 0])
        np.testing.assert_array_equal(hae_latitude.values[0], mock_az_el[:, :, 1])
