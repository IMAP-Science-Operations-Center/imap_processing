from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo.l1c.lo_l1c import (
    N_ESA_ENERGY_STEPS,
    N_OFF_ANGLE_BINS,
    N_SAMPLES_PER_SPIN,
    N_SPIN_ANGLE_BINS,
    OFF_ANGLE_BIN_CENTERS,
    PSET_SHAPE,
    FilterType,
    calculate_bin_weights,
    calculate_exposure_times,
    create_goodtimes_fraction,
    create_pset_counts,
    filter_goodtimes,
    get_representative_spin_times,
    lo_l1c,
    sample_boresight_bins,
    set_background_rates,
    set_pointing_directions,
)
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
            "h_background_rates": ("esa_step", h_rates),
            "h_background_variance": ("esa_step", h_var),
            "o_background_rates": ("esa_step", o_rates),
            "o_background_variance": ("esa_step", o_var),
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
    data = {
        "imap_lo_l1b_de": l1b_de_spin,
        "imap_lo_l1b_goodtimes": xr.Dataset(
            {
                "gt_start_met": ("epoch", [511000000.0]),
                "gt_end_met": ("epoch", [511100000.0]),
            },
            coords={"epoch": met_to_ttj2000ns([511000000.0])},
        ),
    }
    use_fake_spin_data_for_time(511000000)
    use_fake_repoint_data_for_time(np.arange(511000000, 511000000 + 86400 * 5, 86400))
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
    # Verify that pivot_angle is passed through from l1b_de
    assert "pivot_angle" in output_dataset
    assert output_dataset["pivot_angle"].values[0] == 45.0
    mock_add_spacecraft_position_and_velocity_to_pset.assert_called_once()
    assert "sc_position" in output_dataset
    assert "sc_velocity" in output_dataset


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
    # Verify that pivot_angle is passed through from l1b_de
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


def test_calculate_exposure_times(use_fake_spin_data_for_time):
    """Test the statistical exposure time calculation."""
    # Arrange
    pointing_start_met = 511000000.0
    pointing_end_met = 511000100.0  # 100 second pointing
    use_fake_spin_data_for_time(pointing_start_met)

    with (
        patch(
            "imap_processing.lo.l1c.lo_l1c.lo_instrument_pointing"
        ) as mock_lo_instrument_pointing,
        patch(
            "imap_processing.lo.l1c.lo_l1c.met_to_ttj2000ns"
        ) as mock_met_to_ttj2000ns,
        patch("imap_processing.lo.l1c.lo_l1c.ttj2000ns_to_et") as mock_ttj2000ns_to_et,
    ):
        # Mock the time conversions to pass through
        mock_met_to_ttj2000ns.side_effect = lambda x: x * 1e9
        mock_ttj2000ns_to_et.side_effect = lambda x: x / 1e9

        # Mock lo_instrument_pointing to return pointing at spin_angle=270, off_angle=0
        # for all sample times (simulating no off-pointing with 90 degree pivot)
        def mock_pointing(ets, pivot_angle, to_frame):
            n_times = len(np.atleast_1d(ets))
            # Return (longitude, latitude) = (270, 0) for all times
            return np.column_stack([np.full(n_times, 270.0), np.zeros(n_times)])

        mock_lo_instrument_pointing.side_effect = mock_pointing

        # Act
        exposure_times = calculate_exposure_times(
            pointing_start_met, pointing_end_met, n_representative_spins=3
        )

        # Assert
        # Check shape
        assert exposure_times.shape == PSET_SHAPE

        # Check that exposure times sum to approximately total pointing duration / 7
        # Each ESA energy step is only active for 1/7 of the total time
        # (within tolerance due to binning)
        total_duration = pointing_end_met - pointing_start_met
        # Sum over spin_angle and off_angle dimensions for one ESA step
        exposure_sum = exposure_times.values[0, 0, :, :].sum()
        np.testing.assert_allclose(
            exposure_sum, total_duration / N_ESA_ENERGY_STEPS, rtol=0.01
        )

        # Check that all ESA steps have the same exposure (geometry-independent)
        for i in range(1, 7):
            np.testing.assert_array_equal(
                exposure_times.values[0, 0, :, :],
                exposure_times.values[0, i, :, :],
            )


def test_calculate_exposure_times_no_goodtimes_overlap(use_fake_spin_data_for_time):
    """Test that exposure times are zero when goodtimes don't overlap with pointing."""
    # Arrange
    # Pointing period is [511000100, 511000200]
    pointing_start_met = 511000100.0
    pointing_end_met = 511000200.0
    use_fake_spin_data_for_time(pointing_start_met)

    # Goodtimes window [511000000, 511000050] ends before pointing starts
    goodtimes_ds = xr.Dataset(
        {
            "gt_start_met": ("epoch", [511000000.0]),
            "gt_end_met": ("epoch", [511000050.0]),
        },
        coords={"epoch": [0]},
    )

    with (
        patch(
            "imap_processing.lo.l1c.lo_l1c.lo_instrument_pointing"
        ) as mock_lo_instrument_pointing,
        patch(
            "imap_processing.lo.l1c.lo_l1c.met_to_ttj2000ns"
        ) as mock_met_to_ttj2000ns,
        patch("imap_processing.lo.l1c.lo_l1c.ttj2000ns_to_et") as mock_ttj2000ns_to_et,
    ):
        # Mock the time conversions to pass through
        mock_met_to_ttj2000ns.side_effect = lambda x: x * 1e9
        mock_ttj2000ns_to_et.side_effect = lambda x: x / 1e9

        # Mock lo_instrument_pointing
        def mock_pointing(ets, pivot_angle, to_frame):
            n_times = len(np.atleast_1d(ets))
            return np.column_stack([np.full(n_times, 270.0), np.zeros(n_times)])

        mock_lo_instrument_pointing.side_effect = mock_pointing

        # Act
        exposure_times = calculate_exposure_times(
            pointing_start_met,
            pointing_end_met,
            goodtimes_ds=goodtimes_ds,
            n_representative_spins=3,
        )

        # Assert
        # All exposure times should be zero since goodtimes don't overlap
        assert exposure_times.shape == PSET_SHAPE
        np.testing.assert_array_equal(exposure_times.values, 0.0)


def test_get_representative_spin_times(use_fake_spin_data_for_time):
    """Test that representative spins are evenly distributed across pointing."""
    # Arrange
    pointing_start_met = 511000000.0
    pointing_end_met = 511001500.0  # ~100 spins at ~15s each
    use_fake_spin_data_for_time(pointing_start_met)

    # Act
    representative_spins = get_representative_spin_times(
        pointing_start_met, pointing_end_met, n_spins=5
    )

    # Assert
    assert len(representative_spins) == 5
    assert "spin_start_met" in representative_spins.columns
    assert "actual_spin_period" in representative_spins.columns

    # Check that spins are within the pointing period
    assert all(representative_spins["spin_start_met"] >= pointing_start_met)
    assert all(representative_spins["spin_start_met"] < pointing_end_met)


def test_get_representative_spin_times_fewer_available(use_fake_spin_data_for_time):
    """Test that we get all spins when fewer than requested are available."""
    # Arrange - very short pointing with only a few spins
    pointing_start_met = 511000000.0
    pointing_end_met = 511000045.0  # ~3 spins at ~15s each
    use_fake_spin_data_for_time(pointing_start_met)

    # Act
    representative_spins = get_representative_spin_times(
        pointing_start_met, pointing_end_met, n_spins=10
    )

    # Assert - should get all available spins (less than 10)
    assert len(representative_spins) <= 10
    assert len(representative_spins) >= 1


def test_sample_boresight_bins():
    """Test boresight sampling within a single spin."""
    # Arrange
    spin_start_met = 511000000.0
    spin_period = 15.0

    with (
        patch(
            "imap_processing.lo.l1c.lo_l1c.lo_instrument_pointing"
        ) as mock_lo_instrument_pointing,
        patch(
            "imap_processing.lo.l1c.lo_l1c.met_to_ttj2000ns"
        ) as mock_met_to_ttj2000ns,
        patch("imap_processing.lo.l1c.lo_l1c.ttj2000ns_to_et") as mock_ttj2000ns_to_et,
    ):
        # Mock time conversions
        mock_met_to_ttj2000ns.side_effect = lambda x: x * 1e9
        mock_ttj2000ns_to_et.side_effect = lambda x: x / 1e9

        # Mock lo_instrument_pointing to simulate rotating boresight
        def mock_pointing(ets, pivot_angle, to_frame):
            n_times = len(np.atleast_1d(ets))
            # Simulate boresight sweeping through spin angles (0-360)
            # with zero off-angle (latitude)
            spin_angles = np.linspace(0, 360, n_times, endpoint=False)
            off_angles = np.zeros(n_times)
            return np.column_stack([spin_angles, off_angles])

        mock_lo_instrument_pointing.side_effect = mock_pointing

        # Act
        spin_angles, off_angles = sample_boresight_bins(spin_start_met, spin_period)

        # Assert
        assert len(spin_angles) == N_SAMPLES_PER_SPIN
        assert len(off_angles) == N_SAMPLES_PER_SPIN

        # Check spin angles are in valid range [0, 360)
        assert all(spin_angles >= 0)
        assert all(spin_angles < 360)

        # Check off angles are near zero (as mocked)
        np.testing.assert_allclose(off_angles, 0, atol=1e-10)


def test_calculate_bin_weights():
    """Test bin weight calculation from sampled angles."""
    # Arrange - create samples concentrated in specific bins
    # All samples at off_angle=0
    n_samples = 1000
    off_angles = np.full(n_samples, 0.0)

    # Act
    bin_weights = calculate_bin_weights(off_angles)

    # Assert
    assert bin_weights.shape == (N_OFF_ANGLE_BINS,)

    # Weights should sum to 1
    np.testing.assert_allclose(bin_weights.sum(), 1.0)

    # Find the bin that should have all the weight
    # off_angle=0 is in bin 20 (center of [-2, 2] range with 40 bins)
    expected_off_bin = 20  # (0 - (-2)) / 0.1 = 20

    # That bin should have weight close to 1
    assert bin_weights[expected_off_bin] > 0.9


def test_calculate_bin_weights_distributed():
    """Test bin weights with uniformly distributed samples."""
    # Arrange - uniform distribution across off_angles
    np.random.seed(42)
    n_samples = 100000
    off_angles = np.random.uniform(-2, 2, n_samples)

    # Act
    bin_weights = calculate_bin_weights(off_angles)

    # Assert
    assert bin_weights.shape == (N_OFF_ANGLE_BINS,)

    # Weights should sum to 1
    np.testing.assert_allclose(bin_weights.sum(), 1.0)

    # With uniform distribution, weights should be approximately equal
    expected_weight = 1.0 / N_OFF_ANGLE_BINS
    np.testing.assert_allclose(bin_weights.mean(), expected_weight, rtol=0.1)


def test_create_goodtimes_fraction():
    """Test good-times fractional coverage calculation from ancillary data."""
    # Arrange - create a simple goodtimes Dataset
    # Good-times cover the full pointing duration for all spin bins
    # bin_start and bin_end are inclusive, 0-indexed (0-59 for 6-degree bins)
    goodtimes_ds = xr.Dataset(
        {
            "gt_start_met": ("epoch", [500000000.0, 500000000.0]),
            "gt_end_met": ("epoch", [500001000.0, 500001000.0]),
        },
        coords={"epoch": [0, 1]},
    )

    pointing_start_met = 500000000.0
    pointing_end_met = 500001000.0

    # Act
    fraction = create_goodtimes_fraction(
        goodtimes_ds, pointing_start_met, pointing_end_met
    )

    # Assert
    assert fraction.shape == (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS)

    # The current implementation does not filter by bin range or E-Step flags,
    # so coverage is 1.0 everywhere
    np.testing.assert_allclose(fraction, 1.0)


def test_create_goodtimes_fraction_partial_coverage():
    """Test good-times with partial time coverage of pointing period."""
    # Arrange - good-times cover only half of the pointing duration
    goodtimes_ds = xr.Dataset(
        {
            "gt_start_met": ("epoch", [500000000.0]),
            "gt_end_met": ("epoch", [500000500.0]),  # Only first 500s of 1000s pointing
        },
        coords={"epoch": [0]},
    )

    pointing_start_met = 500000000.0
    pointing_end_met = 500001000.0

    # Act
    fraction = create_goodtimes_fraction(
        goodtimes_ds, pointing_start_met, pointing_end_met
    )

    # Assert - all bins should have 50% coverage
    np.testing.assert_allclose(fraction, 0.5)


def test_create_goodtimes_fraction_no_overlap():
    """Test good-times fraction when no good-times overlap with pointing."""
    # Arrange - goodtimes outside pointing period
    goodtimes_ds = xr.Dataset(
        {
            "gt_start_met": ("epoch", [400000000.0]),
            "gt_end_met": ("epoch", [400001000.0]),
        },
        coords={"epoch": [0]},
    )

    pointing_start_met = 500000000.0
    pointing_end_met = 500001000.0

    # Act
    fraction = create_goodtimes_fraction(
        goodtimes_ds, pointing_start_met, pointing_end_met
    )

    # Assert - all zeros since no overlap
    np.testing.assert_allclose(fraction, 0.0)


@pytest.mark.parametrize("species", [FilterType.HYDROGEN, FilterType.OXYGEN])
def test_set_background_rates(l1b_bgrates_ds, attr_mgr, species):
    # Arrange
    sci_deps = {"imap_lo_l1b_bgrates": l1b_bgrates_ds}
    species_key = species.value
    expected_rates = l1b_bgrates_ds[f"{species_key}_background_rates"].values
    expected_var = l1b_bgrates_ds[f"{species_key}_background_variance"].values

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
