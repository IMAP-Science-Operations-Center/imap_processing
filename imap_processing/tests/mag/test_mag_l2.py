from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag.constants import FILLVAL, DataMode
from imap_processing.mag.l2.mag_l2 import mag_l2, retrieve_matrix_from_l2_calibration
from imap_processing.mag.l2.mag_l2_data import MagL2, ValidFrames
from imap_processing.spice.time import (
    et_to_datetime64,
    et_to_ttj2000ns,
    et_to_utc,
    str_to_et,
    ttj2000ns_to_et,
)
from imap_processing.tests.mag.conftest import mag_l1a_dataset_generator


@pytest.mark.parametrize("data_mode", ["norm", "burst"])
def test_mag_l2_attributes(norm_dataset, mag_test_l2_data, data_mode):
    """Test that L2 datasets have correct attributes based on frame and mode."""
    calibration_dataset = mag_test_l2_data[0]
    offset_dataset = mag_test_l2_data[1]

    # Create dataset for the appropriate mode
    test_dataset = norm_dataset.copy()
    test_dataset.attrs["Logical_source"] = f"imap_mag_l1c_{data_mode}-mago"

    # Convert data_mode string to DataMode enum
    mode = DataMode.NORM if data_mode == "norm" else DataMode.BURST

    with patch(
        "imap_processing.mag.l2.mag_l2_data.frame_transform",
        side_effect=lambda *args, **kwargs: args[1],
    ):
        l2_datasets = mag_l2(
            calibration_dataset,
            offset_dataset,
            test_dataset,
            np.datetime64("2025-10-17"),
            mode=mode,
        )

    # Verify we have the expected number of datasets
    # L2 produces 5 frames: SRF, GSE, GSM, RTN, DSRF
    assert len(l2_datasets) == 5, (
        f"Expected 5 {data_mode} datasets, got {len(l2_datasets)}"
    )

    for dataset in l2_datasets:
        assert "Logical_source" in dataset.attrs
        assert "Data_type" in dataset.attrs
        assert dataset.attrs["Logical_source"].startswith(f"imap_mag_l2_{data_mode}-")

        # Verify that data_level is correctly set to "l2" in logical source
        logical_source_parts = dataset.attrs["Logical_source"].split("_")
        assert logical_source_parts[2] == "l2", (
            f"Expected data_level 'l2' in Logical_source, "
            f"got '{logical_source_parts[2]}'"
        )

        # Extract frame from logical source
        frame = dataset.attrs["Logical_source"].split("-")[-1].upper()

        vectors_attrs = dataset[f"b_{frame.lower()}"].attrs
        assert "DICT_KEY" in vectors_attrs

        assert f"CoordinateSystemName:{frame}" in vectors_attrs["DICT_KEY"]

        assert "magnitude" in dataset.data_vars
        assert "range" in dataset.data_vars
        assert dataset["magnitude"].attrs["UNITS"] == "nT"
        assert dataset["range"].attrs["DICT_KEY"] == (
            "SPASE>Support>SupportQuantity:InstrumentMode"
        )
        assert vectors_attrs["CDF_DATA_TYPE"] == "CDF_FLOAT"


def test_mag_l2(norm_dataset, mag_test_l2_data):
    calibration_dataset = mag_test_l2_data[0]

    offset_dataset = mag_test_l2_data[1]
    with patch(
        "imap_processing.mag.l2.mag_l2_data.frame_transform",
        side_effect=lambda *args, **kwargs: args[1],
    ):
        l2 = mag_l2(
            calibration_dataset,
            offset_dataset,
            norm_dataset,
            np.datetime64("2025-10-17"),
        )

    expected_frames = [
        ValidFrames.SRF,
        ValidFrames.GSE,
        ValidFrames.GSM,
        ValidFrames.RTN,
        ValidFrames.DSRF,
    ]

    assert len(l2) == len(expected_frames), (
        f"L2 should produce {len(expected_frames)} frames"
    )

    for i, dataset in enumerate(l2):
        assert expected_frames[i].var_name in dataset.data_vars
        assert expected_frames[i].name in dataset.attrs["Data_type"]


def test_mag_l2_some_epochs_not_in_spice(norm_dataset, mag_test_l2_data):
    def return_some_nan_matrices_for_dsrf(
        et, from_frame, to_frame, allow_spice_noframeconnect
    ):
        matrices = np.tile(np.eye(3), (len(et), 1, 1))
        if to_frame == ValidFrames.DSRF.spice_frame:
            for i in range(10, matrices.shape[0], 10):  # every 10th matrix is NaN
                matrices[i] = np.full((3, 3), np.nan)
        return matrices

    calibration_dataset = mag_test_l2_data[0]
    offset_dataset = mag_test_l2_data[1]

    with patch(
        "imap_processing.spice.geometry.get_rotation_matrix",
        side_effect=return_some_nan_matrices_for_dsrf,
    ):
        l2 = mag_l2(
            calibration_dataset,
            offset_dataset,
            norm_dataset,
            np.datetime64("2025-10-17"),
        )

    assert len(l2) == 5, "L2 should produce 5 frames"

    all_vars = ["b_srf", "b_gse", "b_gsm", "b_rtn", "b_dsrf"]

    for dataset in l2:
        assert len(set(all_vars) & set(dataset.data_vars)) == 1, (
            "Each dataset should have one of the expected vector variables"
        )

    assert (
        l2[-1].attrs["Data_type"] == "L2_norm-dsrf>Level 2 normal rate data in DSRF"
    ), "Last frame should be DSRF"

    dsrf_vectors = l2[-1]["b_dsrf"].data
    for i in range(10, len(dsrf_vectors), 10):
        assert np.isnan(dsrf_vectors[i]).all(), f"Vectors at index {i} should be NaN"


def test_offset_application(norm_dataset, mag_test_l2_data):
    # Test against zeros
    offsets = mag_test_l2_data[1]
    output = MagL2(
        vectors=norm_dataset["vectors"].data[:, :3],
        epoch=norm_dataset["epoch"].data,
        range=norm_dataset["vectors"].data[:, 3],
        global_attributes={},
        quality_flags=None,
        quality_bitmask=None,
        data_mode=DataMode.NORM,
        offsets=offsets["offsets"].data,
        timedelta=offsets["timedeltas"].data,
    )

    expected_vectors = norm_dataset["vectors"].data[:, :3]
    assert np.allclose(output.vectors, expected_vectors, atol=1e-9)
    assert np.allclose(output.epoch, norm_dataset["epoch"], atol=1e-9)

    new_offsets = np.zeros((len(norm_dataset["epoch"]), 3))
    new_offsets[0] = [1, 1, 1]
    new_offsets[1] = [-1, -1, -1]
    new_offsets[-1] = [1, 0, -1]

    new_timeshift = np.zeros(len(norm_dataset["epoch"]))
    new_timeshift[0] = 0.00001
    new_timeshift[1] = -0.00001
    new_timeshift[2] = 1e-9

    expected_timeshift = norm_dataset["epoch"].data
    # Timeshift is provided in seconds, epoch is in nanoseconds
    expected_timeshift[0] = expected_timeshift[0] + 10000
    expected_timeshift[1] = expected_timeshift[1] - 10000
    expected_timeshift[2] = expected_timeshift[2] + 1

    output = MagL2(
        vectors=norm_dataset["vectors"].data[:, :3],
        epoch=norm_dataset["epoch"].data,
        range=norm_dataset["vectors"].data[:, 3],
        global_attributes={},
        quality_flags=None,
        quality_bitmask=None,
        data_mode=None,
        offsets=new_offsets,
        timedelta=new_timeshift,
    )

    expected_vectors = norm_dataset["vectors"].data[:, :3]
    expected_vectors[0] = [2, 2, 2]
    expected_vectors[1] = [1, 1, 1]
    expected_vectors[-1] = [3505, 3504, 3503]

    assert np.allclose(output.vectors, expected_vectors, atol=1e-9)
    assert np.allclose(output.epoch, expected_timeshift, atol=1e-9)


def test_error_raises(mag_test_l2_data):
    dataset = mag_l1a_dataset_generator(3504)
    with pytest.raises(ValueError, match="same timestamps"):
        mag_l2(
            mag_test_l2_data[0],
            mag_test_l2_data[1],
            dataset,
            np.datetime64("2025-10-17"),
        )

    dataset = mag_l1a_dataset_generator(3505)
    with pytest.raises(ValueError, match="same timestamps"):
        mag_l2(
            mag_test_l2_data[0],
            mag_test_l2_data[1],
            dataset,
            np.datetime64("2025-10-17"),
        )


def test_midnight_boundary(norm_dataset):
    day = np.datetime64("2025-10-17").astype("datetime64[D]")

    # Shift timestamps to include midnight in the day and span 2 days
    shifted_timestamps = norm_dataset["epoch"].data - 1.08e13  # 3 hours in ns
    shifted_timestamps = shifted_timestamps + 2496981986944

    midnight = et_to_ttj2000ns(str_to_et("2025-10-17T00:00:00"))

    l2 = MagL2(
        vectors=norm_dataset["vectors"].data[:, :3],
        epoch=shifted_timestamps,
        range=norm_dataset["vectors"].data[:, 3],
        global_attributes={},
        quality_flags=np.zeros(len(norm_dataset["epoch"].data)),
        quality_bitmask=np.zeros(len(norm_dataset["epoch"].data)),
        data_mode=DataMode.NORM,
        offsets=np.zeros((len(norm_dataset["epoch"].data), 3)),
        timedelta=np.zeros(len(norm_dataset["epoch"].data)),
    )

    l2.truncate_to_24h(day)

    # Midnight should be included in the start of the day
    assert l2.epoch[0] == midnight

    l2 = MagL2(
        vectors=norm_dataset["vectors"].data[:, :3],
        epoch=shifted_timestamps,
        range=norm_dataset["vectors"].data[:, 3],
        global_attributes={},
        quality_flags=np.zeros(len(norm_dataset["epoch"].data)),
        quality_bitmask=np.zeros(len(norm_dataset["epoch"].data)),
        data_mode=DataMode.NORM,
        offsets=np.zeros((len(norm_dataset["epoch"].data), 3)),
        timedelta=np.zeros(len(norm_dataset["epoch"].data)),
    )

    l2.truncate_to_24h(day - 1)

    # midnight not included in previous day
    assert midnight not in l2.epoch


@pytest.mark.parametrize(
    ("time_shift", "start_diff", "end_diff"),
    # 3 hours in ns
    [
        (-1.08e13, -1, 0),
        # 19 hours in ns
        (6.84e13, 0, 1),
    ],
)
def test_timestamp_truncation(
    norm_dataset, mag_test_l2_data, time_shift, start_diff, end_diff
):
    day = np.datetime64("2025-10-17").astype("datetime64[D]")
    shifted_timestamps = norm_dataset["epoch"].data + time_shift
    l2 = MagL2(
        vectors=norm_dataset["vectors"].data[:, :3],
        epoch=shifted_timestamps,
        range=norm_dataset["vectors"].data[:, 3],
        global_attributes={},
        quality_flags=np.zeros(len(norm_dataset["epoch"].data)),
        quality_bitmask=np.zeros(len(norm_dataset["epoch"].data)),
        data_mode=DataMode.NORM,
        offsets=np.zeros((len(norm_dataset["epoch"].data), 3)),
        timedelta=np.zeros(len(norm_dataset["epoch"].data)),
    )

    first_epoch_val = np.array(et_to_utc(ttj2000ns_to_et(l2.epoch[0]))).astype(
        "datetime64[D]"
    )

    # Before starting: epoch spans two days
    assert first_epoch_val == day + start_diff

    last_epoch_val = np.array(et_to_utc(ttj2000ns_to_et(l2.epoch[-1]))).astype(
        "datetime64[D]"
    )
    assert last_epoch_val == day + end_diff

    l2.truncate_to_24h(day)

    # after truncation: epoch spans one day
    first_epoch_val = et_to_datetime64(ttj2000ns_to_et(l2.epoch[0])).astype(
        "datetime64[D]"
    )
    last_epoch_val = et_to_datetime64(ttj2000ns_to_et(l2.epoch[-1])).astype(
        "datetime64[D]"
    )

    assert first_epoch_val == day
    assert last_epoch_val == day

    # Timestamps should align with all data
    assert l2.epoch.shape[0] == l2.vectors.shape[0]
    assert l2.epoch.shape[0] == l2.magnitude.shape[0]
    assert l2.epoch.shape[0] == l2.range.shape[0]
    assert l2.epoch.shape[0] == l2.quality_flags.shape[0]
    assert l2.epoch.shape[0] == l2.quality_bitmask.shape[0]

    assert l2.epoch.shape[0] < shifted_timestamps.shape[0]
    post_trunc_shape = l2.epoch.shape[0]

    for ts in l2.epoch:
        assert ts in shifted_timestamps
    # Applying twice shouldn't affect anything
    l2.truncate_to_24h(day)

    assert l2.epoch.shape[0] == post_trunc_shape

    first_epoch_val = et_to_datetime64(ttj2000ns_to_et(l2.epoch[0])).astype(
        "datetime64[D]"
    )
    last_epoch_val = et_to_datetime64(ttj2000ns_to_et(l2.epoch[-1])).astype(
        "datetime64[D]"
    )

    assert first_epoch_val == day
    assert last_epoch_val == day


def test_magnitude():
    # Test magnitude calculation
    test_vector_one = np.array([[6, 9, 12]])
    expected_magnitude = np.sqrt(6**2 + 9**2 + 12**2)

    output_magnitude = MagL2.calculate_magnitude(test_vector_one)

    assert np.allclose(output_magnitude, expected_magnitude, atol=1e-9)

    test_multiple_vectors = np.random.rand(10, 3) * 10
    expected_magnitude = [
        np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) for x in test_multiple_vectors
    ]

    output_magnitude = MagL2.calculate_magnitude(test_multiple_vectors)
    assert np.allclose(output_magnitude, expected_magnitude, atol=1e-9)

    assert output_magnitude.shape == (10,)


@pytest.mark.parametrize(
    ("is_mago", "data_var"),
    [
        (True, "URFTOORFO"),
        (False, "URFTOORFI"),
    ],
)
def test_retrieve_matrix_from_l2_calibration(is_mago, data_var):
    start_day = np.datetime64("2025-10-15").astype("datetime64[D]")
    end_day = np.datetime64("2025-10-20").astype("datetime64[D]")
    epoch_vars = xr.DataArray(
        np.arange(start_day, end_day, dtype="datetime64[D]"),
        dims=["epoch"],
        coords={"epoch": np.arange(5)},
    )
    example_calibration_dataset = xr.Dataset(
        {
            "URFTOORFO": xr.DataArray(
                np.random.rand(5, 3, 3, 4),
                dims=["epoch", "URFTOORFO_dim_0", "URFTOORFO_dim_1", "URFTOORFO_dim_2"],
            ),
            "URFTOORFI": xr.DataArray(
                np.random.rand(5, 3, 3, 4),
                dims=["epoch", "URFTOORFI_dim_0", "URFTOORFI_dim_1", "URFTOORFI_dim_2"],
            ),
        },
        coords={"epoch": epoch_vars},
    )

    calibration_matrix = retrieve_matrix_from_l2_calibration(
        example_calibration_dataset, start_day, use_mago=is_mago
    )

    assert calibration_matrix.shape == (3, 3, 4)
    assert np.array_equal(
        example_calibration_dataset.sel(epoch=start_day)[data_var].data,
        calibration_matrix,
    )

    test_day = np.datetime64("2025-10-17").astype("datetime64[D]")
    calibration_matrix = retrieve_matrix_from_l2_calibration(
        example_calibration_dataset, test_day, use_mago=is_mago
    )

    assert calibration_matrix.shape == (3, 3, 4)
    assert np.array_equal(
        example_calibration_dataset.sel(epoch=test_day)[data_var].data,
        calibration_matrix,
    )


def test_spice_returns(norm_dataset):
    l2 = MagL2(
        vectors=norm_dataset["vectors"].data[:, :3],
        epoch=norm_dataset["epoch"].data,
        range=norm_dataset["vectors"].data[:, 3],
        global_attributes={},
        quality_flags=np.zeros(len(norm_dataset["epoch"].data)),
        quality_bitmask=np.zeros(len(norm_dataset["epoch"].data)),
        data_mode=DataMode.NORM,
        offsets=np.zeros((len(norm_dataset["epoch"].data), 3)),
        timedelta=np.zeros(len(norm_dataset["epoch"].data)),
    )

    assert l2.frame.name == "MAGO"

    with patch(
        "imap_processing.mag.l2.mag_l2_data.frame_transform",
        return_value=np.full(l2.vectors.shape, [-1, -1, -1]),
    ):
        l2.rotate_frame(ValidFrames.DSRF)
        assert l2.frame.name == "DSRF"
        assert not np.array_equal(l2.vectors, norm_dataset["vectors"].data[:, :3])
        assert np.array_equal(l2.vectors[0], [-1, -1, -1])


def test_rotate_frame_preserves_fillval_and_nan(norm_dataset):
    """Test that rotate_frame preserves FILLVAL and NaN vectors."""

    vectors = norm_dataset["vectors"].data[:, :3].copy()
    n = len(vectors)

    # Set some vectors to FILLVAL and NaN
    vectors[0] = [FILLVAL, FILLVAL, FILLVAL]
    vectors[2] = [np.nan, np.nan, np.nan]
    # Partial NaN in a row
    vectors[4] = [1.0, np.nan, 3.0]
    # Partial FILLVAL in a row
    vectors[5] = [FILLVAL, 2.0, 3.0]

    l2 = MagL2(
        vectors=vectors,
        epoch=norm_dataset["epoch"].data,
        range=norm_dataset["vectors"].data[:, 3],
        global_attributes={},
        quality_flags=np.zeros(n),
        quality_bitmask=np.zeros(n),
        data_mode=DataMode.NORM,
        offsets=np.zeros((n, 3)),
        timedelta=np.zeros(n),
    )

    rotated_values = np.full(vectors.shape, 99.0)
    with patch(
        "imap_processing.mag.l2.mag_l2_data.frame_transform",
        return_value=rotated_values,
    ):
        l2.rotate_frame(ValidFrames.DSRF)

    assert l2.frame == ValidFrames.DSRF

    # Full FILLVAL row -> all components should be FILLVAL
    assert np.all(l2.vectors[0] == FILLVAL)
    # Full NaN row -> all components should be FILLVAL
    assert np.all(l2.vectors[2] == FILLVAL)
    # Partial NaN -> affected components should be FILLVAL
    assert l2.vectors[4, 1] == FILLVAL
    # Partial FILLVAL -> affected components should be FILLVAL
    assert l2.vectors[5, 0] == FILLVAL

    # Normal vectors should get the rotated value
    assert np.all(l2.vectors[1] == 99.0)
    assert np.all(l2.vectors[3] == 99.0)


def test_qf(norm_dataset):
    qf = np.zeros(len(norm_dataset["epoch"].data), dtype=int)
    qf[1:4] = 1

    qf_bitmask = np.zeros(len(norm_dataset["epoch"].data), dtype=int)
    qf_bitmask[2] = 1
    qf_bitmask[5:8] = 2
    l2 = MagL2(
        vectors=norm_dataset["vectors"].data[:, :3],
        epoch=norm_dataset["epoch"],
        range=norm_dataset["vectors"].data[:, 3],
        global_attributes={},
        quality_flags=qf,
        quality_bitmask=qf_bitmask,
        data_mode=DataMode.NORM,
        offsets=np.zeros((len(norm_dataset["epoch"].data), 3)),
        timedelta=np.zeros(len(norm_dataset["epoch"].data)),
    )

    l2.frame = ValidFrames.SRF
    attributes = ImapCdfAttributes()
    attributes.add_instrument_global_attrs("mag")
    attributes.add_instrument_variable_attrs("mag", "l2")

    output = l2.generate_dataset(attributes, np.datetime64("2025-10-17"))

    assert "quality_flags" in output.data_vars
    assert "quality_bitmask" in output.data_vars
    assert np.array_equal(output["quality_flags"].data, qf)
    assert np.array_equal(output["quality_bitmask"].data, qf_bitmask)
