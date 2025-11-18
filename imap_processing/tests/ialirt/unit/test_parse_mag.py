"""Tests to support I-ALiRT MAG packet processing."""

import numpy as np
import pandas as pd
import pytest
import spiceypy
import xarray as xr
from scipy.interpolate import make_interp_spline

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.ialirt.l0.parse_mag import (
    apply_gradiometry_correction,
    calculate_l1b,
    calibrate_and_offset_vectors,
    extract_magnetic_vectors,
    get_pkt_counter,
    get_status_data,
    get_time,
    process_packet,
    retrieve_matrix_from_single_l1b_calibration,
    transform_to_frames,
    transform_to_inertial,
)
from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.mag.constants import MAX_FINE_TIME
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.time import et_to_ttj2000ns, met_to_ttj2000ns
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def xtce_mag_path():
    """Returns the xtce directory."""
    return imap_module_directory / "ialirt" / "packet_definitions" / "ialirt_mag.xml"


@pytest.fixture(scope="session")
def binary_packet_path():
    """Returns the paths to the binary packets."""
    directory = imap_module_directory / "tests" / "ialirt" / "data" / "l0"
    filenames = [
        "461971383-404.bin",
        "461971384-405.bin",
        "461971385-406.bin",
        "461971386-407.bin",
        "461971387-408.bin",
        "461971388-409.bin",
        "461971389-410.bin",
        "461971390-411.bin",
        "461971391-412.bin",
    ]
    return tuple(directory / fname for fname in filenames)


@pytest.fixture(scope="session")
def mag_test_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "data"
        / "l0"
        / "sample_decoded_i-alirt_data.csv"
    )
    data = pd.read_csv(data_path)

    return data


@pytest.fixture(scope="session")
def mag_sc_test_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "data"
        / "l0"
        / "MAGScience-IALiRT-20250421-13h16.csv"
    )
    data = pd.read_csv(data_path)

    return data


@pytest.fixture
def xarray_data(binary_packet_path, xtce_mag_path):
    """Create xarray data for multiple packets."""
    apid = 1001

    xarray_data = tuple(
        packet_file_to_datasets(packet, xtce_mag_path, use_derived_value=False)[apid]
        for packet in binary_packet_path
    )

    merged_xarray_data = xr.concat(xarray_data, dim="epoch")
    return merged_xarray_data


@pytest.fixture
def grouped_data():
    """Creates grouped data for tests."""
    epoch = np.arange(12)

    # Example `src_seq_ctr` values for 3 groups:
    # Group 0 - valid, all diffs = 1
    # Group 1 - invalid, has a jump of 5
    # Group 2 - valid, wraps at -16383
    src_seq_ctr = np.concatenate(
        [
            np.arange(100, 104),
            np.array([200, 205, 206, 207]),
            np.array([16382, 16383, 0, 1]),
        ],
        dtype=np.int32,
    )
    mag_acq_tm_coarse = np.repeat(
        np.array([461971382, 461971386, 461971390], dtype=np.uint32), repeats=4
    )

    mag_acq_tm_fine = np.array(
        [1502, 1502, 1505, 1505, 1500, 1500, 1503, 1503, 1497, 1497, 1491, 1491]
    )

    group = np.tile(np.arange(3), 4).reshape(4, 3).T.ravel()

    grouped_data = xr.Dataset(
        data_vars={
            "src_seq_ctr": ("epoch", src_seq_ctr),
            "mag_acq_tm_coarse": ("epoch", mag_acq_tm_coarse),
            "mag_acq_tm_fine": ("epoch", mag_acq_tm_fine),
        },
        coords={"epoch": epoch, "group": ("epoch", group)},
    )

    return grouped_data


@pytest.fixture
def calibration_dataset():
    """Returns the calibration data."""
    calibration_dataset = load_cdf(
        imap_module_directory
        / "tests"
        / "mag"
        / "validation"
        / "calibration"
        / "imap_mag_l1b-calibration_20240229_v002.cdf"
    )
    return calibration_dataset


def test_get_pkt_counter(xarray_data):
    """Tests the get_pkt_counter function."""
    status_values = xarray_data["mag_status"].values
    pkt_counter = get_pkt_counter(status_values)
    assert np.array_equal(pkt_counter, np.array([0, 1, 2, 3, 0, 1, 2, 3, 0]))


def test_get_status_data(xarray_data, mag_test_data):
    """Tests the get_status_data function."""

    status_data = get_status_data(
        xarray_data["mag_status"].values[0:4], np.array([0, 1, 2, 3])
    )
    index = mag_test_data["PRI_COARSETM"] == 461971382
    matching_row = mag_test_data[index]

    for key in status_data.keys():
        assert status_data[key] == matching_row[key.upper()].values[0]


def test_get_time(grouped_data, mag_test_l1b_calibration_data):
    """Tests the get_time function."""
    time_shift_mago = mag_test_l1b_calibration_data[1]
    time_shift_magi = mag_test_l1b_calibration_data[3]

    time_data = get_time(
        grouped_data, 1, np.array([0, 1, 2, 3]), time_shift_mago, time_shift_magi
    )

    assert time_data["pri_coarsetm"] == 461971386
    assert time_data["pri_fintm"] == 1500
    assert time_data["sec_coarsetm"] == 461971386
    assert time_data["sec_fintm"] == 1503


def test_extract_magnetic_vectors():
    """Tests the extract_magnetic_vectors function."""
    science_values = xr.DataArray(
        data=np.array([15797207, 5750698, 15921110, 2342918], dtype=np.uint32)
    )

    vectors = extract_magnetic_vectors(science_values)

    assert vectors == {
        "pri_x": -3829,
        "pri_y": -10409,
        "pri_z": -16470,
        "sec_x": -3345,
        "sec_y": -10717,
        "sec_z": -16378,
    }


def test_calculate_l1b(grouped_data, xarray_data, calibration_dataset):
    """Tests the calculate_l1b function."""

    pkt_counter = np.array([0.0, 1.0, 2.0, 3.0])

    science_data = {
        "pri_x": 1.0,
        "pri_y": 2.0,
        "pri_z": 3.0,
        "sec_x": 4.0,
        "sec_y": 5.0,
        "sec_z": 6.0,
    }

    status_data = {
        "fob_range": 1,
        "fib_range": 1,
    }

    vec_mago, vec_magi, time_data = calculate_l1b(
        grouped_data, 0, pkt_counter, science_data, status_data, calibration_dataset
    )

    assert vec_mago.shape == (4,)
    assert vec_magi.shape == (4,)
    assert "primary_epoch" in time_data
    assert "secondary_epoch" in time_data


def test_l1b_data(xarray_data, mag_test_data, calibration_dataset):
    """Tests the parse_packet function."""

    # Add required parameters.
    xarray_data["met"] = calculate_time(
        xarray_data["mag_acq_tm_coarse"], xarray_data["mag_acq_tm_fine"], 256
    )
    xarray_data["pkt_counter"] = get_pkt_counter(xarray_data["mag_status"])

    grouped_data = find_groups(xarray_data, (0, 3), "pkt_counter", "met")

    unique_groups = np.unique(grouped_data["group"])
    l1b_data = []

    for group in unique_groups:
        # Get status values for each group.
        status_values = grouped_data["mag_status"][
            (grouped_data["group"] == group).values
        ]
        pkt_counter = grouped_data["pkt_counter"][
            (grouped_data["group"] == group).values
        ]

        # Get decoded status data.
        status_data = get_status_data(status_values, pkt_counter)

        # Get science values for each group.
        science_values = grouped_data["mag_data"][
            (grouped_data["group"] == group).values
        ]
        science_data = extract_magnetic_vectors(science_values)
        _, _, time_data = calculate_l1b(
            grouped_data,
            group,
            pkt_counter,
            science_data,
            status_data,
            calibration_dataset,
        )

        l1b_data.append({**status_data, **science_data, **time_data})

    for packet in l1b_data:
        index = packet["pri_coarsetm"] == mag_test_data["PRI_COARSETM"]
        matching_rows = mag_test_data[index]

        data_keys = ["pri_x", "pri_y", "pri_z", "sec_x", "sec_y", "sec_z"]

        for key in packet.keys():
            if key.upper() in matching_rows.keys():
                if key in data_keys:
                    # Convert to int16 for comparison.
                    assert packet[key] == int(
                        np.uint16(matching_rows[key.upper()].values[0]).astype(np.int16)
                    )
                else:
                    assert packet[key] == matching_rows[key.upper()].values[0]


@pytest.mark.external_test_data
def test_process_spacecraft_packet(
    mag_sc_test_data, calibration_dataset, sc_packet_path
):
    """Tests the parse_packet function."""
    packet_path, xtce_ialirt_path = sc_packet_path
    sc_xarray_data = packet_file_to_datasets(
        packet_path, xtce_ialirt_path, use_derived_value=False
    )[478]

    # Add required parameters.
    sc_xarray_data["met"] = calculate_time(
        sc_xarray_data["mag_acq_tm_coarse"], sc_xarray_data["mag_acq_tm_fine"], 256
    )
    sc_xarray_data["pkt_counter"] = get_pkt_counter(sc_xarray_data["mag_status"])

    grouped_data = find_groups(sc_xarray_data, (0, 3), "pkt_counter", "met")

    unique_groups = np.unique(grouped_data["group"])
    l1b_data = []

    for group in unique_groups:
        # Get status values for each group.
        status_values = grouped_data["mag_status"][
            (grouped_data["group"] == group).values
        ]
        pkt_counter = grouped_data["pkt_counter"][
            (grouped_data["group"] == group).values
        ]
        if not np.array_equal(pkt_counter, np.arange(4)):
            continue

        # Get decoded status data.
        status_data = get_status_data(status_values, pkt_counter)

        # Get science values for each group.
        science_values = grouped_data["mag_data"][
            (grouped_data["group"] == group).values
        ]
        science_data = extract_magnetic_vectors(science_values)
        _, _, time_data = calculate_l1b(
            grouped_data,
            group,
            pkt_counter,
            science_data,
            status_data,
            calibration_dataset,
        )

        l1b_data.append({**status_data, **science_data, **time_data})

    sequence = []
    for packet in l1b_data:
        index = (mag_sc_test_data["pri_coarse"] == packet["pri_coarsetm"]) & (
            mag_sc_test_data["pri_fine"] == packet["pri_fintm"]
        )
        matching_rows = mag_sc_test_data[index]

        if matching_rows.empty:
            continue

        row = matching_rows.iloc[0]

        # Row that does not match
        if row["sequence"] == 2931:
            continue

        sequence.append(row["sequence"])

        assert row["x_pri"] == packet["pri_x"]
        assert row["y_pri"] == packet["pri_y"]
        assert row["z_pri"] == packet["pri_z"]
        assert row["x_sec"] == packet["sec_x"]
        assert row["y_sec"] == packet["sec_y"]
        assert row["z_sec"] == packet["sec_z"]

        # Timestamp check
        time_data_pri_met = float(row["pri_coarse"] + row["pri_fine"] / MAX_FINE_TIME)
        time_data_primary_ttj2000ns = met_to_ttj2000ns(time_data_pri_met)
        _, time_shift_mago = retrieve_matrix_from_single_l1b_calibration(
            calibration_dataset, is_mago=True
        )
        primary_epoch = time_data_primary_ttj2000ns + time_shift_mago.data * 1e9

        assert packet["primary_epoch"] == primary_epoch


def test_calibrate_and_offset_vectors(ialirt_mag_test_l1d_data):
    """Tests calibrate_and_offset_vectors function."""

    # MAGo and MAGi raw counts
    mago_vectors = np.array([[1.0, 2.0, 3.0, 0]])
    magi_vectors = np.array([[7.0, 8.0, 9.0, 2]])

    # Calibration and offsets from ancillary cdf
    mago_calibration = ialirt_mag_test_l1d_data["URFTOORFO"][0]
    magi_calibration = ialirt_mag_test_l1d_data["URFTOORFI"][0]
    offsets = ialirt_mag_test_l1d_data["offsets"][0]

    mago_out = calibrate_and_offset_vectors(
        mago_vectors, mago_calibration, offsets, is_magi=False
    )
    magi_out = calibrate_and_offset_vectors(
        magi_vectors, magi_calibration, offsets, is_magi=True
    )

    # Every offset is zero.
    # For every range (0 to 3), the 3 by 3 calibration matrix is the identity matrix.
    np.testing.assert_allclose(mago_out.squeeze(), mago_vectors.squeeze()[0:3])
    np.testing.assert_allclose(magi_out.squeeze(), magi_vectors.squeeze()[0:3])


def test_apply_gradiometry_correction(ialirt_mag_test_l1d_data):
    """Tests apply_gradiometry_correction function."""

    gradiometer_factor = ialirt_mag_test_l1d_data["gradiometer_factor"].values

    # MAGo and MAGi vectors.
    mago_vector_eclipj2000 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    magi_vector_eclipj2000 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    time_data = {
        "primary_epoch": np.array([1.0, 2.0]),
        "secondary_epoch": np.array([3.0, 4.0]),
    }

    mago_corrected, magnitude = apply_gradiometry_correction(
        mago_vector_eclipj2000,
        np.array(time_data["primary_epoch"]),
        magi_vector_eclipj2000,
        np.array(time_data["secondary_epoch"]),
        gradiometer_factor.squeeze(),
    )

    spline = make_interp_spline(
        time_data["secondary_epoch"], magi_vector_eclipj2000, k=1
    )
    interpolated_vectors = spline(time_data["primary_epoch"])

    # Spot check.
    np.testing.assert_array_equal(
        interpolated_vectors, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    offset = interpolated_vectors - mago_vector_eclipj2000

    offset_value = np.apply_along_axis(
        np.dot,
        1,
        offset,
        gradiometer_factor.squeeze(),
    )
    expected_mago_corrected = mago_vector_eclipj2000 - offset_value
    np.testing.assert_array_equal(mago_corrected, expected_mago_corrected)

    expected_magnitude = np.sqrt(np.sum(mago_corrected**2, axis=1))
    np.testing.assert_array_equal(magnitude, expected_magnitude)


@pytest.mark.external_kernel
def test_transform_to_frames(furnish_kernels, spice_test_data_path):
    """Test transform_to_frames over multiple spin phases."""

    kernels = [
        "imap_science_100.tf",
        "imap_130.tf",
        "naif0012.tls",
        "de440s.bsp",
        "imap_spk_demo.bsp",
        "pck00011.tpc",
    ]
    inst_frame = SpiceFrame.IMAP_MAG_O

    # Use a fixed spin axis pointing at +Z (RA=0, Dec=90)
    ra = np.array([0.0, 0.0, 0.0, 0.0])
    dec = np.array([90.0, 90.0, 90.0, 90.0])
    spin_phase = np.array([0.0, 90.0, 180.0, 270.0])  # degrees

    # Unit vector pointing along +X in instrument frame
    mag_vector = np.array([1.0, 0.0, 0.0])

    attitude_time = np.array([817561851.0, 817561853.0, 817561855.0, 817561857.0])
    target_time = 817561854.0  # halfway between 90° and 180° spin phase

    with furnish_kernels(kernels):
        # With spin phase halfway between 90 and 180, the +X MAG_O vector should
        # be pointing at 135. Since MAG is not mounted perfectly with the
        # instrument +X pointing in the S/C -Y direction, we need to rotate the
        # expected vector from the instrument frame to the S/C frame.
        inst_to_sc_rot = spiceypy.pxform(
            inst_frame.name, SpiceFrame.IMAP_SPACECRAFT.name, 0.0
        )
        inverse_spin_phase_rot = spiceypy.axisar(
            np.array([0, 0, 1]), float(np.radians(-135.0))
        )
        expected_vector = (
            inverse_spin_phase_rot @ inst_to_sc_rot @ np.array([1.0, 0.0, 0.0])
        )

        inertial_vector = transform_to_inertial(
            np.radians(spin_phase),
            np.radians(ra),
            np.radians(dec),
            et_to_ttj2000ns(attitude_time),
            et_to_ttj2000ns(target_time),
            mag_vector,
            inst_frame,
        )

        gse_vector, gsm_vector, rtn_vector = transform_to_frames(
            et_to_ttj2000ns(target_time),
            inertial_vector,
        )

        rot_ecl_to_gse = spiceypy.pxform(
            SpiceFrame.ECLIPJ2000.name, SpiceFrame.IMAP_GSE.name, target_time
        )
        expected_gse = spiceypy.mxv(rot_ecl_to_gse, inertial_vector)
        rot_ecl_to_gsm = spiceypy.pxform(
            SpiceFrame.ECLIPJ2000.name, SpiceFrame.IMAP_GSM.name, target_time
        )
        expected_gsm = spiceypy.mxv(rot_ecl_to_gsm, inertial_vector)
        rot_ecl_to_rtn = spiceypy.pxform(
            SpiceFrame.ECLIPJ2000.name, SpiceFrame.IMAP_RTN.name, target_time
        )
        expected_rtn = spiceypy.mxv(rot_ecl_to_rtn, inertial_vector)

    np.testing.assert_allclose(inertial_vector, expected_vector, atol=1e-05)
    np.testing.assert_allclose(gse_vector, expected_gse, atol=1e-05)
    np.testing.assert_allclose(gsm_vector, expected_gsm, atol=1e-05)
    np.testing.assert_allclose(rtn_vector, expected_rtn, atol=1e-05)


@pytest.mark.external_test_data
def test_process_packet(
    sc_packet_path, calibration_dataset, ialirt_mag_test_l1d_data, furnish_kernels
):
    """Test the process_packet function."""

    kernels = [
        "imap_science_100.tf",
        "imap_130.tf",
        "naif0012.tls",
        "de440s.bsp",
        "imap_spk_demo.bsp",
        "pck00011.tpc",
        "imap_sclk_0000.tsc",
    ]

    packet_path, xtce_ialirt_path = sc_packet_path
    sc_xarray_data = packet_file_to_datasets(
        packet_path, xtce_ialirt_path, use_derived_value=False
    )[478]
    with furnish_kernels(kernels):
        mag_data = process_packet(
            sc_xarray_data, calibration_dataset, ialirt_mag_test_l1d_data
        )

    assert isinstance(mag_data[0], dict)
    expected_keys = {
        "apid",
        "met",
        "met_in_utc",
        "ttj2000ns",
        "mag_epoch",
        "mag_B_GSE",
        "mag_B_GSM",
        "mag_B_RTN",
        "mag_B_magnitude",
        "mag_phi_B_GSM",
        "mag_theta_B_GSM",
        "mag_phi_B_GSE",
        "mag_theta_B_GSE",
    }
    assert expected_keys.issubset(mag_data[0].keys())
