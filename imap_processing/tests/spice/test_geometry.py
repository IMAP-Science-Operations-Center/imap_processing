"""Tests coverage for imap_processing/spice/geometry.py"""

import numpy as np
import pytest

from imap_processing.spice.geometry import (
    SpiceBody,
    generate_spin_data,
    get_spacecraft_spin_phase,
    get_spin_data,
    imap_state,
)


@pytest.mark.parametrize(
    "et",
    [
        798033670,
        np.linspace(798033670, 798033770),
    ],
)
def test_imap_state(et, use_test_metakernel):
    """Test coverage for imap_state()"""
    state = imap_state(et, observer=SpiceBody.EARTH)
    if hasattr(et, "__len__"):
        np.testing.assert_array_equal(state.shape, (len(et), 6))
    else:
        assert state.shape == (6,)


@pytest.mark.external_kernel()
@pytest.mark.metakernel("imap_ena_sim_metakernel.template")
def test_imap_state_ecliptic(use_test_metakernel):
    """Tests retrieving IMAP state in the ECLIPJ2000 frame"""
    state = imap_state(798033670)
    assert state.shape == (6,)


def test_get_spacecraft_spin_phase(generate_spin_data):
    """Test get_spacecraft_spin_phase() with generated spin data."""

    start_time = 453051323.0

    spin_phases = get_spacecraft_spin_phase(query_met_times=start_time)

    # Uncomment this once spin phase calculation is implemented
    # assert np.all(
    #     (spin_phases >= 0) & (spin_phases <= 1)
    # ), "Spin phases must be in [0, 1] range."

    # Ensure the length of spin phases matches the query times
    assert len(spin_phases) == 1, "Spin phases length should match query times length."


def test_get_spin_data():
    """Test get_spin_data() with generated spin data."""

    start_time = 453051323.0

    spin_data = get_spin_data(start_met=start_time)

    assert len(spin_data.keys()) == 8, "Spin data must have 8 fields."


def test_generate_spin_data():
    """Test generate_spin_data() with generated spin data."""

    start_time = 453051323.0
    end_time = 453051323.0 + 86400

    spin_data = generate_spin_data(start_time, end_time)

    assert len(spin_data.keys()) == 8, "Spin data must have 8 fields."
    assert len(spin_data["spin_number"]) > 0, "Spin number must have some values."
    assert len(spin_data["spin_start_sec"]) > 0, "Spin start sec must have some values."
    assert (
        len(spin_data["spin_start_subsec"]) > 0
    ), "Spin start subsec must have some values."
    assert (
        len(spin_data["spin_period_sec"]) > 0
    ), "Spin period sec must have some values."
    assert (
        len(spin_data["spin_period_valid"]) > 0
    ), "Spin period valid must have some values."
    assert (
        len(spin_data["spin_phas_valid"]) > 0
    ), "Spin phas valid must have some values."
    assert (
        len(spin_data["spin_period_source"]) > 0
    ), "Spin period source must have some values."
    assert (
        len(spin_data["thruster_firing"]) > 0
    ), "Thruster firing must have some values."
