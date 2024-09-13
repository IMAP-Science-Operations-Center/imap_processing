"""Tests coverage for imap_processing/spice/geometry.py"""

import numpy as np
import pandas as pd
import pytest

from imap_processing.spice.geometry import (
    SpiceBody,
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

    # Ensure the length of spin phases matches the query times
    assert len(spin_phases) == 1, "Spin phases length should match query times length."


@pytest.mark.usefixtures("_set_spin_data_filepath")
def test_get_spin_data():
    """Test get_spin_data() with generated spin data."""

    spin_data = get_spin_data()

    (
        np.testing.assert_array_equal(spin_data["spin_number"], np.arange(5761)),
        "One day should have 5,761 records of 15 seconds when including end_met.",
    )
    assert isinstance(spin_data, pd.DataFrame), "Return type must be pandas.DataFrame."

    assert set(spin_data.columns) == {
        "spin_number",
        "spin_start_sec",
        "spin_start_subsec",
        "spin_period_sec",
        "spin_period_valid",
        "spin_phas_valid",
        "spin_period_source",
        "thruster_firing",
    }, "Spin data must have the specified fields."
