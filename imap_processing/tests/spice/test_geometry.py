"""Tests coverage for imap_processing/spice/geometry.py"""

import os
from pathlib import Path

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

    # Uncomment this once spin phase calculation is implemented
    # assert np.all(
    #     (spin_phases >= 0) & (spin_phases <= 1)
    # ), "Spin phases must be in [0, 1] range."

    # Ensure the length of spin phases matches the query times
    assert len(spin_phases) == 1, "Spin phases length should match query times length."


def test_get_spin_data(generate_spin_data, tmpdir):
    """Test get_spin_data() with generated spin data."""

    # SWE test data time minus 56120 seconds to get mid-night time
    start_time = 453051323.0 - 56120
    spin_df = generate_spin_data(start_time)
    spin_csv_file_path = Path(tmpdir) / "spin_data.spin.csv"
    spin_df.to_csv(spin_csv_file_path, index=False)
    os.environ["SPIN_DATA_FILEPATH"] = str(spin_csv_file_path)

    spin_data = get_spin_data()

    assert len(spin_data.keys()) == 8, "Spin data must have 8 fields."
    assert len(spin_data) == 5760, "One day should have 5,760 records of 15 seconds."
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
