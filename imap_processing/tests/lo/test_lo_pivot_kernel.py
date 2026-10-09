"""Tests for the IMAP-Lo pivot platform kernel generation."""

from pathlib import Path

import numpy as np
import pytest
import spiceypy
import xarray as xr

from imap_processing.lo import lo_pivot_kernel
from imap_processing.lo.lo_pivot_kernel import (
    calculate_pivot_segment,
    generate_lo_pivot_kernel,
)
from imap_processing.spice.geometry import SpiceFrame, instrument_pointing
from imap_processing.spice.repoint import get_pointing_times_from_id
from imap_processing.spice.time import (
    et_to_met,
    met_to_sclkticks,
    met_to_ttj2000ns,
    sct_to_et,
    str_to_et,
)


@pytest.fixture
def furnish_lo_pivot_kernels(furnish_kernels):
    """Furnish the kernels needed to write and read the Lo pivot kernel."""
    with furnish_kernels(["naif0012.tls", "imap_sclk_0000.tsc", "imap_140.tf"]):
        yield


def make_nhk(met: np.ndarray, pivot: np.ndarray) -> xr.Dataset:
    """Make a minimal Lo L1B NHK dataset."""
    return xr.Dataset(
        {"pcc_coarse_pot_pri": ("epoch", np.asarray(pivot, dtype=np.float64))},
        coords={"epoch": met_to_ttj2000ns(np.asarray(met, dtype=np.float64))},
    )


@pytest.fixture
def pointings(furnish_lo_pivot_kernels, use_fake_repoint_data_for_time):
    """
    Fake repoint table with two short pointings on 2025-11-10 (100 and 101) and
    a pointing running from 2025-11-10 into 2025-11-11 (102).

    Returns the start and end MET of each pointing, keyed by repoint id.
    """
    t0 = float(et_to_met(str_to_et("2025-11-10T01:00:00")))
    repoint_starts = t0 + np.array([0, 6, 12, 36]) * 3600.0
    use_fake_repoint_data_for_time(repoint_starts, repoint_id_start=100)
    return {
        repoint_id: get_pointing_times_from_id(repoint_id)
        for repoint_id in [100, 101, 102]
    }


@pytest.fixture
def nhk_files(pointings, monkeypatch, tmp_path):
    """
    Lo L1B NHK files for each pointing, served by a mocked load_cdf.

    Each NHK file samples the pivot every 10 s from 10 minutes before the
    pointing (the pivot move) to the end of the pointing.
    """
    pivots = {100: 90.0, 101: 75.0, 102: 105.0}
    datasets = {}
    paths = {}
    for repoint_id, (start_met, end_met) in pointings.items():
        met = np.arange(start_met - 600, end_met, 10.0)
        pivot = np.where(met < start_met, 60.0, pivots[repoint_id])
        name = f"imap_lo_l1b_nhk_20251110-repoint{repoint_id:05d}_v001.cdf"
        datasets[name] = make_nhk(met, pivot)
        paths[repoint_id] = tmp_path / name
    monkeypatch.setattr(
        lo_pivot_kernel, "load_cdf", lambda path: datasets[Path(path).name]
    )
    return {"paths": paths, "pivots": pivots}


def test_calculate_pivot_segment(nhk_files, pointings):
    """One segment with the pointing coverage and the pointing's pivot angle."""
    segment, pivot_angle = calculate_pivot_segment(nhk_files["paths"][101], 101)
    assert segment.shape == (1,)
    assert segment[0]["pointing_id"] == 101
    assert pivot_angle == 75.0
    np.testing.assert_allclose(
        [segment[0]["start_sclk_ticks"], segment[0]["end_sclk_ticks"]],
        met_to_sclkticks(np.array(pointings[101])),
        rtol=0,
        atol=1,
    )


def read_comments(kernel_path: Path) -> str:
    """Read the comment area of a CK."""
    handle = spiceypy.dafopr(str(kernel_path))
    try:
        _, comments, _ = spiceypy.dafec(handle, 30, 200)
    finally:
        spiceypy.dafcls(handle)
    return "\n".join(comments)


def test_generate_lo_pivot_kernel(nhk_files, pointings, tmp_path):
    """The kernel covers the pointing with the IMAP_LO attitude."""
    kernel_paths = generate_lo_pivot_kernel(nhk_files["paths"][101], "repoint00101", 3)
    assert kernel_paths == [
        tmp_path / "imap/spice/ck/imap_lopivot-repoint00101_2025_314_2025_314_003.bc"
    ]
    kernel_path = kernel_paths[0]

    # Coverage is exactly the pointing, excluding the repoint maneuvers.
    cover = spiceypy.ckcov(
        str(kernel_path), SpiceFrame.IMAP_LO.value, False, "INTERVAL", 0, "SCLK"
    )
    assert spiceypy.wncard(cover) == 1
    start_met, end_met = pointings[101]
    np.testing.assert_allclose(
        spiceypy.wnfetd(cover, 0),
        met_to_sclkticks(np.array([start_met, end_met])),
        rtol=0,
        atol=1,
    )

    spiceypy.furnsh(str(kernel_path))
    try:
        # Across the pointing, the IMAP_LO boresight is the pivot angle away
        # from spacecraft +Z.
        for et in sct_to_et(met_to_sclkticks(np.linspace(start_met, end_met, 3))):
            boresight_sc = instrument_pointing(
                et, SpiceFrame.IMAP_LO, SpiceFrame.IMAP_SPACECRAFT, cartesian=True
            )
            np.testing.assert_allclose(
                np.rad2deg(np.arccos(boresight_sc[2])), 75.0, atol=1e-8
            )
    finally:
        spiceypy.unload(str(kernel_path))

    comments = read_comments(kernel_path)
    assert kernel_path.name in comments
    assert "imap_lo_l1b_nhk_20251110-repoint00101_v001.cdf" in comments
    assert "Repoint ID: repoint00101" in comments
    assert "Pivot angle (deg): 75.0000" in comments


def test_generate_lo_pivot_kernel_same_day_pointings(nhk_files):
    """Short pointings on the same day produce distinct kernels."""
    kernel_100 = generate_lo_pivot_kernel(nhk_files["paths"][100], "repoint00100", 1)
    kernel_101 = generate_lo_pivot_kernel(nhk_files["paths"][101], "repoint00101", 1)
    assert kernel_100[0].name == "imap_lopivot-repoint00100_2025_314_2025_314_001.bc"
    assert kernel_101[0].name == "imap_lopivot-repoint00101_2025_314_2025_314_001.bc"
    assert "Pivot angle (deg): 90.0000" in read_comments(kernel_100[0])


def test_generate_lo_pivot_kernel_multi_day_pointing(nhk_files):
    """The end date in the filename is the end of the pointing."""
    kernel_path = generate_lo_pivot_kernel(nhk_files["paths"][102], "repoint00102", 1)
    assert kernel_path[0].name == "imap_lopivot-repoint00102_2025_314_2025_315_001.bc"


def test_generate_lo_pivot_kernel_exists(nhk_files):
    """An existing kernel is not overwritten or appended to."""
    nhk_path = nhk_files["paths"][100]
    kernel_path = generate_lo_pivot_kernel(nhk_path, "repoint00100", 1)[0]
    original = kernel_path.read_bytes()
    with pytest.raises(FileExistsError, match=kernel_path.name):
        generate_lo_pivot_kernel(nhk_path, "repoint00100", 1)
    assert kernel_path.read_bytes() == original


def test_generate_lo_pivot_kernel_wrong_repointing(nhk_files):
    """The NHK file must be for the requested pointing."""
    with pytest.raises(ValueError, match="is not an NHK file for repoint00101"):
        generate_lo_pivot_kernel(nhk_files["paths"][100], "repoint00101", 1)


def test_calculate_pivot_segment_no_samples(nhk_files, monkeypatch):
    """No valid pivot samples is an error, not a 90 deg default."""
    nhk = make_nhk(np.arange(0, 3600 * 2, 10.0), np.full(720, np.nan))
    monkeypatch.setattr(lo_pivot_kernel, "load_cdf", lambda path: nhk)
    with pytest.raises(ValueError, match="No valid pivot angle samples"):
        calculate_pivot_segment(nhk_files["paths"][100], 100)


def test_calculate_pivot_segment_empty_nhk(nhk_files, monkeypatch):
    """An NHK file with no records is the same error as no valid samples."""
    nhk = make_nhk(np.array([]), np.array([]))
    monkeypatch.setattr(lo_pivot_kernel, "load_cdf", lambda path: nhk)
    with pytest.raises(ValueError, match="No valid pivot angle samples"):
        calculate_pivot_segment(nhk_files["paths"][100], 100)


def test_generate_lo_pivot_kernel_write_failure(nhk_files, monkeypatch, tmp_path):
    """A failed write leaves no partial kernel, so a retry succeeds."""

    def failing_ckw02(*args, **kwargs):
        raise spiceypy.utils.exceptions.SpiceyError("simulated write failure")

    ckopn = spiceypy.ckopn
    handles = []

    def recording_ckopn(*args):
        handles.append(ckopn(*args))
        return handles[-1]

    with monkeypatch.context() as m:
        m.setattr(spiceypy, "ckw02", failing_ckw02)
        m.setattr(spiceypy, "ckopn", recording_ckopn)
        with pytest.raises(
            spiceypy.utils.exceptions.SpiceyError, match="simulated write failure"
        ):
            generate_lo_pivot_kernel(nhk_files["paths"][100], "repoint00100", 1)

    # The CK was closed, which Windows needs to delete the temporary file.
    with pytest.raises(spiceypy.utils.exceptions.SpiceyError):
        spiceypy.dafhsf(handles[0])

    ck_dir = tmp_path / "imap/spice/ck"
    assert list(ck_dir.iterdir()) == []

    kernel_path = generate_lo_pivot_kernel(nhk_files["paths"][100], "repoint00100", 1)
    assert [p.name for p in ck_dir.iterdir()] == [kernel_path[0].name]


def test_generate_lo_pivot_kernel_appears_during_write(nhk_files, monkeypatch):
    """A kernel created by another process mid-write is not overwritten."""
    write = lo_pivot_kernel.write_lo_pivot_ck
    final_path = {}

    def write_then_race(kernel_path, *args):
        write(kernel_path, *args)
        final_path["path"] = kernel_path.parent.parent / kernel_path.name
        final_path["path"].write_bytes(b"other process")

    monkeypatch.setattr(lo_pivot_kernel, "write_lo_pivot_ck", write_then_race)
    with pytest.raises(FileExistsError):
        generate_lo_pivot_kernel(nhk_files["paths"][100], "repoint00100", 1)
    assert final_path["path"].read_bytes() == b"other process"
    assert [p.name for p in final_path["path"].parent.iterdir()] == [
        final_path["path"].name
    ]
