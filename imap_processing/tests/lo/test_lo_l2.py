"""Test suite for IMAP-Lo L2 map processing."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.ena_maps.ena_maps import match_coords_to_indices
from imap_processing.ena_maps.utils.naming import MapDescriptor
from imap_processing.lo.constants import LoConstants
from imap_processing.lo.l2.lo_l2 import (
    LoSpinAnglePointingSet,
    _complete_pointings,
    _dps_spin_angles,
    _spin_phase_mask,
    lo_l2,
)
from imap_processing.spice.time import met_to_ttj2000ns

# A full-spin map, so that every spin-angle bin lands on it.
FULL_DESCRIPTOR = "l090-ena-h-sf-nsp-full-hae-6deg-3mo"
RAM_DESCRIPTOR = "l090-ena-h-sf-nsp-ram-hae-6deg-3mo"

N_ESA = LoConstants.N_ESA_LEVELS
N_SPIN_BINS = LoConstants.N_SPIN_ANGLE_BINS
PIVOT = 90.0

# Good-time window [MET seconds] that the "in-window" histogram epochs fall in.
GT_START = 511_000_000.0
GT_END = 511_000_600.0
IN_METS = [511_000_150.0, 511_000_200.0, 511_000_250.0]
OUT_METS = [510_990_000.0, 511_010_000.0]


def product_attrs(repointing, product):
    """The global attributes an L1B input of a pointing is written with."""

    return {
        "Repointing": f"repoint{repointing:05d}",
        "Logical_source": f"imap_lo_l1b_{product}",
    }


def make_pointing(repointing=100, pivot=PIVOT, seed=42):
    """Build the three synthetic L1B inputs of one pointing.

    The in-window epochs carry modest counts and exposure; the out-of-window
    epochs carry large values that good-time filtering must exclude.
    """
    mets = np.array(IN_METS + OUT_METS)
    in_idx = np.arange(len(IN_METS))
    out_idx = np.arange(len(IN_METS), mets.size)

    rng = np.random.default_rng(seed)
    counts = np.zeros((mets.size, N_ESA, N_SPIN_BINS))
    exposure = np.zeros_like(counts)
    for i in in_idx:
        counts[i] = rng.integers(0, 4, size=(N_ESA, N_SPIN_BINS)).astype(float)
        exposure[i] = 2.0 * (np.arange(N_ESA)[:, None] + 1)
    for i in out_idx:
        counts[i] = 999.0
        exposure[i] = 999.0

    histrates = xr.Dataset(
        {
            "h_counts": (["epoch", "esa_step", "spin_bin_6"], counts),
            "exposure_time_6deg": (["epoch", "esa_step", "spin_bin_6"], exposure),
            "esa_mode": ("epoch", np.zeros(mets.size, dtype=int)),
        },
        coords={"epoch": met_to_ttj2000ns(mets)},
        attrs=product_attrs(repointing, "histrates"),
    )
    goodtimes = xr.Dataset(
        {
            "pivot": ("epoch", [pivot]),
            "gt_start_met": ("epoch", [GT_START]),
            "gt_end_met": ("epoch", [GT_END]),
        },
        coords={"epoch": [0]},
        attrs=product_attrs(repointing, "goodtimes"),
    )
    background = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    bgrates = xr.Dataset(
        {"h_background_rates": (["epoch", "esa_step"], background[np.newaxis, :])},
        coords={"epoch": [0], "esa_step": np.arange(1, N_ESA + 1)},
        attrs=product_attrs(repointing, "bgrates"),
    )

    return {
        "repointing": repointing,
        "goodtimes": goodtimes,
        "bgrates": bgrates,
        "histrates": histrates,
        "expected_counts": counts[in_idx].sum(axis=(0, 2)),
        "expected_exposure": exposure[in_idx].sum(axis=(0, 2)),
        "background": background,
    }


def as_dependencies(*pointings, products=("goodtimes", "bgrates", "histrates")):
    """Turn pointings into the sci_dependencies lo_l2 takes.

    The CLI keys the inputs by repointing and then by product descriptor, see
    ``cli.Lo.do_processing``.
    """
    return {
        pointing["repointing"]: {product: pointing[product] for product in products}
        for pointing in pointings
    }


def identity_pointing(et, az_el, *args, **kwargs):
    """Stand in for the SPICE DPS transform: spin angle -> lon, off -> lat.

    Like the real ``frame_transform_az_el``, the singleton off-angle dimension
    is squeezed out.
    """
    return np.asarray(az_el)[:, 0, :]


def make_pointing_set(sky_map, spin_angles, pivot=PIVOT):
    """Build the in-memory pointing set of one pointing, sky pointing mocked."""
    values = {
        name: np.ones((N_ESA, spin_angles.size))
        for name in ("ena_count", "exposure_factor", "bg_rate_exposure")
    }
    with patch(
        "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
        side_effect=identity_pointing,
    ):
        return LoSpinAnglePointingSet(
            met_to_ttj2000ns(GT_START),
            pivot,
            spin_angles,
            values,
            sky_map.spice_reference_frame,
        )


@pytest.fixture
def one_pointing():
    """A single synthetic pointing."""
    return make_pointing()


@pytest.fixture
def full_map(one_pointing):
    """The full-spin map of one pointing, with the sky pointing mocked."""
    with patch(
        "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
        side_effect=identity_pointing,
    ):
        (dataset,) = lo_l2(as_dependencies(one_pointing), [], FULL_DESCRIPTOR)
    return dataset, one_pointing


class TestMapStructure:
    """The shape and contents of the produced map."""

    expected_variables = (
        "ena_count",
        "exposure_factor",
        "ena_count_rate",
        "ena_count_rate_stat_uncert",
        "ena_intensity",
        "ena_intensity_stat_uncert",
        "ena_intensity_sys_err",
        "ena_intensity_sys_err_plus",
        "ena_intensity_sys_err_minus",
        "bg_rate",
        "bg_rate_stat_uncert",
        "bg_intensity",
        "bg_intensity_stat_uncert",
    )

    def test_map_dims_and_source(self, full_map):
        """The map is a 6 degree rectangular map of the descriptor."""
        dataset, _ = full_map

        assert dataset.attrs["Logical_source"] == f"imap_lo_l2_{FULL_DESCRIPTOR}"
        assert dict(dataset.sizes) == {
            "epoch": 1,
            "energy": N_ESA,
            "longitude": 60,
            "latitude": 30,
        }

    def test_map_variables(self, full_map):
        """Every map variable is on the (epoch, energy, sky) grid."""
        dataset, _ = full_map

        for variable in self.expected_variables:
            assert variable in dataset.data_vars, f"missing {variable}"
            assert dataset[variable].dims == (
                "epoch",
                "energy",
                "longitude",
                "latitude",
            )

    def test_energy_coordinate(self, full_map):
        """The energy coordinate and its widths come from the ESA constants."""
        dataset, _ = full_map

        np.testing.assert_allclose(
            dataset["energy"].values, LoConstants.ESA_ENERGY[:N_ESA]
        )
        np.testing.assert_allclose(
            dataset["energy_delta_plus"].values, LoConstants.ESA_ENERGY_DELTA[0]
        )

    def test_map_writes_to_cdf(self, full_map):
        """The map can be written out as a valid CDF."""
        dataset, _ = full_map
        dataset.attrs["Data_version"] = "001.0001"
        dataset.attrs["Start_date"] = "20260101"

        cdf_path = write_cdf(dataset)

        assert cdf_path.exists()
        assert cdf_path.name == f"imap_lo_l2_{FULL_DESCRIPTOR}_20260101_v001.0001.cdf"
        assert dict(load_cdf(cdf_path).sizes) == dict(dataset.sizes)


class TestAccumulation:
    """What the map accumulates from its inputs."""

    def test_counts_are_conserved(self, full_map):
        """Every in-window count lands somewhere on the map."""
        dataset, pointing = full_map

        per_energy = dataset["ena_count"].values.sum(axis=(0, 2, 3))
        np.testing.assert_allclose(per_energy, pointing["expected_counts"])

    def test_exposure_is_conserved(self, full_map):
        """Every in-window second of exposure lands somewhere on the map."""
        dataset, pointing = full_map

        per_energy = dataset["exposure_factor"].values.sum(axis=(0, 2, 3))
        np.testing.assert_allclose(per_energy, pointing["expected_exposure"], rtol=1e-6)

    def test_out_of_goodtime_epochs_are_excluded(self, full_map):
        """The 999-per-bin epochs outside the good times do not reach the map."""
        dataset, pointing = full_map

        per_energy = dataset["ena_count"].values.sum(axis=(0, 2, 3))
        assert per_energy.max() < 999.0 * N_SPIN_BINS
        np.testing.assert_allclose(per_energy, pointing["expected_counts"])

    def test_pointings_accumulate(self, one_pointing):
        """Two pointings contribute twice the counts of one."""
        other = make_pointing(repointing=101, seed=7)

        with patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
            side_effect=identity_pointing,
        ):
            (one,) = lo_l2(as_dependencies(one_pointing), [], FULL_DESCRIPTOR)
            (both,) = lo_l2(as_dependencies(one_pointing, other), [], FULL_DESCRIPTOR)

        np.testing.assert_allclose(
            both["ena_count"].values.sum(axis=(0, 2, 3)),
            one["ena_count"].values.sum(axis=(0, 2, 3)) + other["expected_counts"],
        )

    def test_ram_map_keeps_half_the_spin(self, one_pointing):
        """A ram map takes fewer counts than the full spin it is cut from."""
        with patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
            side_effect=identity_pointing,
        ):
            (full,) = lo_l2(as_dependencies(one_pointing), [], FULL_DESCRIPTOR)
            (ram,) = lo_l2(as_dependencies(one_pointing), [], RAM_DESCRIPTOR)

        full_counts = full["ena_count"].values.sum()
        ram_counts = ram["ena_count"].values.sum()
        assert 0 < ram_counts < full_counts


class TestRatesAndIntensities:
    """The maths turning accumulated counts into intensities."""

    def test_rate_and_intensity(self, full_map):
        """Where exposed, rate = counts/exposure and intensity = rate/(G*E)."""
        dataset, _ = full_map

        counts = dataset["ena_count"].values
        exposure = dataset["exposure_factor"].values
        exposed = exposure > 0
        assert exposed.any()

        np.testing.assert_allclose(
            dataset["ena_count_rate"].values[exposed],
            counts[exposed] / exposure[exposed],
            rtol=1e-5,
        )
        assert np.all(dataset["ena_count_rate"].values[~exposed] == 0)

        geometric_factor = (
            np.array(LoConstants.GEO_FACTOR[:N_ESA]) * LoConstants.GEO_FACTOR_SCALE
        )
        energy = np.array(LoConstants.ESA_ENERGY[:N_ESA])
        expected = dataset["ena_count_rate"] / xr.DataArray(
            geometric_factor * energy, dims=["energy"]
        )
        np.testing.assert_allclose(
            dataset["ena_intensity"].values[exposed],
            expected.values[exposed],
            rtol=1e-5,
        )

    def test_statistical_uncertainty_is_poisson(self, full_map):
        """The rate uncertainty is the Poisson count error over the exposure."""
        dataset, _ = full_map

        counts = dataset["ena_count"].values
        exposure = dataset["exposure_factor"].values
        exposed = exposure > 0

        np.testing.assert_allclose(
            dataset["ena_count_rate_stat_uncert"].values[exposed],
            np.sqrt(counts[exposed]) / exposure[exposed],
            rtol=1e-5,
        )

    def test_background_rate(self, full_map):
        """The background rate is the input rate wherever the map was exposed."""
        dataset, pointing = full_map

        exposure = dataset["exposure_factor"].values
        for energy_index in range(N_ESA):
            exposed = exposure[0, energy_index] > 0
            bg_rate = dataset["bg_rate"].values[0, energy_index]
            np.testing.assert_allclose(
                bg_rate[exposed], pointing["background"][energy_index], rtol=1e-5
            )
            assert np.all(bg_rate[~exposed] == 0)

    def test_systematic_error_bounds(self, full_map):
        """The systematic error is bracketed by the G-factor excursions."""
        dataset, _ = full_map

        intensity = dataset["ena_intensity"].values
        plus = dataset["ena_intensity_sys_err_plus"].values
        minus = dataset["ena_intensity_sys_err_minus"].values
        symmetric = dataset["ena_intensity_sys_err"].values
        lit = intensity > 0

        assert np.all(plus[lit] > 0)
        assert np.all(minus[lit] > 0)
        # The symmetric error is the geometric mean of the two excursions
        np.testing.assert_allclose(
            symmetric[lit], np.sqrt(plus[lit] * minus[lit]), rtol=1e-4
        )
        # The lower G-factor bound gives the bigger flux excursion
        assert np.all(plus[lit] >= minus[lit])


class TestGeometry:
    """The spin-angle to sky-pixel geometry."""

    def test_dps_spin_angles_carry_the_offset(self):
        """The hardware spin bins are rotated onto the instrument frame."""
        angles = _dps_spin_angles()

        assert angles.size == N_SPIN_BINS
        # IMAP-Lo sits 60 degrees from the spacecraft spin pulse, so bin 0's
        # center (3 degrees) becomes 63 degrees in the despun frame.
        np.testing.assert_allclose(angles[0], 63.0)
        np.testing.assert_allclose(np.diff(np.sort(angles)), 6.0)

    def test_bins_land_in_the_pixel_they_are_nearest(self):
        """Each spin-angle bin is projected into the pixel it points into."""
        sky_map = MapDescriptor.from_string(FULL_DESCRIPTOR).to_empty_map()
        spin_angles = _dps_spin_angles()
        pointing_set = make_pointing_set(sky_map, spin_angles)

        pixels = match_coords_to_indices(pointing_set, sky_map).values

        # A direction never lands further than half a pixel from its pixel's
        # center, in either axis.
        centers = sky_map.az_el_points.values[pixels]
        directions = pointing_set.az_el_points.values
        half_pixel = sky_map.spacing_deg / 2
        assert np.all(np.abs(centers[:, 0] - directions[:, 0]) <= half_pixel)
        assert np.all(np.abs(centers[:, 1] - directions[:, 1]) <= half_pixel)

    def test_bins_of_a_spin_land_in_distinct_pixels(self):
        """The 60 six-degree spin bins fill a row of the six-degree map."""
        sky_map = MapDescriptor.from_string(FULL_DESCRIPTOR).to_empty_map()
        pointing_set = make_pointing_set(sky_map, _dps_spin_angles())

        pixels = match_coords_to_indices(pointing_set, sky_map).values

        assert len(np.unique(pixels)) == N_SPIN_BINS

    @pytest.mark.parametrize(
        "spin_phase, expected",
        [("full", 60), ("ram", 30), ("anti", 30)],
    )
    def test_spin_phase_mask(self, spin_phase, expected):
        """Ram and anti-ram split the spin; a full map keeps all of it."""
        angles = _dps_spin_angles()

        mask = _spin_phase_mask(angles, PIVOT, spin_phase)

        assert mask.sum() == expected

    def test_spin_phase_mask_rejects_unknown(self):
        """An unknown spin phase is an error, not a silently empty map."""
        with pytest.raises(ValueError, match="Invalid spin phase"):
            _spin_phase_mask(_dps_spin_angles(), PIVOT, "sideways")


class TestPointingSelection:
    """Reducing the grouped inputs to the pointings that can be mapped."""

    def test_complete_pointings_are_kept_in_product_order(self, one_pointing):
        """Each pointing's products are ordered goodtimes, bgrates, histrates."""
        other = make_pointing(repointing=101)

        pointings = _complete_pointings(as_dependencies(one_pointing, other))

        assert set(pointings) == {100, 101}
        assert pointings[100] == (
            one_pointing["goodtimes"],
            one_pointing["bgrates"],
            one_pointing["histrates"],
        )

    def test_incomplete_pointings_are_dropped(self, one_pointing, caplog):
        """A pointing missing one of the three products cannot be mapped."""
        incomplete = make_pointing(repointing=101)
        dependencies = as_dependencies(one_pointing) | as_dependencies(
            incomplete, products=("goodtimes", "histrates")
        )

        pointings = _complete_pointings(dependencies)

        assert set(pointings) == {100}
        assert "repoint00101" in caplog.text
        assert "bgrates" in caplog.text

    def test_missing_product_raises(self, one_pointing):
        """A map cannot be made without all three products."""
        dependencies = as_dependencies(one_pointing, products=("goodtimes", "bgrates"))

        with pytest.raises(KeyError, match="histrates"):
            _complete_pointings(dependencies)


class TestUnsupported:
    """Map flavours the Lo pipeline does not make."""

    def test_oxygen_not_supported(self, one_pointing):
        """Only hydrogen geometric factors are defined."""
        with pytest.raises(NotImplementedError, match="species o"):
            lo_l2(
                as_dependencies(one_pointing),
                [],
                "l090-ena-o-sf-nsp-full-hae-6deg-3mo",
            )

    def test_healpix_not_supported(self, one_pointing):
        """Lo makes rectangular maps only."""
        with pytest.raises(NotImplementedError, match="HEALPix"):
            lo_l2(
                as_dependencies(one_pointing),
                [],
                "l090-ena-h-sf-nsp-full-hae-nside8-3mo",
            )
