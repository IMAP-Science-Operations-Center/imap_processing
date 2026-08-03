"""Test suite for IMAP-Lo L2 map processing."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.ena_maps.ena_maps import match_coords_to_indices
from imap_processing.ena_maps.utils.naming import MapDescriptor
from imap_processing.lo.constants import LoConstants
from imap_processing.lo.l2.lo_l2 import (
    LoSpinAnglePointingSet,
    _bootstrap_correct_intensity,
    _complete_pointings,
    _dps_spin_angles,
    _esa_calibration,
    _extrapolate_top_intensity,
    _spin_phase_mask,
    lo_l2,
)
from imap_processing.spice.time import met_to_ttj2000ns

ANCILLARY_DIR = imap_module_directory / "tests/lo/test_anc"

# A full-spin map, so that every spin-angle bin lands on it. The "ns" and
# "nbs" after "ena" ask for neither the sputter nor the bootstrap correction,
# so these are the uncorrected maps.
FULL_DESCRIPTOR = "l090-enansnbs-h-sf-nsp-full-hae-6deg-3mo"
RAM_DESCRIPTOR = "l090-enansnbs-h-sf-nsp-ram-hae-6deg-3mo"

# The same full-spin map, sputter corrected only.
SPUTTER_DESCRIPTOR = "l090-enasnbs-h-sf-nsp-full-hae-6deg-3mo"

# The same full-spin map, bootstrap corrected only.
BOOTSTRAP_DESCRIPTOR = "l090-enansbs-h-sf-nsp-full-hae-6deg-3mo"

# The contents of imap_lo_sputter-correction-factors-small, as
# {target ESA step: {source ESA step: factor}}, 1-based as in the ancillary.
SPUTTER_FACTORS = {2: {3: 0.5}, 5: {3: 0.25, 6: 0.1}}

# The contents of imap_lo_bootstrap-correction-factors-small, as
# {target ESA step: {source ESA step: coefficient}}, 1-based as in the
# ancillary. Step 8 is the virtual ESA step above the top of the map.
BOOTSTRAP_FACTORS = {2: {3: 0.4, 5: 0.2}, 6: {7: 0.5}, 7: {8: 0.6}}

N_ESA = LoConstants.N_ESA_LEVELS
N_SPIN_BINS = LoConstants.N_SPIN_ANGLE_BINS
PIVOT = 90.0

# Good-time window [MET seconds] that the "in-window" histogram epochs fall in.
GT_START = 511_000_000.0
GT_END = 511_000_600.0
IN_METS = [511_000_150.0, 511_000_200.0, 511_000_250.0]
OUT_METS = [510_990_000.0, 511_010_000.0]

# The ESA level energies [keV] of imap_lo_hydrogen-geometric-factor-small, by
# ESA mode, which the map takes its energy binning from.
ESA_ENERGIES = {
    0: np.array([0.010, 0.020, 0.040, 0.080, 0.160, 0.320, 0.640]),
    1: np.array([0.011, 0.022, 0.044, 0.088, 0.176, 0.352, 0.704]),
}

# The hydrogen geometric factors of the same file, by ESA mode.
GEO_FACTORS = {
    0: np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) * 1e-5,
    1: np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]) * 1e-5,
}

# The ESA passband half-widths of the same file, a tenth of each center energy.
ESA_ENERGY_DELTAS = {mode: energies / 10 for mode, energies in ESA_ENERGIES.items()}


@pytest.fixture(autouse=True)
def use_test_geometric_factors():
    """Point the map at the small geometric factor ancillary in ``test_anc``.

    The map reads its geometric factors, and the ESA level energies they were
    measured at, straight out of the ancillary shipped with the package. The
    test file stands in for it so the tests do not have to track the flight
    calibration.
    """
    with patch("imap_processing.lo.l2.lo_l2.ANCILLARY_DATA_DIR", ANCILLARY_DIR):
        yield


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
            # The sputter correction reads the oxygen counts of the same
            # pointing. Making them the hydrogen counts lets a test predict the
            # correction from the hydrogen counts the uncorrected map reports.
            "o_counts": (["epoch", "esa_step", "spin_bin_6"], counts),
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
    energy = np.arange(1.0, N_ESA + 1.0)
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
            energy,
        )


@pytest.fixture
def one_pointing():
    """A single synthetic pointing."""
    return make_pointing()


@pytest.fixture
def anc_dependencies():
    """The ancillary files a map takes as a dependency.

    Every map but a raw one is flux corrected, so the ESA eta fit factors are
    required; without them the map cannot be made.
    """
    return [ANCILLARY_DIR / "imap_lo_esa-eta-fit-factors_20240101_v001.csv"]


@pytest.fixture
def full_map(one_pointing, anc_dependencies):
    """The full-spin map of one pointing, with the sky pointing mocked."""
    with patch(
        "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
        side_effect=identity_pointing,
    ):
        (dataset,) = lo_l2(
            as_dependencies(one_pointing), anc_dependencies, FULL_DESCRIPTOR
        )
    return dataset, one_pointing


@pytest.fixture
def sputter_maps(one_pointing, anc_dependencies):
    """The sputter corrected and uncorrected maps of the same pointing.

    The two differ only in whether the correction was applied, so the
    uncorrected map supplies the counts the correction is predicted from.
    """
    with patch(
        "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
        side_effect=identity_pointing,
    ):
        (corrected,) = lo_l2(
            as_dependencies(one_pointing), anc_dependencies, SPUTTER_DESCRIPTOR
        )
        (raw,) = lo_l2(as_dependencies(one_pointing), anc_dependencies, FULL_DESCRIPTOR)
    return corrected, raw


@pytest.fixture
def bootstrap_maps(one_pointing, anc_dependencies):
    """The bootstrap corrected and uncorrected maps of the same pointing.

    The two differ only in whether the correction was applied, so the
    uncorrected map supplies the intensities the correction is predicted from.
    """
    with patch(
        "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
        side_effect=identity_pointing,
    ):
        (corrected,) = lo_l2(
            as_dependencies(one_pointing), anc_dependencies, BOOTSTRAP_DESCRIPTOR
        )
        (raw,) = lo_l2(as_dependencies(one_pointing), anc_dependencies, FULL_DESCRIPTOR)
    return corrected, raw


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
        """The energy coordinate and its widths both come from the ancillary."""
        dataset, _ = full_map

        # The pointings are all in ESA mode 0.
        np.testing.assert_allclose(dataset["energy"].values, ESA_ENERGIES[0])
        np.testing.assert_allclose(
            dataset["energy_delta_minus"].values, ESA_ENERGY_DELTAS[0]
        )
        np.testing.assert_allclose(
            dataset["energy_delta_plus"].values, ESA_ENERGY_DELTAS[0]
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

    def test_pointings_accumulate(self, one_pointing, anc_dependencies):
        """Two pointings contribute twice the counts of one."""
        other = make_pointing(repointing=101, seed=7)

        with patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
            side_effect=identity_pointing,
        ):
            (one,) = lo_l2(
                as_dependencies(one_pointing), anc_dependencies, FULL_DESCRIPTOR
            )
            (both,) = lo_l2(
                as_dependencies(one_pointing, other), anc_dependencies, FULL_DESCRIPTOR
            )

        np.testing.assert_allclose(
            both["ena_count"].values.sum(axis=(0, 2, 3)),
            one["ena_count"].values.sum(axis=(0, 2, 3)) + other["expected_counts"],
        )

    def test_ram_map_keeps_half_the_spin(self, one_pointing, anc_dependencies):
        """A ram map takes fewer counts than the full spin it is cut from."""
        with patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
            side_effect=identity_pointing,
        ):
            (full,) = lo_l2(
                as_dependencies(one_pointing), anc_dependencies, FULL_DESCRIPTOR
            )
            (ram,) = lo_l2(
                as_dependencies(one_pointing), anc_dependencies, RAM_DESCRIPTOR
            )

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

        geometric_factor = GEO_FACTORS[0]
        energy = ESA_ENERGIES[0]
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


class TestSputterCorrection:
    """Removing the counts oxygen sputters into the hydrogen channels."""

    @staticmethod
    def sputtered(counts, target_esa, power=1):
        """The counts sputtered into a target ESA step, from the source steps.

        ``power`` is 1 for the counts themselves and 2 for their variance,
        which each source term contributes to scaled by the square of its
        factor.
        """
        return sum(
            factor**power * counts[:, source - 1]
            for source, factor in SPUTTER_FACTORS.get(target_esa, {}).items()
        )

    def test_correction_removes_the_sputtered_counts(self, sputter_maps):
        """The rate is the counts less the sputtered ones, over the exposure."""
        corrected, raw = sputter_maps

        counts = raw["ena_count"].values
        exposure = raw["exposure_factor"].values

        for target_esa in range(1, N_ESA + 1):
            exposed = exposure[:, target_esa - 1] > 0
            expected = np.maximum(
                counts[:, target_esa - 1] - self.sputtered(counts, target_esa), 0.0
            )
            np.testing.assert_allclose(
                corrected["ena_count_rate"].values[:, target_esa - 1][exposed],
                expected[exposed] / exposure[:, target_esa - 1][exposed],
                rtol=1e-5,
            )

    def test_uncorrected_steps_are_untouched(self, sputter_maps):
        """A step that nothing sputters into keeps the rate it already had."""
        corrected, raw = sputter_maps

        untouched = [esa for esa in range(1, N_ESA + 1) if esa not in SPUTTER_FACTORS]
        assert untouched, "the test factors must leave some steps uncorrected"

        for target_esa in untouched:
            np.testing.assert_allclose(
                corrected["ena_count_rate"].values[:, target_esa - 1],
                raw["ena_count_rate"].values[:, target_esa - 1],
                rtol=1e-6,
            )

    def test_corrected_steps_lose_intensity(self, sputter_maps):
        """The corrected steps come out below the uncorrected ones somewhere."""
        corrected, raw = sputter_maps

        for target_esa in SPUTTER_FACTORS:
            correction = (
                raw["ena_intensity"].values[:, target_esa - 1]
                - corrected["ena_intensity"].values[:, target_esa - 1]
            )
            assert np.all(correction >= 0)
            assert np.any(correction > 0), f"ESA {target_esa} was not corrected"

    def test_uncertainty_gains_the_source_counts(self, sputter_maps):
        """Subtracting a measured quantity can only add to the variance."""
        corrected, raw = sputter_maps

        counts = raw["ena_count"].values
        exposure = raw["exposure_factor"].values

        for target_esa in range(1, N_ESA + 1):
            exposed = exposure[:, target_esa - 1] > 0
            variance = counts[:, target_esa - 1] + self.sputtered(
                counts, target_esa, power=2
            )
            np.testing.assert_allclose(
                corrected["ena_count_rate_stat_uncert"].values[:, target_esa - 1][
                    exposed
                ],
                np.sqrt(variance[exposed]) / exposure[:, target_esa - 1][exposed],
                rtol=1e-5,
            )

    def test_rate_is_never_negative(self, sputter_maps):
        """Over-subtracting a low-count pixel floors it rather than going below 0."""
        corrected, raw = sputter_maps

        counts = raw["ena_count"].values
        # The test pointing is sparse enough that some pixel is over-subtracted,
        # which is the case this floor exists for.
        over_subtracted = [
            counts[:, esa - 1] - self.sputtered(counts, esa) < 0
            for esa in SPUTTER_FACTORS
        ]
        assert np.any(over_subtracted), "no pixel exercised the floor"

        assert np.all(corrected["ena_count_rate"].values >= 0)
        assert np.all(corrected["ena_intensity"].values >= 0)

    def test_counts_stay_as_observed(self, sputter_maps):
        """The correction applies from the rate onward, not to the raw counts."""
        corrected, raw = sputter_maps

        np.testing.assert_array_equal(
            corrected["ena_count"].values, raw["ena_count"].values
        )
        np.testing.assert_array_equal(
            corrected["exposure_factor"].values, raw["exposure_factor"].values
        )


class TestBootstrapCorrection:
    """Removing the intensity that bled down from the higher ESA steps."""

    # The ESA steps the test ancillary corrects from steps of the map itself,
    # rather than from the virtual step above the top of it.
    MAPPED_SOURCE_STEPS = (2, 6)

    @staticmethod
    def bled(intensity, target_esa, power=1):
        """The intensity bled into a target ESA step, from the source steps.

        ``power`` is 1 for the intensity itself and 2 for its variance, which
        each source term contributes to scaled by the square of its
        coefficient.
        """
        return sum(
            (LoConstants.BOOTSTRAP_SCALE * coefficient) ** power
            * intensity[:, source - 1]
            for source, coefficient in BOOTSTRAP_FACTORS.get(target_esa, {}).items()
        )

    def test_correction_removes_the_bled_intensity(self, bootstrap_maps):
        """A corrected step loses the scaled intensity of the steps above it."""
        corrected, raw = bootstrap_maps

        intensity = raw["ena_intensity"].values
        for target_esa in self.MAPPED_SOURCE_STEPS:
            expected = np.maximum(
                intensity[:, target_esa - 1] - self.bled(intensity, target_esa), 0.0
            )
            np.testing.assert_allclose(
                corrected["ena_intensity"].values[:, target_esa - 1],
                expected,
                rtol=1e-4,
            )

    def test_uncorrected_steps_are_untouched(self, bootstrap_maps):
        """A step that nothing bleeds into keeps the intensity it already had."""
        corrected, raw = bootstrap_maps

        untouched = [esa for esa in range(1, N_ESA + 1) if esa not in BOOTSTRAP_FACTORS]
        assert untouched, "the test coefficients must leave some steps uncorrected"

        for target_esa in untouched:
            np.testing.assert_allclose(
                corrected["ena_intensity"].values[:, target_esa - 1],
                raw["ena_intensity"].values[:, target_esa - 1],
                rtol=1e-6,
            )

    def test_top_step_is_corrected_against_the_virtual_step(self, bootstrap_maps):
        """The top step has only the extrapolated step above it to lose to."""
        corrected, raw = bootstrap_maps

        top = N_ESA
        assert set(BOOTSTRAP_FACTORS[top]) == {N_ESA + 1}, (
            "the top step must be fed by the virtual step alone"
        )

        correction = (
            raw["ena_intensity"].values[:, top - 1]
            - corrected["ena_intensity"].values[:, top - 1]
        )
        assert np.all(correction >= 0)
        assert np.any(correction > 0), "the virtual step corrected nothing"

    def test_uncertainty_gains_the_source_intensities(self, bootstrap_maps):
        """Subtracting a measured quantity can only add to the variance."""
        corrected, raw = bootstrap_maps

        variance = raw["ena_intensity_stat_uncert"].values ** 2
        for target_esa in self.MAPPED_SOURCE_STEPS:
            expected = variance[:, target_esa - 1] + self.bled(
                variance, target_esa, power=2
            )
            np.testing.assert_allclose(
                corrected["ena_intensity_stat_uncert"].values[:, target_esa - 1],
                np.sqrt(expected),
                rtol=1e-4,
            )

    def test_intensity_is_never_negative(self, bootstrap_maps):
        """The corrected map holds no negative intensity."""
        corrected, _ = bootstrap_maps

        assert np.all(corrected["ena_intensity"].values >= 0)

    def test_over_subtraction_is_floored_at_zero(self):
        """Over-subtracting a pixel floors it rather than going below zero."""
        calibration = _esa_calibration("h", 0)
        intensity = np.ones((1, N_ESA, 4))
        # Every step loses twice its own intensity to the step above it.
        coefficients = np.zeros((N_ESA, N_ESA + 1))
        coefficients[np.arange(N_ESA), np.arange(1, N_ESA + 1)] = (
            2.0 / LoConstants.BOOTSTRAP_SCALE
        )

        corrected, _, plus, minus = _bootstrap_correct_intensity(
            intensity,
            np.ones_like(intensity),
            calibration,
            coefficients,
            (2, 2),
            calibration.geometric_factor_low[:, np.newaxis] > 0,
        )

        np.testing.assert_array_equal(corrected, np.zeros_like(corrected))
        assert np.all(plus >= 0)
        assert np.all(minus >= 0)

    def test_systematic_error_brackets_the_correction(self, bootstrap_maps):
        """The corrected steps keep a two-sided systematic error."""
        corrected, _ = bootstrap_maps

        plus = corrected["ena_intensity_sys_err_plus"].values
        minus = corrected["ena_intensity_sys_err_minus"].values
        symmetric = corrected["ena_intensity_sys_err"].values
        lit = corrected["ena_intensity"].values > 0

        assert np.all(plus >= 0)
        assert np.all(minus >= 0)
        assert np.all(plus[lit] > 0)
        np.testing.assert_allclose(
            symmetric[lit], np.sqrt(plus[lit] * minus[lit]), rtol=1e-4
        )
        # The correction is bracketed by a smaller and a larger subtraction, so
        # its systematic error is wider than the G-factor one it starts from.
        for target_esa in self.MAPPED_SOURCE_STEPS:
            assert np.any(plus[:, target_esa - 1] > minus[:, target_esa - 1])

    def test_counts_and_rates_stay_as_observed(self, bootstrap_maps):
        """The correction applies to the intensities alone."""
        corrected, raw = bootstrap_maps

        for variable in ("ena_count", "exposure_factor", "ena_count_rate"):
            np.testing.assert_array_equal(
                corrected[variable].values, raw[variable].values
            )


class TestVirtualStepExtrapolation:
    """Extrapolating the ESA step above the top of the map."""

    energy = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64])
    grid_shape = (4, 3)

    def intensity(self, second, top):
        """A map of the two top ESA steps, one epoch, on a 4x3 sky grid."""
        values = np.zeros((1, N_ESA, np.prod(self.grid_shape)))
        values[0, -2] = np.asarray(second, dtype=float).ravel()
        values[0, -1] = np.asarray(top, dtype=float).ravel()
        return values

    def test_power_law_between_the_top_two_steps(self):
        """A pixel with both steps lit extrapolates along its own spectrum."""
        gamma = 1.4
        top = np.full(np.prod(self.grid_shape), 3.0)
        second = top * (self.energy[-1] / self.energy[-2]) ** gamma

        extrapolated = _extrapolate_top_intensity(
            self.intensity(second, top), self.energy, self.grid_shape
        )

        np.testing.assert_allclose(
            extrapolated[0],
            top * LoConstants.ESA_8_ENERGY_RATIO**-gamma,
            rtol=1e-6,
        )

    def test_missing_spectrum_borrows_from_the_neighborhood(self):
        """A pixel with no spectrum of its own uses its neighbors' median."""
        gamma = 1.4
        top = np.full(np.prod(self.grid_shape), 3.0)
        second = top * (self.energy[-1] / self.energy[-2]) ** gamma
        # One pixel is dark in the second step, so it has no spectrum, but its
        # neighbors all share the same one.
        second[0] = 0.0

        extrapolated = _extrapolate_top_intensity(
            self.intensity(second, top), self.energy, self.grid_shape
        )

        np.testing.assert_allclose(
            extrapolated[0],
            top * LoConstants.ESA_8_ENERGY_RATIO**-gamma,
            rtol=1e-6,
        )

    def test_map_with_no_spectrum_falls_back_to_the_nominal_index(self):
        """With no pixel to learn a spectrum from, a nominal one stands in."""
        top = np.full(np.prod(self.grid_shape), 3.0)

        extrapolated = _extrapolate_top_intensity(
            self.intensity(np.zeros_like(top), top), self.energy, self.grid_shape
        )

        np.testing.assert_allclose(
            extrapolated[0],
            top
            * LoConstants.ESA_8_ENERGY_RATIO
            ** -LoConstants.BOOTSTRAP_DEFAULT_SPECTRAL_INDEX,
            rtol=1e-6,
        )


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

    def test_oxygen_not_supported(self, one_pointing, anc_dependencies):
        """Only hydrogen geometric factors are defined."""
        with pytest.raises(NotImplementedError, match="species o"):
            lo_l2(
                as_dependencies(one_pointing),
                anc_dependencies,
                "l090-ena-o-sf-nsp-full-hae-6deg-3mo",
            )

    def test_healpix_not_supported(self, one_pointing, anc_dependencies):
        """Lo makes rectangular maps only."""
        with pytest.raises(NotImplementedError, match="HEALPix"):
            lo_l2(
                as_dependencies(one_pointing),
                anc_dependencies,
                "l090-enansnbs-h-sf-nsp-full-hae-nside8-3mo",
            )
