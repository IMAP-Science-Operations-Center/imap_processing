"""Test suite for IMAP-Lo L2 map processing."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.ena_maps.ena_maps import match_coords_to_indices
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.ena_maps.utils.corrections import PowerLawFluxCorrector
from imap_processing.ena_maps.utils.naming import MapDescriptor
from imap_processing.lo.constants import LoConstants
from imap_processing.lo.l2.lo_l2 import (
    ANCILLARY_DATA_DIR as PACKAGE_ANCILLARY_DIR,
)
from imap_processing.lo.l2.lo_l2 import (
    FILLED_VARIABLES,
    FILLVAL_FLOAT,
    ISN_MASKED_VARIABLES,
    LoSpinAnglePointingSet,
    _bootstrap_correct_intensity,
    _complete_pointings,
    _compton_getting_correct_intensity,
    _dps_spin_angles,
    _esa_calibration,
    _extrapolate_top_intensity,
    _spacecraft_frame_energy,
    _spin_phase_mask,
    finalize_dataset,
    lo_l2,
    load_bootstrap_correction_data,
    load_isn_mask_parameters,
    load_sputter_correction_data,
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

# The same map in the heliospheric frame, which is what asks for the
# Compton-Getting correction. Neither of the other corrections is made.
CG_DESCRIPTOR = "l090-enansnbs-h-hf-nsp-full-hae-6deg-3mo"

# The same uncorrected full-spin map, with the ISN band masked out. The "msk"
# follows the bootstrap code. The mask is tuned per pivot angle, which is the
# sensor of the descriptor, so each pivot gets its own descriptor: 90 masks the
# bright pixels, 75 has too narrow a band to mask any, and 105 masks the top
# half of each ESA level. See imap_lo_isn-mask-parameters-small_v001.csv.
MASK_DESCRIPTOR = "l090-enansnbsmsk-h-sf-nsp-full-hae-6deg-3mo"
NARROW_MASK_DESCRIPTOR = "l075-enansnbsmsk-h-sf-nsp-full-hae-6deg-3mo"
OUTLIER_MASK_DESCRIPTOR = "l105-enansnbsmsk-h-sf-nsp-full-hae-6deg-3mo"

# A pivot angle the mask ancillary carries no tuning for.
UNTUNED_MASK_DESCRIPTOR = "l060-enansnbsmsk-h-sf-nsp-full-hae-6deg-3mo"

# The combined maps, written "ilo" rather than with a pivot angle of their own.
# These select no pivot angle, so every pointing given to them is accumulated.
COMBINED_DESCRIPTOR = "ilo-enansnbs-h-sf-nsp-full-hae-6deg-3mo"
COMBINED_RAM_DESCRIPTOR = "ilo-enansnbs-h-sf-nsp-ram-hae-6deg-3mo"
COMBINED_MASK_DESCRIPTOR = "ilo-enansnbsmsk-h-sf-nsp-full-hae-6deg-3mo"

# The pivot angles a combined map is built from in these tests.
COMBINED_PIVOTS = (75.0, 90.0, 105.0)

# The fraction of an ESA level's peak intensity that the pivot 90 tuning masks
# from, and the variables it blanks out.
MASK_THRESHOLD_FRACTION = 0.5

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

# The fill value as it lands in the map, which writes its variables as float32.
FILL = np.float32(FILLVAL_FLOAT)


def is_fill(values):
    """Whether each element of a map variable holds the fill value."""
    return np.asarray(values) == FILL


def measured(values):
    """The elements of a map variable that are not filled."""
    values = np.asarray(values)
    return values[~is_fill(values)]


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

    A heliospheric frame map is Compton-Getting corrected, which is the only
    thing the ESA eta fit factors are read for.
    """
    return [ANCILLARY_DIR / "imap_lo_esa-eta-fit-factors_20240101_v001.csv"]


@pytest.fixture
def shipped_ancillaries():
    """Read the calibration ancillaries shipped with the package.

    Undoes the autouse ``use_test_geometric_factors`` patch (a more common use-case)
        for tests that need the shipped ancillaries.
    """
    with patch("imap_processing.lo.l2.lo_l2.ANCILLARY_DATA_DIR", PACKAGE_ANCILLARY_DIR):
        yield


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


@pytest.fixture
def cg_maps(one_pointing, anc_dependencies):
    """The Compton-Getting corrected and uncorrected maps of one pointing.

    The two differ only in the frame they are made in, which is what asks for
    the correction, so the spacecraft frame map supplies the intensities the
    correction is predicted from.
    """
    with patch(
        "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
        side_effect=identity_pointing,
    ):
        (corrected,) = lo_l2(
            as_dependencies(one_pointing), anc_dependencies, CG_DESCRIPTOR
        )
        (raw,) = lo_l2(as_dependencies(one_pointing), anc_dependencies, FULL_DESCRIPTOR)
    return corrected, raw


@pytest.fixture
def masked_maps(one_pointing, anc_dependencies):
    """The ISN masked and unmasked maps of the same pointing.

    The two differ only in whether the band was masked out, so the unmasked map
    supplies the intensities the mask is predicted from.
    """
    with patch(
        "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
        side_effect=identity_pointing,
    ):
        (masked,) = lo_l2(
            as_dependencies(one_pointing), anc_dependencies, MASK_DESCRIPTOR
        )
        (raw,) = lo_l2(as_dependencies(one_pointing), anc_dependencies, FULL_DESCRIPTOR)
    return masked, raw


@pytest.fixture
def pointings_at_every_pivot():
    """One synthetic pointing at each of the pivot angles a map combines."""
    return [
        make_pointing(repointing=100 + i, pivot=pivot, seed=i + 1)
        for i, pivot in enumerate(COMBINED_PIVOTS)
    ]


class TestCombinedMap:
    """A map that selects no pivot angle and takes every pointing it is given."""

    def test_every_pivot_angle_is_accumulated(
        self, pointings_at_every_pivot, anc_dependencies
    ):
        """The counts of all three pointings land on the one combined map."""
        with patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
            side_effect=identity_pointing,
        ):
            (combined,) = lo_l2(
                as_dependencies(*pointings_at_every_pivot),
                anc_dependencies,
                COMBINED_DESCRIPTOR,
            )

        expected = sum(
            pointing["expected_counts"].sum() for pointing in pointings_at_every_pivot
        )
        assert combined["ena_count"].values.sum() == pytest.approx(expected)

    def test_the_combined_map_is_the_sum_of_its_pointings(
        self, pointings_at_every_pivot, anc_dependencies
    ):
        """Accumulating together matches accumulating each pointing alone."""
        with patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
            side_effect=identity_pointing,
        ):
            (combined,) = lo_l2(
                as_dependencies(*pointings_at_every_pivot),
                anc_dependencies,
                COMBINED_DESCRIPTOR,
            )
            singly = [
                lo_l2(as_dependencies(pointing), anc_dependencies, COMBINED_DESCRIPTOR)[
                    0
                ]
                for pointing in pointings_at_every_pivot
            ]

        for variable in ("ena_count", "exposure_factor"):
            np.testing.assert_allclose(
                combined[variable].values,
                sum(single[variable].values for single in singly),
                rtol=1e-5,
                err_msg=variable,
            )

    def test_a_pointing_is_projected_from_its_own_pivot_angle(
        self, pointings_at_every_pivot, anc_dependencies
    ):
        """The pivot angles of a combined map need not agree with each other.

        A RAM map keeps the spin-angle bins looking into the RAM direction,
        which is a function of the pointing's own pivot angle, so the three
        pointings would not all survive one shared pivot angle.
        """
        with patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
            side_effect=identity_pointing,
        ):
            (combined,) = lo_l2(
                as_dependencies(*pointings_at_every_pivot),
                anc_dependencies,
                COMBINED_RAM_DESCRIPTOR,
            )
            singly = [
                lo_l2(
                    as_dependencies(pointing),
                    anc_dependencies,
                    COMBINED_RAM_DESCRIPTOR,
                )[0]
                for pointing in pointings_at_every_pivot
            ]

        # Each pointing keeps its own RAM half of the spin, and the combined
        # map is exactly those three halves added together.
        assert (combined["ena_count"].values > 0).any()
        np.testing.assert_allclose(
            combined["ena_count"].values,
            sum(single["ena_count"].values for single in singly),
            rtol=1e-5,
        )

    def test_a_pivot_angle_map_of_the_same_pointings_is_no_larger(
        self, pointings_at_every_pivot, anc_dependencies
    ):
        """The combined map holds at least what any single-pivot map holds.

        ``lo_l2`` itself does no pivot filtering, so a single-pivot descriptor
        given these same pointings accumulates them all too; the point here is
        that the combined descriptor loses nothing by naming no pivot angle.
        """
        with patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
            side_effect=identity_pointing,
        ):
            (combined,) = lo_l2(
                as_dependencies(*pointings_at_every_pivot),
                anc_dependencies,
                COMBINED_DESCRIPTOR,
            )
            (single_pivot,) = lo_l2(
                as_dependencies(*pointings_at_every_pivot),
                anc_dependencies,
                FULL_DESCRIPTOR,
            )

        np.testing.assert_allclose(
            combined["ena_count"].values, single_pivot["ena_count"].values, rtol=1e-5
        )

    def test_the_isn_mask_reads_the_pivot_angles_off_the_pointings(
        self, pointings_at_every_pivot, anc_dependencies
    ):
        """A combined map is masked with the union of its pivots' tunings.

        The pivot 105 tuning of the test ancillary masks the top half of each
        ESA level and the pivot 90 one the bright pixels, so a map holding both
        is masked at least everywhere either of them would mask.
        """
        with patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
            side_effect=identity_pointing,
        ):
            (combined,) = lo_l2(
                as_dependencies(*pointings_at_every_pivot),
                anc_dependencies,
                COMBINED_MASK_DESCRIPTOR,
            )
            (unmasked,) = lo_l2(
                as_dependencies(*pointings_at_every_pivot),
                anc_dependencies,
                COMBINED_DESCRIPTOR,
            )

        masked = is_fill(combined["ena_intensity"].values)
        intensity = unmasked["ena_intensity"].values
        unexposed = is_fill(intensity)

        # The union of the three tunings: the widest band (pivot 75's 1 degree
        # loses to the 90 degrees of the others), the faintest brightness taken
        # as bright, and the shortest outlier tail.
        as_masked = np.where(unexposed, 0.0, intensity)
        peak = np.max(as_masked, axis=(-2, -1), keepdims=True)
        median = np.percentile(as_masked, 50, axis=(-2, -1), keepdims=True)
        expected = (
            (as_masked >= MASK_THRESHOLD_FRACTION * peak)
            | (as_masked > median)
            | unexposed
        )

        assert expected.any(), "the tuning must mask something"
        np.testing.assert_array_equal(masked, expected)

    def test_a_combined_map_of_one_pivot_angle_uses_that_pivots_tuning(
        self, one_pointing, anc_dependencies
    ):
        """With a single pivot angle in the data the union is that pivot's row."""
        with patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
            side_effect=identity_pointing,
        ):
            # The pointing is at pivot 90, so this must match the map made with
            # the pivot 90 descriptor.
            (combined,) = lo_l2(
                as_dependencies(one_pointing),
                anc_dependencies,
                COMBINED_MASK_DESCRIPTOR,
            )
            (by_descriptor,) = lo_l2(
                as_dependencies(one_pointing), anc_dependencies, MASK_DESCRIPTOR
            )

        assert is_fill(combined["ena_intensity"].values).any()
        np.testing.assert_array_equal(
            is_fill(combined["ena_intensity"].values),
            is_fill(by_descriptor["ena_intensity"].values),
        )

    def test_an_unrecognisable_pivot_angle_cannot_tune_the_mask(
        self, anc_dependencies, caplog
    ):
        """A pivot angle matching no nominal one is dropped, and none is left."""
        off_nominal = make_pointing(repointing=100, pivot=42.0)

        with pytest.raises(ValueError, match="reports a nominal pivot angle"):
            lo_l2(
                as_dependencies(off_nominal),
                anc_dependencies,
                COMBINED_MASK_DESCRIPTOR,
            )
        assert "match none of the nominal pivot angles" in caplog.text

    def test_a_pivot_angle_the_mask_has_no_tuning_for_is_refused(
        self, anc_dependencies
    ):
        """A combined map holding an untuned pivot angle cannot be masked."""
        # 60 is a nominal pivot angle, but the mask ancillary has no row for it.
        untuned = make_pointing(repointing=100, pivot=60.0)

        with pytest.raises(ValueError, match=r"no mask tuning for the \[60\]"):
            lo_l2(as_dependencies(untuned), anc_dependencies, COMBINED_MASK_DESCRIPTOR)


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
        assert np.all(is_fill(dataset["ena_count_rate"].values[~exposed]))

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
            assert np.all(is_fill(bg_rate[~exposed]))

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

        assert np.all(measured(corrected["ena_count_rate"].values) >= 0)
        assert np.all(measured(corrected["ena_intensity"].values) >= 0)

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

        # The correction runs before the unexposed pixels are filled, so it saw
        # the zero they were divided down to.
        intensity = np.where(
            is_fill(raw["ena_intensity"].values), 0.0, raw["ena_intensity"].values
        )
        for target_esa in self.MAPPED_SOURCE_STEPS:
            expected = np.maximum(
                intensity[:, target_esa - 1] - self.bled(intensity, target_esa), 0.0
            )
            step = corrected["ena_intensity"].values[:, target_esa - 1]
            keep = ~is_fill(step)
            assert keep.any()
            np.testing.assert_allclose(step[keep], expected[keep], rtol=1e-4)

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

        uncert = raw["ena_intensity_stat_uncert"].values
        variance = np.where(is_fill(uncert), 0.0, uncert) ** 2
        for target_esa in self.MAPPED_SOURCE_STEPS:
            expected = variance[:, target_esa - 1] + self.bled(
                variance, target_esa, power=2
            )
            step = corrected["ena_intensity_stat_uncert"].values[:, target_esa - 1]
            keep = ~is_fill(step)
            assert keep.any()
            np.testing.assert_allclose(step[keep], np.sqrt(expected[keep]), rtol=1e-4)

    def test_intensity_is_never_negative(self, bootstrap_maps):
        """The corrected map holds no negative intensity."""
        corrected, _ = bootstrap_maps

        assert np.all(measured(corrected["ena_intensity"].values) >= 0)

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
            calibration["geometric_factor_low"].values[:, np.newaxis] > 0,
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

        assert np.all(measured(plus) >= 0)
        assert np.all(measured(minus) >= 0)
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


class TestComptonGettingCorrection:
    """Moving the intensities into the frame the heliosphere sees them in."""

    def test_the_eta_fit_factors_are_required(self, one_pointing):
        """A heliospheric frame map cannot be made without the transmission factors."""
        with pytest.raises(ValueError, match="ESA eta fit factors"):
            lo_l2(as_dependencies(one_pointing), [], CG_DESCRIPTOR)

    def test_a_spacecraft_frame_map_needs_no_eta_fit_factors(self, one_pointing):
        """The factors are read for the correction alone, so an sf map goes without."""
        with patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
            side_effect=identity_pointing,
        ):
            (dataset,) = lo_l2(as_dependencies(one_pointing), [], FULL_DESCRIPTOR)

        assert (dataset["ena_intensity"].values > 0).any()

    def test_counts_and_rates_stay_as_observed(self, cg_maps):
        """The correction applies to the intensities alone."""
        corrected, raw = cg_maps

        for variable in ("ena_count", "exposure_factor", "ena_count_rate"):
            np.testing.assert_array_equal(
                corrected[variable].values, raw[variable].values
            )

    def test_intensities_are_corrected(self, cg_maps):
        """The intensities and the background come out shifted."""
        corrected, raw = cg_maps

        for variable in ("ena_intensity", "bg_intensity"):
            lit = raw[variable].values > 0
            assert lit.any()
            assert np.any(
                ~np.isclose(corrected[variable].values[lit], raw[variable].values[lit])
            ), f"{variable} was not corrected"

    def test_the_correction_leaks_no_nan(self, cg_maps):
        """A pixel the correction says nothing about comes out at zero, not NaN."""
        corrected, _ = cg_maps

        for variable in (
            "ena_intensity",
            "ena_intensity_stat_uncert",
            "ena_intensity_sys_err_plus",
            "ena_intensity_sys_err_minus",
            "bg_intensity",
            "bg_intensity_stat_uncert",
        ):
            values = corrected[variable].values
            assert np.isfinite(values).all(), f"{variable} holds NaN or inf"
            assert np.all(measured(values) >= 0), f"{variable} went negative"

    def test_dark_pixels_stay_dark(self, cg_maps):
        """A pixel that saw nothing has no spectrum to correct."""
        corrected, raw = cg_maps

        intensity = raw["ena_intensity"].values
        dark = (intensity <= 0) & ~is_fill(intensity)
        assert dark.any()
        assert np.all(corrected["ena_intensity"].values[dark] == 0)

    def test_accumulator_is_not_written_out(self, cg_maps):
        """The projected cos(alpha) is an accumulator, not a map variable."""
        corrected, _ = cg_maps

        assert "cos_alpha_exposure" not in corrected.data_vars


class TestSpacecraftFrameEnergy:
    """The kinematics of an ENA seen from a moving spacecraft."""

    # The energy [eV] an ENA of the top ESA level has in the helio frame.
    energy = np.array([16.0, 30.0, 56.0, 106.0, 200.0, 404.0, 787.0])

    def spacecraft_energy(self, cos_alpha):
        """The spacecraft frame energies of one pixel at every ESA level."""
        alpha = np.full((1, N_ESA, 1), float(cos_alpha))
        return _spacecraft_frame_energy(alpha, self.energy)[0, :, 0]

    def test_head_on_ena_gains_the_spacecraft_speed(self):
        """Looking into the ram direction, the speeds add."""
        energy_u = LoConstants.CG_ENA_ENERGY_AT_SPACECRAFT_SPEED_EV

        np.testing.assert_allclose(
            self.spacecraft_energy(1.0),
            (np.sqrt(self.energy) + np.sqrt(energy_u)) ** 2,
            rtol=1e-12,
        )

    def test_overtaken_ena_loses_the_spacecraft_speed(self):
        """Looking away from the ram direction, the speeds subtract."""
        energy_u = LoConstants.CG_ENA_ENERGY_AT_SPACECRAFT_SPEED_EV

        np.testing.assert_allclose(
            self.spacecraft_energy(-1.0),
            (np.sqrt(self.energy) - np.sqrt(energy_u)) ** 2,
            rtol=1e-12,
        )

    def test_side_on_ena_loses_the_spacecraft_energy(self):
        """Looking across the ram direction, the energies subtract."""
        energy_u = LoConstants.CG_ENA_ENERGY_AT_SPACECRAFT_SPEED_EV

        np.testing.assert_allclose(
            self.spacecraft_energy(0.0), self.energy - energy_u, rtol=1e-12
        )

    def test_ram_side_is_the_energetic_one(self):
        """The shift grows monotonically towards the ram direction."""
        cosines = np.linspace(-1.0, 1.0, 21)
        energies = np.array([self.spacecraft_energy(c) for c in cosines])

        assert np.all(np.diff(energies, axis=0) > 0)


class TestComptonGettingMaths:
    """The correction applied to a spectrum whose answer is known."""

    energy_kev = np.array([0.016, 0.030, 0.056, 0.106, 0.200, 0.404, 0.787])
    spectral_index = -1.8

    @pytest.fixture
    def calibration(self):
        """A calibration carrying only the energies the correction reads."""
        ones = np.ones(N_ESA)
        arrays = dict(
            energy=self.energy_kev,
            energy_delta_minus=np.zeros(N_ESA),
            energy_delta_plus=np.zeros(N_ESA),
            geometric_factor=ones,
            geometric_factor_low=ones,
            geometric_factor_high=ones,
        )
        return xr.Dataset(
            {
                name: xr.DataArray(array, dims=[CoordNames.ENERGY_L2.value])
                for name, array in arrays.items()
            }
        )

    @pytest.fixture
    def flux_corrector(self):
        """The ESA transmission factors of the test eta fit ancillary."""
        return PowerLawFluxCorrector(
            ANCILLARY_DIR / "imap_lo_esa-eta-fit-factors_20240101_v001.csv"
        )

    @pytest.fixture
    def transparent_corrector(self, tmp_path):
        """A corrector whose every ESA level transmits perfectly.

        With no transmission to undo, the source spectrum is the observed one
        and the correction is the energy shift alone, which is what makes the
        expected values here analytic.
        """
        coefficients = tmp_path / "imap_lo_esa-eta-fit-factors_20240101_v999.csv"
        rows = "\n".join(f"{step},1,0,0,0,0,0" for step in range(1, N_ESA + 1))
        coefficients.write_text(f"esa_step,M0,M1,M2,M3,M4,M5\n{rows}\n")
        return PowerLawFluxCorrector(coefficients)

    def power_law(self, n_pixels):
        """A spectrum of exactly the test spectral index, at every pixel."""
        spectrum = 1e5 * (self.energy_kev / self.energy_kev[0]) ** self.spectral_index
        return np.tile(spectrum[np.newaxis, :, np.newaxis], (1, 1, n_pixels))

    def test_correction_follows_the_source_power_law(
        self, calibration, transparent_corrector
    ):
        """A power-law spectrum is scaled by the shift along its own slope."""
        cos_alpha = np.linspace(-1.0, 1.0, 12)[np.newaxis, np.newaxis, :]
        cos_alpha = np.tile(cos_alpha, (1, N_ESA, 1))
        intensity = self.power_law(cos_alpha.shape[-1])

        corrected, _ = _compton_getting_correct_intensity(
            intensity, (), cos_alpha, calibration, transparent_corrector
        )

        # The index of a pure power law is recovered exactly, and a perfectly
        # transmitting ESA leaves the source intensity as the observed one.
        energy_ev = self.energy_kev * 1e3
        energy_sc = _spacecraft_frame_energy(cos_alpha, energy_ev)
        expected = intensity * (energy_sc / energy_ev[:, np.newaxis]) ** (
            self.spectral_index + 1.0
        )

        np.testing.assert_allclose(corrected, expected, rtol=1e-10)

    def test_transmission_is_divided_out(
        self, calibration, flux_corrector, transparent_corrector
    ):
        """An ESA that over-transmits has to be corrected back down."""
        cos_alpha = np.full((1, N_ESA, 4), 0.5)
        intensity = self.power_law(4)

        corrected, _ = _compton_getting_correct_intensity(
            intensity, (), cos_alpha, calibration, flux_corrector
        )
        transparent, _ = _compton_getting_correct_intensity(
            intensity, (), cos_alpha, calibration, transparent_corrector
        )

        # The transmission of a falling spectrum is above one at every level,
        # so dividing it out leaves less intensity than a perfect ESA would.
        index = np.full((N_ESA, 1), self.spectral_index)
        assert np.all(flux_corrector.eta_esa(np.arange(N_ESA) + 1, index) > 1.0)
        assert np.all(corrected < transparent)

    def test_uncertainties_keep_their_fraction(self, calibration, flux_corrector):
        """Every uncertainty moves by the same factor as its intensity."""
        cos_alpha = np.tile(
            np.linspace(-1.0, 1.0, 12)[np.newaxis, np.newaxis, :], (1, N_ESA, 1)
        )
        intensity = self.power_law(cos_alpha.shape[-1])
        stat_uncert = 0.1 * intensity
        sys_err = 0.05 * intensity

        corrected, (corrected_stat, corrected_sys) = _compton_getting_correct_intensity(
            intensity,
            (stat_uncert, sys_err),
            cos_alpha,
            calibration,
            flux_corrector,
        )

        np.testing.assert_allclose(corrected_stat, 0.1 * corrected, rtol=1e-10)
        np.testing.assert_allclose(corrected_sys, 0.05 * corrected, rtol=1e-10)

    def test_unlit_levels_are_left_at_zero(self, calibration, flux_corrector):
        """A level with no intensity has no spectrum, and stays empty."""
        cos_alpha = np.full((1, N_ESA, 4), 0.5)
        intensity = self.power_law(4)
        intensity[0, :, 2] = 0.0

        corrected, (corrected_stat,) = _compton_getting_correct_intensity(
            intensity, (intensity,), cos_alpha, calibration, flux_corrector
        )

        np.testing.assert_array_equal(corrected[0, :, 2], np.zeros(N_ESA))
        np.testing.assert_array_equal(corrected_stat[0, :, 2], np.zeros(N_ESA))
        assert np.all(corrected[0, :, [0, 1, 3]] > 0)


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


class TestCorrectionFactors:
    """Reading the sputter and bootstrap correction ancillaries."""

    def test_sputter_factors_are_selected_by_species_pair(
        self, shipped_ancillaries, tmp_path
    ):
        """Only the rows of the requested pair come back, in ESA step order."""
        factors = load_sputter_correction_data()

        assert list(factors.columns) == [
            "source_species",
            "target_species",
            "target_esa",
            "source_esa",
            "sputter_factor",
        ]
        assert not factors.empty
        assert (factors["source_species"] == "o").all()
        assert (factors["target_species"] == "h").all()

        with patch("imap_processing.lo.l2.lo_l2.ANCILLARY_DATA_DIR", tmp_path):
            with pytest.raises(ValueError, match="No sputter correction files"):
                load_sputter_correction_data()

    def test_bootstrap_factors_relate_lower_steps_to_higher(
        self, shipped_ancillaries, tmp_path
    ):
        """Each factor carries a step pair, the source below the target."""
        factors = load_bootstrap_correction_data()

        assert list(factors.columns) == [
            "esa_step_i",
            "esa_step_k",
            "bootstrap_factor",
        ]
        assert not factors.empty
        assert (factors["esa_step_i"] < factors["esa_step_k"]).all()
        # Steps are 1-based, and step 8 is the virtual E8 channel.
        assert factors["esa_step_i"].min() >= 1
        assert factors["esa_step_k"].max() <= N_ESA + 1
        assert (factors["bootstrap_factor"] > 0).all()

        with patch("imap_processing.lo.l2.lo_l2.ANCILLARY_DATA_DIR", tmp_path):
            with pytest.raises(ValueError, match="No bootstrap correction factor"):
                load_bootstrap_correction_data()


class TestIsnMask:
    """Blanking out the pixels the interstellar neutral flow dominates."""

    def test_the_bright_pixels_of_every_level_are_masked(self, masked_maps):
        """A pixel above the level's threshold comes back undefined."""
        masked, raw = masked_maps

        intensity = raw["ena_intensity"].values
        # The unexposed pixels are filled in both maps, mask or no mask.
        unexposed = is_fill(intensity)
        as_masked = np.where(unexposed, 0.0, intensity)
        peak = np.max(as_masked, axis=(-2, -1), keepdims=True)
        expected = (as_masked >= MASK_THRESHOLD_FRACTION * peak) | unexposed
        assert expected.any(), "the tuning must mask something"
        assert not expected.all(), "the tuning must leave something unmasked"

        np.testing.assert_array_equal(is_fill(masked["ena_intensity"].values), expected)

    def test_the_unmasked_pixels_keep_their_values(self, masked_maps):
        """Masking changes nothing about the pixels it does not blank out."""
        masked, raw = masked_maps

        keep = ~is_fill(masked["ena_intensity"].values)
        np.testing.assert_allclose(
            masked["ena_intensity"].values[keep],
            raw["ena_intensity"].values[keep],
            rtol=1e-6,
        )

    def test_every_species_variable_is_masked_together(self, masked_maps):
        """The counts, rates and uncertainties are blanked with the intensity."""
        masked, raw = masked_maps

        # What the mask blanked, as opposed to what the unexposed pixels of the
        # unmasked map are already filled with.
        expected = is_fill(masked["ena_intensity"].values) & ~is_fill(
            raw["ena_intensity"].values
        )
        assert expected.any(), "the tuning must mask something"
        for name in ISN_MASKED_VARIABLES:
            blanked = is_fill(masked[name].values) & ~is_fill(raw[name].values)
            np.testing.assert_array_equal(blanked, expected, err_msg=name)

    def test_the_exposure_and_background_are_left_alone(self, masked_maps):
        """The mask says nothing about what the map was pointed at."""
        masked, raw = masked_maps

        for name in (
            "exposure_factor",
            "bg_rate",
            "bg_rate_stat_uncert",
            "bg_intensity",
            "bg_intensity_stat_uncert",
        ):
            np.testing.assert_allclose(
                masked[name].values, raw[name].values, rtol=1e-6, err_msg=name
            )

    def test_an_unmasked_map_is_left_whole(self, full_map):
        """A descriptor without the mask code blanks nothing out."""
        dataset, _ = full_map

        # Only the pixels the map was never exposed in are filled, and the raw
        # accumulators keep even those.
        unexposed = dataset["exposure_factor"].values == 0
        for name in ISN_MASKED_VARIABLES:
            filled = name in FILLED_VARIABLES
            expected = unexposed if filled else np.zeros_like(unexposed)
            np.testing.assert_array_equal(
                is_fill(dataset[name].values), expected, err_msg=name
            )

    def test_a_narrow_band_masks_nothing_off_the_ecliptic(
        self, one_pointing, anc_dependencies
    ):
        """A pixel outside the angular width of the band survives its brightness."""
        with patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
            side_effect=identity_pointing,
        ):
            (masked,) = lo_l2(
                as_dependencies(one_pointing),
                anc_dependencies,
                NARROW_MASK_DESCRIPTOR,
            )

        # The mocked sky pointing puts every pixel of the test map on the
        # ecliptic, which the pivot 75 tuning is too narrow to reach. The counts
        # are blanked by the mask alone, never by the exposure.
        assert not is_fill(masked["ena_count"].values).any()

    def test_the_outlier_tail_of_a_level_is_masked(
        self, one_pointing, anc_dependencies
    ):
        """The pixels above the level's percentile are masked wherever they are."""
        with patch(
            "imap_processing.lo.l1c.lo_l1c.frame_transform_az_el",
            side_effect=identity_pointing,
        ):
            (masked,) = lo_l2(
                as_dependencies(one_pointing),
                anc_dependencies,
                OUTLIER_MASK_DESCRIPTOR,
            )
            (raw,) = lo_l2(
                as_dependencies(one_pointing), anc_dependencies, FULL_DESCRIPTOR
            )

        # The pivot 105 tuning cannot mask a band, so only the top half of each
        # level's own intensity distribution is masked. The unexposed pixels are
        # filled either way.
        intensity = raw["ena_intensity"].values
        unexposed = is_fill(intensity)
        as_masked = np.where(unexposed, 0.0, intensity)
        median = np.percentile(as_masked, 50, axis=(-2, -1), keepdims=True)
        expected = (as_masked > median) | unexposed
        assert expected.any(), "the tuning must mask something"

        np.testing.assert_array_equal(is_fill(masked["ena_intensity"].values), expected)

    def test_an_untuned_pivot_angle_is_refused(self, one_pointing, anc_dependencies):
        """A map cannot be masked at a pivot the ancillary says nothing about."""
        with pytest.raises(ValueError, match=r"no mask tuning for the \[60\] degree"):
            lo_l2(
                as_dependencies(one_pointing),
                anc_dependencies,
                UNTUNED_MASK_DESCRIPTOR,
            )

    def test_mask_parameters_cover_every_level_of_every_tuned_pivot(
        self, shipped_ancillaries, tmp_path
    ):
        """The shipped ancillary tunes all 7 levels of each pivot it carries."""
        parameters = load_isn_mask_parameters()

        assert list(parameters.columns) == [
            "pivot_angle",
            "esa_step",
            "intensity_threshold_fraction",
            "angular_width_deg",
            "outlier_percentile",
        ]
        assert not parameters.empty
        for pivot, tuning in parameters.groupby("pivot_angle"):
            assert sorted(tuning["esa_step"]) == list(range(1, N_ESA + 1)), pivot
        assert (parameters["intensity_threshold_fraction"] > 0).all()
        assert (parameters["angular_width_deg"] > 0).all()
        assert parameters["outlier_percentile"].between(0, 100).all()

        with patch("imap_processing.lo.l2.lo_l2.ANCILLARY_DATA_DIR", tmp_path):
            with pytest.raises(ValueError, match="No ISN mask parameter files"):
                load_isn_mask_parameters()


class TestFinalizeDataset:
    """Attaching the CDF attributes to a finished map."""

    def test_attributes_are_filled_in_from_the_descriptor(self):
        """The map is labelled with its descriptor and its variables described."""
        dataset = xr.Dataset(
            {
                "ena_intensity": (
                    ["epoch", "energy"],
                    np.zeros((1, N_ESA)),
                ),
                "not_a_map_variable": ("epoch", np.zeros(1)),
            },
            coords={"epoch": [0], "energy": ESA_ENERGIES[0]},
        )

        finalized = finalize_dataset(dataset, FULL_DESCRIPTOR)

        assert finalized.attrs["Logical_source"] == f"imap_lo_l2_{FULL_DESCRIPTOR}"
        assert FULL_DESCRIPTOR in finalized.attrs["Data_type"]

        # A known map variable picks up its attributes from the enamaps config.
        assert finalized["ena_intensity"].attrs["FIELDNAM"] == "Intensity"
        assert finalized["ena_intensity"].attrs["UNITS"] == "cm -2 s -1 sr -1 keV -1"

        # A variable the config says nothing about is left without attributes,
        # rather than failing the map.
        assert finalized["not_a_map_variable"].attrs == {}


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
