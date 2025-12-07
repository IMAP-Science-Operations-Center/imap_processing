"""Make heliocentric HEALPix index maps for Ultra L1C processing."""

import logging

import healpy as hp
import numpy as np
import xarray as xr

from imap_processing.spice.geometry import (
    SpiceBody,
    SpiceFrame,
    get_rotation_matrix,
    imap_state,
)
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins

logger = logging.getLogger(__name__)

# TODO why this constant offset??? Ask Nick for his CoordConverters.convertToRaDec
#  function
ULTRA_90_PHI_CORRECTION_DEG = 139.609025


def is_inside_fov(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Determine if angles are inside the field of view (FOV).

    Parameters
    ----------
    theta : np.ndarray
        Theta angles in radians.
    phi : np.ndarray
        Phi angles in radians.

    Returns
    -------
    np.ndarray
        Boolean array indicating if angles are in FOV.
    """
    numer_term = 5.0
    denom_term = 2.80
    theta_offset_deg = 0.0
    phi_limit_deg = 60.0

    cos_phi = np.cos(phi)

    theta_boundary = np.arctan(
        numer_term * cos_phi / (1.0 + denom_term * cos_phi)
    ) - np.radians(theta_offset_deg)

    theta_check = np.abs(theta) < np.abs(theta_boundary)
    phi_check = np.abs(phi) < np.radians(phi_limit_deg)

    return theta_check & phi_check


def vector_ijk_to_theta_phi(
    inst_vecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert instrument vectors to theta/phi.

    This matches Java's vectorIJK2ThetaPhi implementation.

    Parameters
    ----------
    inst_vecs : np.ndarray
        Array of shape (n, 3) with components (x, y, z).

    Returns
    -------
    theta : np.ndarray
        Declination in radians, range [-π, π].
    phi : np.ndarray
        Right ascension in radians, range [-π, π].
    """
    # Extract components
    i_comp = inst_vecs[:, 0]  # x component
    j_comp = inst_vecs[:, 1]  # y component
    k_comp = inst_vecs[:, 2]  # z component

    # Normalize
    magnitude = np.linalg.norm(inst_vecs, axis=1)

    # Compute declination and right ascension
    theta = np.arcsin(i_comp / magnitude)
    phi = np.arctan2(j_comp, k_comp)

    # Wrap to [-π, π]
    theta = np.where(theta > np.pi, theta - 2 * np.pi, theta)
    phi = np.where(phi > np.pi, phi - 2 * np.pi, phi)

    return theta, phi


def make_helio_index_maps(
    nside: int,
    spin_duration: float,
    num_steps: int,
    start_et: float,
    instrument_frame: SpiceFrame = SpiceFrame.IMAP_ULTRA_90,
    compute_bsf: bool = False,
    boundary_points: int = 8,
) -> xr.Dataset:
    """
    Create HEALPix index maps for heliocentric observations.

    This function generates exposure maps that account for spacecraft
    velocity aberration, multiple energy bins, and spin phase sampling.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter (determines angular resolution).
    spin_duration : float
        Total spin period in seconds.
    num_steps : int
        Number of spin phase steps to sample.
    start_et : float
        Start ephemeris time.
    instrument_frame : SpiceFrame, optional
        SpiceFrame of the instrument (default IMAP_ULTRA_90).
    compute_bsf : bool, optional
        If True, compute boundary scale factors (default False).
    boundary_points : int, optional
        Number of boundary points to sample per pixel (default 8).

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions (step, energy, pixel) containing index,
        theta, phi, and bsf data variables, plus ra and dec coordinates.
    """
    # Get spacecraft velocity at START time
    state = imap_state(start_et, ref_frame=SpiceFrame.IMAP_DPS, observer=SpiceBody.SUN)
    sc_vel = state[3:6]  # Extract [vx, vy, vz]

    logger.info("Spacecraft velocity: %s km/s", sc_vel)
    logger.info("Speed: %.2f km/s", np.linalg.norm(sc_vel))

    # Build energy bins
    _, energy_midpoints, energy_bin_geometric_means = build_energy_bins()
    num_energy_bins = len(energy_bin_geometric_means)

    # Get number of pixels
    npix = hp.nside2npix(nside)

    # Compute RA/Dec for pixel centers
    pixel_indices = np.arange(npix)

    # Time parameters
    end_et = start_et + spin_duration
    dt_step = spin_duration / num_steps

    # Pre-compute all pixel vectors once
    pixel_vecs = np.array(hp.pix2vec(nside, pixel_indices, nest=False)).T  # (npix, 3)

    # Initialize output arrays
    index_map = np.zeros((num_steps, num_energy_bins, npix))
    theta_map = np.zeros((num_steps, num_energy_bins, npix))
    phi_map = np.zeros((num_steps, num_energy_bins, npix))
    bsf_map = np.zeros((num_steps, num_energy_bins, npix))

    logger.info(
        "Processing %d time steps, %d energy bins, %d pixels...",
        num_steps,
        num_energy_bins,
        npix,
    )
    if compute_bsf:
        logger.info(
            "Computing boundary scale factors with %d points per pixel",
            boundary_points,
        )

    # OUTER LOOP: time steps
    time_id = 0
    t = start_et
    while t < (end_et - dt_step / 2):
        # Get rotation matrix for this time step
        rotation_matrix = get_rotation_matrix(
            np.array([t]),
            from_frame=instrument_frame,
            to_frame=SpiceFrame.IMAP_DPS,
        )[0]

        # MIDDLE LOOP: energy bins
        for energy_id in range(num_energy_bins):
            # Convert energy to velocity (km/s)
            energy_mean = energy_bin_geometric_means[energy_id]
            kps = (
                np.sqrt(2 * energy_mean * UltraConstants.KEV_J / UltraConstants.MASS_H)
                / 1e3
            )

            # Transform pixel vectors to heliocentric frame
            helio_velocity = (
                sc_vel.reshape(1, 3) + kps * pixel_vecs
            )  # Galilean transform
            helio_normalized = helio_velocity / np.linalg.norm(
                helio_velocity, axis=1, keepdims=True
            )

            # Transform to instrument frame
            inst_vecs = helio_normalized @ rotation_matrix
            inst_vecs = inst_vecs.astype(np.float32)

            theta, phi = vector_ijk_to_theta_phi(inst_vecs)
            theta = theta.astype(np.float32)
            phi = phi.astype(np.float32)

            # Apply phi correction
            phi_correction = np.radians(ULTRA_90_PHI_CORRECTION_DEG)
            phi = phi + phi_correction
            phi = np.where(phi > np.pi, phi - 2 * np.pi, phi)

            # Check FOV
            in_fov_mask = is_inside_fov(theta, phi)
            fov_pixels = np.where(in_fov_mask)[0]

            # Store results for FOV pixels
            theta_map[time_id, energy_id, fov_pixels] = np.degrees(theta[fov_pixels])
            phi_map[time_id, energy_id, fov_pixels] = np.degrees(phi[fov_pixels])
            index_map[time_id, energy_id, fov_pixels] = 1.0

            # Compute boundary scale factor if requested
            if compute_bsf:
                for pix_id in fov_pixels:
                    # Get boundary vectors for this pixel
                    boundary_step = boundary_points // 4
                    boundary_vecs = hp.boundaries(
                        nside, pix_id, step=boundary_step, nest=False
                    )
                    boundary_vecs = boundary_vecs.T

                    # Include center pixel
                    sample_vecs = np.vstack(
                        [boundary_vecs, pixel_vecs[pix_id : pix_id + 1]]
                    )

                    # Transform boundary vectors to heliocentric frame
                    helio_boundary_vel = sc_vel.reshape(1, 3) + kps * sample_vecs
                    helio_boundary_norm = helio_boundary_vel / np.linalg.norm(
                        helio_boundary_vel, axis=1, keepdims=True
                    )

                    # Transform to instrument frame
                    inst_boundary = helio_boundary_norm @ rotation_matrix

                    # Convert to theta/phi
                    theta_b, phi_b = vector_ijk_to_theta_phi(inst_boundary)
                    phi_b = phi_b + phi_correction
                    phi_b = np.where(phi_b > np.pi, phi_b - 2 * np.pi, phi_b)

                    # Check how many sample points are in FOV
                    in_fov_boundary = is_inside_fov(theta_b, phi_b)
                    bsf = np.sum(in_fov_boundary) / len(sample_vecs)

                    bsf_map[time_id, energy_id, pix_id] = bsf

        # Increment time
        time_id += 1
        t += dt_step

    # Create coordinate arrays
    step_indices = np.arange(num_steps)
    spin_phases = np.linspace(0, 360, num_steps, endpoint=False)

    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "index": (
                ["step", "energy", "pixel"],
                index_map,
                {"long_name": "Pixel in FOV flag"},
            ),
            "theta": (
                ["step", "energy", "pixel"],
                theta_map,
                {"long_name": "Instrument theta angle", "units": "degrees"},
            ),
            "phi": (
                ["step", "energy", "pixel"],
                phi_map,
                {"long_name": "Instrument phi angle", "units": "degrees"},
            ),
            "bsf": (
                ["step", "energy", "pixel"],
                bsf_map,
                {"long_name": "Boundary scale factor", "units": "fractional"},
            ),
        },
        coords={
            "step": (["step"], step_indices),
            "energy": (
                ["energy"],
                energy_bin_geometric_means,
                {"long_name": "Energy bin geometric mean", "units": "keV"},
            ),
            "pixel": (["pixel"], pixel_indices),
            "spin_phase": (
                ["step"],
                spin_phases,
                {"long_name": "Spin phase", "units": "degrees"},
            ),
            "energy_midpoint": (
                ["energy"],
                energy_midpoints,
                {"long_name": "Energy bin midpoint", "units": "keV"},
            ),
        },
    )

    return ds
