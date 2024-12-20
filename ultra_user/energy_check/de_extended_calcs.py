import numpy as np
from numpy.typing import NDArray
import logging
import xarray
import imap_processing.ultra.l1b.ultra_l1b_extended as l1b_ext
import imap_processing.ultra.constants as constants
from numpy import ndarray
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# pull 1a stuff
def get_xf(l1a_dset: xarray.Dataset) -> NDArray:
    return l1b_ext.get_front_x_position(l1a_dset["START_TYPE"].data, l1a_dset["START_POS_TDC"].data)


def get_1bdict(l1a_dset_uf: xarray.Dataset, sensor="ultra45") -> dict:
    result = dict()
    indices = np.nonzero(
        np.isin(l1a_dset_uf["STOP_TYPE"], [l1b_ext.StopType.Top.value,
                                           l1b_ext.StopType.Bottom.value])
    )[0]
    l1a_dset = l1a_dset_uf.isel(epoch=indices)

    xf = get_xf(l1a_dset)
    result["xf"] = xf
    tof, t2, xb, yb = l1b_ext.get_ph_tof_and_back_positions(l1a_dset, xf, sensor)
    result["tof"] = tof
    result["t2"] = t2
    result["xb"] = xb
    result["yb"] = yb
    d, yf = l1b_ext.get_front_y_position(l1a_dset["START_TYPE"].data, yb)
    result["d"] = d
    result["yf"] = yf
    r = l1b_ext.get_path_length((xf, yf), (xb, yb), d)
    result["r"] = r
    ctof = l1b_ext.get_ctof(tof, r,"PH")
    result["ctof"] = ctof
    dmin = ndarray(ctof.size)
    dmin[:] = constants.UltraConstants.DMIN_PH_CTOF
    species = species_from_ctof(ctof)
    result["species"] = species
    v = de_velocity((xf, yf), (xb, yb), d, tof)
    result["v"] = np.asarray(v)
    v_mag = velocity_magnitude(ctof, dmin)
    result["v_mag"] = v_mag
    energy = de_energy_kev(v, species)
    result["energy"] = energy
    ih = np.where(species == "H")
    result["ih"] = ih
    ssd_indices = np.where(l1a_dset["STOP_TYPE"].data >= 8)[0]
    ph_indices = np.where(l1a_dset["STOP_TYPE"].data < 8)[0]
    result["issd"] = ssd_indices
    result["ph"] = ph_indices
    theta,phi = event_az_el(v)
    result["event_theta"]=theta
    result["event_phi"]=phi

    return result


# new stuff
def de_velocity(
        front_position: tuple[NDArray, NDArray],
        back_position: tuple[NDArray, NDArray],
        d: np.ndarray,
        tof: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tof[tof < 0].any():
        logger.info("Negative tof values found.")

    # distances in .1 mm
    delta_x = .1 * ((front_position[0] - back_position[0]))
    delta_y = .1 * ((front_position[1] - back_position[1]))
    delta_z = .1 * d

    v_x = mmperns_kmpers * delta_x / tof
    v_y = mmperns_kmpers * delta_y / tof
    v_z = mmperns_kmpers * delta_z / tof

    return v_x, v_y, v_z


def velocity_magnitude(ctof: ndarray, dmin: ndarray) -> np.ndarray:
    cctof = ctof
    cctof[ctof < 1] = 1
    # ctof ihas units of 0.1ns
    return 10 * mmperns_kmpers * dmin / cctof


mH_kg = 1.6735575e-27
Joule_to_keV = 6.242e+15
mmperns_kmpers = 1000


def de_energy_kev(v: tuple[NDArray, NDArray, NDArray],
                  species: NDArray) -> NDArray:
    vv = np.asarray(v) * 1.e3  # convert km/s to m/s
    v2 = np.sum((vv * vv), 0)

    iH = np.where(species == "H")
    result = np.full_like(v2, np.nan)

    result[iH] = 0.5 * mH_kg * v2[iH] * Joule_to_keV
    return result


def species_from_ctof(ctof: ndarray) -> ndarray:
    result = np.ndarray(ctof.size, dtype=object)
    result[:] = "unknown"
    result[np.where(np.logical_and(ctof > 50, ctof < 200))] = "H"
    return result


def event_az_el(v: tuple[NDArray, NDArray, NDArray]) -> tuple[NDArray, NDArray]:
    vv = np.asarray(v)
    vmag = np.sqrt(np.sum((vv * vv), 0))

    ux = vv[0, :] / vmag
    uy = vv[1, :] / vmag
    uz = vv[2, :] / vmag

    theta = np.arccos(uz / vmag)
    phi = np.arctan2(uy, ux)

    return theta, phi
