"""
SWE L2 processing module.

This module contains functions to process L1B data to L2 data products.
"""

import numpy as np
import numpy.typing as npt
import xarray as xr

from imap_processing.spice.geometry import get_spacecraft_spin_phase
from imap_processing.swe.utils.swe_utils import read_lookup_table

# TODO: add these to instrument status summary
ENERGY_CONVERSION_FACTOR = 4.75
# 7 CEMs geometric factors in cm^2 sr eV/eV units.
GEOMETRIC_FACTORS = np.array(
    [
        435e-6,
        599e-6,
        808e-6,
        781e-6,
        876e-6,
        548e-6,
        432e-6,
    ]
)
ELECTRON_MASS = 9.10938356e-31  # kg

# See doc string of calculate_phase_space_density() for more details.
VELOCITY_CONVERSION_FACTOR = 1.237e31


def get_particle_energy() -> npt.NDArray:
    """
    Get particle energy.

    Calculate particle energy and add to the lookup table.
    To convert Volts to Energy, multiply ESA voltage in Volts by
    energy conversion factor to get electron energy in eV.

    Returns
    -------
    lookup_table : pandas.DataFrame
        Lookup table with energy column added.
    """
    # The lookup table gives voltage applied to analyzers.
    lookup_table = read_lookup_table()

    # Convert voltage to electron energy in eV by apply conversion factor.
    lookup_table["energy"] = lookup_table["esa_v"].values * ENERGY_CONVERSION_FACTOR
    return lookup_table


def calculate_phase_space_density(l1b_dataset: xr.Dataset) -> xr.Dataset:
    """
    Convert counts to phase space density.

    Calculate phase space density, fv, in units of s^3/ (cm^6 * ster).
        fv = 2 * (C/tau) / (G * v^4)
        where:
            C / tau = corrected count rate. L1B science data.
            G = geometric factor, in (cm^2 * ster). 7 CEMS value.
            v = electron speed, computed from energy, in cm/s.
                We need to use this formula to convert energy to speed:
                    E = 0.5 * m * v^2
                    where E is electron energy, in eV
                    (result from get_particle_energy() function),
                    m is mass of electron (9.10938356e-31 kg),
                    and v is what we want to calculate. Reorganizing above
                    formula result in v = sqrt(2 * E / m). This will be used
                    to calculate electron speed, v.

                Now to convert electron speed units to cm/s:
                    v = sqrt(2 * E / m)
                      = sqrt(2 * E(eV) * 1.60219e-19(J/eV) / 9.10938e-31 kg)
                        where J = kg * m^2 / s^2.
                      = sqrt(2 * 1.60219 * 10e−19 m^2/s^2 * E(eV) / 9.10938e-31)
                      = sqrt(2 * 1.60219 * 10e−19 * 10e4 cm^2/s^2 * E(eV) / 9.10938e-31)
                      = sqrt(3.20438 * 10e-15 * E(eV) / 9.10938e-31) cm/s
                      = sqrt( (3.20438 * 10e-15 / 9.10938e-31) * E(eV) ) cm/s
        fv = 2 * (C/tau) / (G * v^4)
           = 2 * (C/tau) / (G * (sqrt( (3.20438 * 10e-15 / 9.10938e-31) * E(eV) ))^4)
           = 2 * (C/tau) / (G * (sqrt(3.5176e16)^4 * E(eV)^2)
           = 2 * (C/tau) / (G * 1.237e31 * E(eV)^2)
        Ruth Skoug also got the same result, 1.237e31.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        The L1B dataset to process.

    Returns
    -------
    phase_space_density_dataset : xarray.Dataset
        Phase space density. We need to call this phase space density because
        there will be density in L3 processing.
    """
    # Get esa_table_num for each full sweep.
    esa_table_nums = l1b_dataset["esa_table_num"].values[:, 0]
    # Get energy values from lookup table.
    particle_energy = get_particle_energy()
    # Get 720 (24 energy steps x 30 angle) particle energy for each full
    # sweep data.
    particle_energy_data = np.array(
        [
            particle_energy[particle_energy["table_index"] == val]["energy"].tolist()
            for val in esa_table_nums
        ]
    )
    particle_energy_data = particle_energy_data.reshape(-1, 24, 30)

    # Calculate phase space density using formula:
    #   2 * (C/tau) / (G * 1.237e31 * E(eV)^2)
    # See doc string for more details.
    density = (2 * l1b_dataset["science_data"]) / (
        GEOMETRIC_FACTORS[np.newaxis, np.newaxis, np.newaxis, :]
        * VELOCITY_CONVERSION_FACTOR
        * particle_energy_data[:, :, :, np.newaxis] ** 2
    )

    # Return density as xr.dataset with phase space density and
    # energy in eV value that flux calculation can use.
    phase_space_density_dataset = xr.Dataset(
        {
            "phase_space_density": (["epoch", "energy", "angle", "cem"], density.data),
            "energy_in_eV": (["epoch", "energy", "angle"], particle_energy_data),
        },
        coords=l1b_dataset.coords,
    )

    return phase_space_density_dataset


def calculate_flux(l1b_dataset: xr.Dataset) -> npt.NDArray:
    """
    Calculate flux.

    To get flux, j = 2 * m * E * f
        where:
        f = calculate_phase_space_density() result
        m - mass of electron
        E - energy in joules. E(eV) * 1.60219e-19(J/eV)

    To convert flux units:
        j = 2 * m * E * f
          = 2 * 9.10938356e-31 kg * E(eV) * 1.60219e-19(J/eV) * (s^3/ (cm^6 * ster)
          = 2 * 9.10938356e-31 kg * E(eV) * 1.60219e-19 (kg * m^2 / s^2) *
            (s^3 / (cm^6 * ster)
          = 2 * 9.10938356e-31 * E(eV) * 1.60219e-19  kg^2 * ( 10^4 cm^2 / s^2) *
            (s^3 / (cm^6 * ster)
          = 2 * 9.10938356e-31 * E(eV) * 1.60219e-19 * 10^4 ((kg^2 * cm^2 * s^3) /
            (s^2 * cm^6 * ster))
        TODO: ask Ruth Skoug what units to use for flux and work out remaining units.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        The L1B dataset to process.

    Returns
    -------
    flux : numpy.ndarray
        Flux values.
    """
    phase_space_density_ds = calculate_phase_space_density(l1b_dataset)
    # TODO: update this once Ruth sends the correct conversion factors.
    flux = (
        2
        * ELECTRON_MASS
        * phase_space_density_ds["energy_in_eV"].data[:, :, :, np.newaxis]
        * 1.60219e-19
        * 10e4
        * phase_space_density_ds["phase_space_density"].data
    )
    return flux


def swe_l2(l1b_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    Will process data to L2.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        The L1B dataset to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    data : xarray.Dataset
        Processed data to L2.
    """
    flux = calculate_flux(l1b_dataset)

    # Calculate spin phase using SWE sci_step_acq_time_sec calculated in l1b.
    # L1B dataset stores it by (epoch, energy, angle, cem).
    data_acq_time = l1b_dataset["sci_step_acq_time_sec"].data.flatten()

    # calculate spin phase
    spin_phase = get_spacecraft_spin_phase(
        query_met_times=data_acq_time,
    ).reshape(-1, 24, 30, 7)
    # TODO: organize flux data by energy and spin_phase.
    # My understanding from conversation with Ruth is that this is the hardest part
    # and last part of the L2 processing.

    # TODO: Correct return value. This is just a placeholder.
    return xr.Dataset(
        {
            "flux": (["epoch", "energy", "spin_phase", "cem"], flux),
            "spin_phase": (["epoch", "energy", "spin_phase", "cem"], spin_phase),
        },
        coords=l1b_dataset.coords,
    )
