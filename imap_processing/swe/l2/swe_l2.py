"""
SWE L2 processing module.

This module contains functions to process L1B data to L2 data products.
"""

import numpy as np
import numpy.typing as npt
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.spin import get_instrument_spin_phase, get_spin_angle
from imap_processing.swe.utils.swe_utils import (
    ESA_VOLTAGE_ROW_INDEX_DICT,
    read_lookup_table,
)

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
# See doc string of calculate_flux() for more details.
FLUX_CONVERSION_FACTOR = 6.187e30

CEM_DETECTORS_ANGLE = np.array([-63, -42, -21, 0, 21, 42, 63])


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

    Calculate phase space density is represented by this symbol, fv.
    Its unit is s^3/ (cm^6 * ster).

    The formula to calculate phase space density,

    Where:
        C / tau = corrected count rate which in the input L1B science data.
        G = geometric factor, in (cm^2 * ster). 7 CEMs geometric factor value.
        eV = eV in electron-volts, calculated by get_particle_energy().
        E = Energy in Joules. eV * 1.60219e-19(J/eV).
        m = mass of electron (9.10938356e-31 kg).
        s = second.
        v = sqrt(2 * E / m). Electron speed, computed from energy. In cm/s.
        J = kg * m^2 / s^2. J for joules.
        fv = phase space density.

    v   = sqrt(2 * E / m)
        = sqrt(2 * eV * 1.60219e-19(J/eV) / 9.10938e-31 kg)
        = sqrt(2 * 1.60219 * 10e−19 m^2/s^2 * eV / 9.10938e-31)
        = sqrt(2 * 1.60219 * 10e−19 * 10e4 cm^2/s^2 * eV / 9.10938e-31)
        = sqrt(3.20438 * 10e-15 * eV / 9.10938e-31) cm/s
        = sqrt((3.20438 * 10e-15 / 9.10938e-31) * eV) cm/s

    fv  = 2 * (C/tau) / (G * v^4)
        = 2 * (C/tau) / (G * (sqrt( (3.20438 * 10e-15 / 9.10938e-31) * eV ))^4)
        = 2 * (C/tau) / (G * (sqrt(3.5176e16)^4 * eV^2)
        = 2 * (C/tau) / (G * 1.237e31 * eV^2)
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
    #   2 * (C/tau) / (G * 1.237e31 * eV^2)
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
            "phase_space_density": (
                ["epoch", "esa_step", "spin_sector", "cem_id"],
                density.data,
            ),
            "energy_in_eV": (
                ["epoch", "esa_step", "spin_sector"],
                particle_energy_data,
            ),
        },
        coords=l1b_dataset.coords,
    )

    return phase_space_density_dataset


def calculate_flux(l1b_dataset: xr.Dataset) -> npt.NDArray:
    """
    Calculate flux.

    Flux is represented by this symbol, j. Its unit is
    1 / (2 * eV * cm^2 * s * ster).

    The formula to calculate flux,

    Where:
        fv = the phase space density of solar wind electrons
            given by calculate_phase_space_density() result.
        eV = Energy in electron-volts, calculated by get_particle_energy().
        E  = Energy in Joules. eV * 1.60219e-19(J/eV).
        v  = sqrt( (3.20438 * 10e-15 / 9.10938e-31) * eV ) cm/s. See
            calculate_phase_space_density() for this calculation.
        j  = flux factor.

    Flux units workout:
    j   = (fv * v^4) / (2 * eV)
        = ((s^3 / (cm^6 * ster)) * (cm^4/s^4)) / (2 * eV)
        = ((s^3 * cm^4) / (cm^6 * s^4 * ster)) / (2 * eV)
        = (1 / (cm^2 * s * ster)) / (2 * eV)
        = 1 / (2 * eV * cm^2 * s * ster)

    Flux conversion factor workout:
    j   = (fv * v^4) / (2 * eV)
        = ( fv * (sqrt( (3.20438 * 10e-15 / 9.10938e-31) * eV )^4) ) / (2 * eV)
        = ( fv * ((3.20438 * 10e-15 / 9.10938e-31) * eV)^1/2) ^ 4 ) / (2 * eV)
        = ( fv * (3.20438 * 10e-15 / 9.10938e-31)^2 * eV^2) ) / (2 * eV)
        = ( fv * 1.237e31 * eV^2) ) / (2 * eV)
        = ( fv * 1.237e31 * eV ) / 2
        = (fv * 6.187e30 * eV)
        Ruth Skoug confirmed this factor, 6.187e30.

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
    flux = (
        FLUX_CONVERSION_FACTOR
        * phase_space_density_ds["energy_in_eV"].data[:, :, :, np.newaxis]
        * phase_space_density_ds["phase_space_density"].data
    )
    return flux


def put_data_into_angle_bins(
    data: np.ndarray, angle_bin_indices: npt.NDArray[np.int_]
) -> npt.NDArray:
    """
    Put data in its angle bins.

    This function bins SWE data into 30 predefined angle bins
    while preserving the original energy step structure. For each
    full cycle, it assigns data to the corresponding angle bin
    based on the provided indices.

    Since multiple data points may fall into the same angle bin,
    the function accumulates values and computes the average across
    all 7 CEMs, ensuring that each bin contains a representative
    mean value while maintaining the 7 CEM structure.

    Parameters
    ----------
    data : numpy.ndarray
        Data to put in bins. Shape: (full_cycle_data, energy_step, angle_bin, 7).
    angle_bin_indices : numpy.ndarray
        Indices of angle bins to put data in. Shape:
        (full_cycle_data, energy_step, angle_bin).

    Returns
    -------
    numpy.ndarray
        Data in bins. Shape: (full_cycle_data, 24, 30, 7).
    """
    # Initialize with zeros instead of NaN because np.add.at() does not
    # work with nan values. It results in nan + value = nan
    binned_data = np.zeros((data.shape[0], 24, 30, 7), dtype=np.float64)

    time_indices = np.arange(data.shape[0])[:, None, None]
    energy_indices = np.arange(24)[None, :, None]

    # Use np.add.at() to accumulate values into bins
    np.add.at(binned_data, (time_indices, energy_indices, angle_bin_indices), data)

    # Count occurrences in each bin to compute the mean.
    # Ensure float dtype for division
    bin_counts = np.zeros_like(binned_data, dtype=float)
    np.add.at(bin_counts, (time_indices, energy_indices, angle_bin_indices), 1)

    # Compute the mean. Replace zero counts with NaN to indicate no data in the bin
    # because zero physical counts could be valid data.
    bin_counts[bin_counts == 0] = np.nan
    binned_data /= bin_counts

    return binned_data


def find_angle_bin_indices(
    inst_spin_angle: np.ndarray,
) -> npt.NDArray[np.int_]:
    """
    Find angle bin indices.

    The spin angle bins are centered at:
      [ 6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126, 138, 150, 162, 174,
        186, 198, 210, 222, 234, 246, 258, 270, 282, 294, 306, 318, 330,
        342, 354]

    An input angle is assigned to a bin based on the following conditions:
      - phi_begin <= center - 6
      - phi_center = 6
      - phi_end < center + 6

    For example, if the input angle is 8.4, it falls within the bin centered at 6.

    To make binning easier, we define bin edges as:
      [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168,
       180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324,
       336, 348]

    SWE uses the right-side behavior of `np.searchsorted`, where `a[i-1] <= v < a[i]`.

    Example test cases:
      - `np.searchsorted(x, [6], side="right") -> [1]` (Bin center test)
      - `np.searchsorted(x, [8.4], side="right") -> [1]` (Edge case near center)
      - `np.searchsorted(x, [12], side="right") -> [2]` (Bin end test)
      - `np.searchsorted(x, [0], side="right") -> [1]` (Bin start test)

    Using `i-1` ensures that all input angles are assigned to the correct bin of
    centered angle bins.

    Parameters
    ----------
    inst_spin_angle : numpy.ndarray
        Instrument spin angle.

    Returns
    -------
    spin_angle_bins_indices : numpy.ndarray
        Spin angle bin indices.
    """
    spin_angle_bin_edges = np.arange(0, 360, 12)
    # Ensure that inst_spin_angle is np.array for below conditions
    # check to work properly.
    inst_spin_angle = np.array(inst_spin_angle)
    # Check that there are no angle values outside the range [0, 360).
    if np.any((inst_spin_angle < 0) | (inst_spin_angle >= 360)):
        raise ValueError("Input angle values must be in the range [0, 360)")

    spin_angle_bins_indices = np.searchsorted(
        spin_angle_bin_edges, inst_spin_angle, side="right"
    )
    spin_angle_bins_indices = spin_angle_bins_indices - 1
    return spin_angle_bins_indices


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
    cdf_attributes = ImapCdfAttributes()
    cdf_attributes.add_instrument_global_attrs("swe")
    cdf_attributes.add_instrument_variable_attrs("swe", "l2")
    cdf_attributes.add_global_attribute("Data_version", data_version)

    # Energy values in eV.
    energy_xr = xr.DataArray(
        np.array(list(ESA_VOLTAGE_ROW_INDEX_DICT.keys())) * ENERGY_CONVERSION_FACTOR,
        name="energy",
        dims=["energy"],
        attrs=cdf_attributes.get_variable_attributes("energy"),
    )

    energy_label = xr.DataArray(
        np.array(list(ESA_VOLTAGE_ROW_INDEX_DICT.keys())).astype(str),
        name="energy_label",
        dims=["energy"],
        attrs=cdf_attributes.get_variable_attributes("energy_label"),
    )

    # Angle of each CEM detectors.
    inst_el_xr = xr.DataArray(
        CEM_DETECTORS_ANGLE,
        name="inst_el",
        dims=["inst_el"],
        attrs=cdf_attributes.get_variable_attributes("inst_el"),
    )
    inst_el_label = xr.DataArray(
        CEM_DETECTORS_ANGLE.astype(str),
        name="inst_el_label",
        dims=["inst_el"],
        attrs=cdf_attributes.get_variable_attributes("inst_el_label"),
    )

    # Spin Angle bins storing bin center values.
    inst_az_xr = xr.DataArray(
        np.arange(0, 360, 12) + 6,
        name="inst_az",
        dims=["inst_az"],
        attrs=cdf_attributes.get_variable_attributes("inst_az"),
    )
    inst_az_label = xr.DataArray(
        inst_az_xr.values.astype(str),
        name="inst_az_label",
        dims=["inst_az"],
        attrs=cdf_attributes.get_variable_attributes("inst_az_label"),
    )

    dataset = xr.Dataset(
        coords={
            "epoch": l1b_dataset["epoch"],
            "esa_step": l1b_dataset["esa_step"],
            "energy": energy_xr,
            "spin_sector": l1b_dataset["spin_sector"],
            "inst_az": inst_az_xr,
            "cem_id": l1b_dataset["cem_id"],
            "inst_el": inst_el_xr,
            "esa_step_label": l1b_dataset["esa_step_label"],
            "energy_label": energy_label,
            "spin_sector_label": l1b_dataset["spin_sector_label"],
            "inst_az_label": inst_az_label,
            "cem_id_label": l1b_dataset["cem_id_label"],
            "inst_el_label": inst_el_label,
        },
        attrs=cdf_attributes.get_global_attributes("imap_swe_l2_sci"),
    )

    ############################################################
    # Calculate phase space density and flux. Store data in shape
    # (epoch, esa_step, spin_sector, cem_id). This is for L3 purposes.
    ############################################################
    phase_space_density = calculate_phase_space_density(l1b_dataset)[
        "phase_space_density"
    ]
    dataset["phase_space_density_spin_sector"] = xr.DataArray(
        phase_space_density,
        name="phase_space_density_spin_sector",
        dims=["epoch", "esa_step", "spin_sector", "cem_id"],
        attrs=cdf_attributes.get_variable_attributes("phase_space_density_spin_sector"),
    )

    flux = calculate_flux(l1b_dataset)
    dataset["flux_spin_sector"] = xr.DataArray(
        flux,
        name="flux_spin_sector",
        dims=["epoch", "esa_step", "spin_sector", "cem_id"],
        attrs=cdf_attributes.get_variable_attributes("flux_spin_sector"),
    )

    # Carry over acquisition times for L3 purposes.
    dataset["acquisition_time"] = l1b_dataset["acquisition_time"]

    # Calculate spin phase using SWE acquisition_time from the
    # L1B dataset. The L1B dataset stores acquisition_time with
    # dimensions (epoch, esa_step, spin_sector). Use center time
    # to calculate spin phase. This center time calculation is
    # necessary to accurately determine the center angle of the data.
    #
    # To determine the center acquisition time, we adjust the
    # recorded acquisition_time as follows:
    #   acquisition_time + (acq_duration / 1000000) / 2
    #
    # Here, acq_duration is given in microseconds and is stored
    # in the L1B dataset with dimensions (epoch, cycle). Since acq_duration
    # remains the same for all quarter cycles within a full sweep,
    # we use the first acq_duration value for each full sweep to perform
    # this adjustment.

    acq_duration = l1b_dataset["acq_duration"].data[:, 0] / 2000000
    data_acq_time = (
        l1b_dataset["acquisition_time"].data + acq_duration[:, np.newaxis, np.newaxis]
    )

    # Calculate spin phase
    inst_spin_phase = get_instrument_spin_phase(
        query_met_times=data_acq_time.ravel(),
        instrument=SpiceFrame.IMAP_SWE,
    )

    # Convert spin phase to spin angle in degrees.
    inst_spin_angle = get_spin_angle(inst_spin_phase, degrees=True).reshape(-1, 24, 30)

    # Save spin angle in dataset per SWE request.
    dataset["inst_az_spin_sector"] = xr.DataArray(
        inst_spin_angle,
        name="inst_az_spin_sector",
        dims=["epoch", "energy", "inst_az"],
        attrs=cdf_attributes.get_variable_attributes("inst_az_spin_sector"),
    )

    spin_angle_bins_indices = find_angle_bin_indices(inst_spin_angle)

    # Put flux data in its spin angle bins using the indices.
    flux_binned_data = put_data_into_angle_bins(flux, spin_angle_bins_indices)
    dataset["flux"] = xr.DataArray(
        flux_binned_data,
        name="flux",
        dims=["epoch", "energy", "inst_az", "inst_el"],
        attrs=cdf_attributes.get_variable_attributes("flux"),
    )

    # Put phase space density data in its spin angle bins using the indices.
    phase_space_density_binned_data = put_data_into_angle_bins(
        phase_space_density.data, spin_angle_bins_indices
    )
    dataset["phase_space_density"] = xr.DataArray(
        phase_space_density_binned_data,
        name="phase_space_density",
        dims=["epoch", "energy", "inst_az", "inst_el"],
        attrs=cdf_attributes.get_variable_attributes("phase_space_density"),
    )

    return dataset
