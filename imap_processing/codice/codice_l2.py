"""
Perform CoDICE l2 processing.

This module processes CoDICE l1 files and creates L2 data products.

Notes
-----
from imap_processing.codice.codice_l2 import process_codice_l2
dataset = process_codice_l2(l1_filename)
"""

import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from imap_data_access import ProcessingInputCollection, ScienceFilePath
from numpy.typing import NDArray

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice.constants import (
    GAIN_ID_TO_STR,
    HALF_SPIN_FILLVAL,
    HI_L2_ELEVATION_ANGLE,
    HI_OMNI_VARIABLE_NAMES,
    HI_SECTORED_VARIABLE_NAMES,
    L2_HI_SECTORED_ANGLE,
    LO_NSW_ANGULAR_VARIABLE_NAMES,
    LO_NSW_SPECIES_VARIABLE_NAMES,
    LO_POSITION_TO_ELEVATION_ANGLE,
    LO_SW_ANGULAR_VARIABLE_NAMES,
    LO_SW_PICKUP_ION_SPECIES_VARIABLE_NAMES,
    LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES,
    NSW_POSITIONS,
    PUI_POSITIONS,
    SOLAR_WIND_POSITIONS,
    SSD_ID_TO_ELEVATION,
    SSD_ID_TO_SPIN_ANGLE,
    SW_POSITIONS,
)
from imap_processing.codice.utils import apply_replacements_to_attrs

logger = logging.getLogger(__name__)


def get_lo_de_energy_luts(
    dependencies: ProcessingInputCollection,
) -> tuple[NDArray, NDArray]:
    """
    Get the LO DE lookup tables for energy conversions.

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        The collection of processing input files.

    Returns
    -------
    energy_lut : np.ndarray
        An array of energy in keV for each energy table index.
    energy_bins_lut : np.ndarray
        An array of energy bins.
    """
    # Get lookup tables
    energy_table_file = dependencies.get_file_paths(
        descriptor="l2-lo-onboard-energy-table"
    )[0]
    energy_bins_file = dependencies.get_file_paths(
        descriptor="l2-lo-onboard-energy-bins"
    )[0]
    energy_lut = pd.read_csv(energy_table_file, header=None, skiprows=1).to_numpy()
    energy_bins_lut = pd.read_csv(energy_bins_file, header=None, skiprows=1).to_numpy()[
        :, 1
    ]

    return energy_lut, energy_bins_lut


def get_mpq_calc_energy_conversion_vals(
    dependencies: ProcessingInputCollection,
) -> np.ndarray:
    """
    Get the mass per charge (MPQ) esa step to energy kev conversion lookup table values.

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        The collection of processing input files.

    Returns
    -------
    esa_kev : np.ndarray
        An array of energy in keV for each esa step.
    """
    mpq_calc_lut_file = dependencies.get_file_paths(descriptor="l2-lo-onboard-mpq-cal")[
        0
    ]
    mpq_df = pd.read_csv(mpq_calc_lut_file, header=None)
    k_factor = float(mpq_df.loc[0, 10])
    esa_v = mpq_df.loc[4, 4:].to_numpy().astype(np.float64)
    # Calculate the energy in keV for each esa step
    esa_kev = esa_v * k_factor / 1000
    return esa_kev


def get_mpq_calc_tof_conversion_vals(
    dependencies: ProcessingInputCollection,
) -> np.ndarray:
    """
    Get the MPQ calculation tof to ns conversion lookup table values.

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        The collection of processing input files.

    Returns
    -------
    tof_ns : np.ndarray
        Tof in ns for each TOF bit.
    """
    mpq_calc_lut_file = dependencies.get_file_paths(descriptor="l2-lo-onboard-mpq-cal")[
        0
    ]
    mpq_df = pd.read_csv(mpq_calc_lut_file, header=None)
    ns_channel_sq = float(mpq_df.loc[2, 1])
    ns_channel = float(mpq_df.loc[3, 1])
    tof_offset = float(mpq_df.loc[4, 1])
    # Get the TOF bit to ns lookup
    tof_bits = mpq_df.loc[6:, 0].to_numpy().astype(np.int64)
    # Calculate the TOF in ns for each TOF bit
    tof_ns = tof_bits**2 * ns_channel_sq + tof_bits * ns_channel + tof_offset

    return tof_ns


def get_hi_de_luts(
    dependencies: ProcessingInputCollection | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load lookup tables for hi direct-event processing.

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        The collection of processing input files.

    Returns
    -------
    energy_table : np.ndarray
        2D array of energy lookup table with shape (ssd_energy, col).
    tof_table : np.ndarray
        2D array of tof lookup table with shape (tof_index, col).
    """
    energy_table_file_path = dependencies.get_file_paths(
        descriptor="l2-hi-energy-table"
    )[0]
    tof_table_file_path = dependencies.get_file_paths(descriptor="l2-hi-tof-table")[0]
    # Read TOF CSV, skip first column which is an index
    # Each row corresponds to a tof index and the columns are tof (ns) and E/n (MeV/n)
    tof_table = (
        pd.read_csv(tof_table_file_path, header=None, skiprows=1).iloc[:, 1:].to_numpy()
    )
    # Read energy table CSV, skip first column which is an index
    # Each row corresponds to an ssd energy index and the columns map to a combination
    # of gain and ssd id
    energy_table = (
        pd.read_csv(energy_table_file_path, header=None, skiprows=1)
        .iloc[:, 1:]
        .to_numpy()
    )
    return energy_table, tof_table


def get_geometric_factor_lut(
    dependencies: ProcessingInputCollection | None,
    path: Path | None = None,
) -> dict:
    """
    Get the geometric factor lookup table.

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        The collection of processing input files.
    path : pathlib.Path
        Optional path used for I-ALiRT.

    Returns
    -------
    geometric_factor_lut : dict
        A dict with a full and reduced mode array with shape (esa_steps, position).
    """
    if path is not None:
        csv_path = path
    else:
        csv_path = Path(dependencies.get_file_paths(descriptor="l2-lo-gfactor")[0])

    geometric_factors = pd.read_csv(csv_path)
    # sort by esa step. They should already be sorted, but just in case
    full = geometric_factors[geometric_factors["mode"] == "full"].sort_values(
        by="esa_step"
    )
    reduced = geometric_factors[geometric_factors["mode"] == "reduced"].sort_values(
        by="esa_step"
    )

    # Sort position columns to ensure the correct order
    position_names_sorted = sorted(
        [col for col in full if col.startswith("position")],
        key=lambda x: int(x.split("_")[-1]),
    )

    return {
        "full": full[position_names_sorted].to_numpy(),
        "reduced": reduced[position_names_sorted].to_numpy(),
    }


def get_efficiency_lut(
    dependencies: ProcessingInputCollection | None,
    path: Path | None = None,
) -> pd.DataFrame:
    """
    Get the efficiency lookup table.

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        The collection of processing input files.
    path : pathlib.Path
        Optional path used for I-ALiRT.

    Returns
    -------
    efficiency_lut : pandas.DataFrame
        Contains the efficiency lookup table. Columns are:
        species, product, esa_step, position_1, position_2, ..., position_24.
    """
    if path is not None:
        csv_path = path
    else:
        csv_path = Path(dependencies.get_file_paths(descriptor="l2-lo-efficiency")[0])
    return pd.read_csv(csv_path)


def get_species_efficiency(species: str, efficiency: pd.DataFrame) -> xr.DataArray:
    """
    Get the efficiency values for a given species.

    Parameters
    ----------
    species : str
        The species name.
    efficiency : pandas.DataFrame
        The efficiency lookup table.

    Returns
    -------
    efficiency : xarray.DataArray
        A 2D array of efficiencies with shape (epoch, esa_steps).
    """
    species_efficiency = efficiency[efficiency["species"] == species].sort_values(
        by="esa_step"
    )
    # Sort position columns to ensure the correct order
    position_names_sorted = sorted(
        [col for col in species_efficiency if col.startswith("position")],
        key=lambda x: int(x.split("_")[-1]),
    )
    # Shape: (esa_step, inst_az)
    return xr.DataArray(
        species_efficiency[position_names_sorted].to_numpy(),
        dims=("esa_step", "inst_az"),
    )


def compute_geometric_factors(
    dataset: xr.Dataset, geometric_factor_lookup: dict
) -> xr.DataArray:
    """
    Calculate geometric factors needed for intensity calculations.

    Geometric factors are determined by comparing the half-spin values per
    esa_step in the HALF_SPIN_LUT to the rgfo_half_spin values in the provided
    L2 dataset.

    If the half-spin value is less than the corresponding rgfo_half_spin value,
    the geometric factor is set to 0.75 (full mode); otherwise, it is set to 0.5
    (reduced mode).

    NOTE: Half spin values are associated with ESA steps which corresponds to the
    index of the energy_per_charge dimension that is between 0 and 127.

    Parameters
    ----------
    dataset : xarray.Dataset
        The L2 dataset containing rgfo_half_spin data variable.
    geometric_factor_lookup : dict
        A dict with a full and reduced mode array with shape (esa_steps, position).

    Returns
    -------
    geometric_factors : xarray.DataArray
        A 3D array of geometric factors with shape (epoch, esa_steps, positions).
    """
    # Get half spin values per esa step from the dataset
    half_spin_per_esa_step = dataset.half_spin_per_esa_step.values

    # Expand dimensions to compare each rgfo_half_spin value against
    # all half_spin_values
    rgfo_half_spin = dataset.rgfo_half_spin.data[:, np.newaxis]  # Shape: (epoch, 1)
    # Perform the comparison and calculate modes
    # Modes will be true (reduced mode) anywhere half_spin > rgfo_half_spin otherwise
    # false (full mode)
    # TODO: The mode calculation will need to be revisited after FW changes in january
    #  2026. We also need to fix this on days when the sci Lut changes.
    # After November 24th 2025 we need to do this step a different way.
    start_date = dataset.attrs.get("Logical_file_id", None)
    if start_date is None:
        raise ValueError("Dataset is missing Logical_file_id attribute.")
    processing_date = datetime.datetime.strptime(start_date.split("_")[4], "%Y%m%d")
    date_switch = datetime.datetime(2025, 11, 24)
    # Only consider valid half spins
    valid_half_spin = half_spin_per_esa_step != HALF_SPIN_FILLVAL
    if processing_date < date_switch:
        modes = (
            valid_half_spin
            & (half_spin_per_esa_step > rgfo_half_spin)
            & (rgfo_half_spin > 0)
        )
    else:
        # After November 24th, 2025, we no longer apply reduced geometric factors;
        # always use the full geometric factor lookup.
        modes = np.zeros_like(half_spin_per_esa_step, dtype=bool)

    # Get the geometric factors based on the modes
    gf = np.where(
        modes[:, :, np.newaxis],  # Shape (epoch, esa_step, 1)
        geometric_factor_lookup["reduced"],  # Shape (1, esa_step, 24) - reduced mode
        geometric_factor_lookup["full"],  # Shape (1, esa_step, 24) - full mode
    )  # Shape: (epoch, esa_step, inst_az)

    return xr.DataArray(gf, dims=("epoch", "esa_step", "inst_az"))


def calculate_intensity(
    dataset: xr.Dataset,
    species_list: list,
    geometric_factors: xr.DataArray,
    efficiency: pd.DataFrame,
    positions: list,
    average_across_positions: bool = False,
) -> xr.Dataset:
    """
    Calculate species or angular intensities.

    Parameters
    ----------
    dataset : xarray.Dataset
        The L2 dataset to process.
    species_list : list
        List of species variable names to calculate intensity.
    geometric_factors : np.ndarray
        The geometric factors array with shape (epoch, esa_steps).
    efficiency : pandas.DataFrame
        The efficiency lookup table.
    positions : list
        A list of position indices to select from the geometric factor and
        efficiency lookup tables.
    average_across_positions : bool
        Whether to average the efficiencies and geometric factors across the selected
        positions. Default is False.

    Returns
    -------
    xarray.Dataset
        The updated L2 dataset with species intensities calculated.
    """
    # Select the relevant positions from the geometric factors
    # TODO revisit gfactor calculation. For pickup ions, only position 0 is used
    #   Eventually, the CoDICE team wants to standardize this.
    if species_list == LO_SW_PICKUP_ION_SPECIES_VARIABLE_NAMES:
        geometric_factors = geometric_factors.isel(inst_az=[0])
    else:
        geometric_factors = geometric_factors.isel(inst_az=positions)
    if average_across_positions:
        # take the mean geometric factor across positions
        geometric_factors = geometric_factors.mean(dim="inst_az")
        scalar = len(positions)
    else:
        scalar = 1
    # Calculate the angular intensities using the provided geometric factors and
    # efficiency.
    # intensity = species_rate / (gm * eff * esa_step) for position and spin angle
    for species in species_list:
        # Select the relevant positions for the species from the efficiency LUT
        # Shape: (epoch, esa_step, inst_az)
        species_eff = get_species_efficiency(species, efficiency).isel(
            inst_az=positions
        )
        if species_eff.size == 0:
            logger.warning(f"No efficiency data found for species {species}. Skipping.")
            continue

        if average_across_positions:
            # Take the mean efficiency across positions
            species_eff = species_eff.mean(dim="inst_az")

        # Shape: (epoch, esa_step, inst_az) or
        # (epoch, esa_step) if averaged
        denominator = scalar * geometric_factors * species_eff * dataset["energy_table"]
        if species not in dataset:
            logger.warning(
                f"Species {species} not found in dataset. Filling with NaNS."
            )
            dataset[species] = np.full(dataset["esa_step"].data.shape, np.nan)
        else:
            # Only replace the data with calculated intensity to keep the attributes
            dataset[species].data = (dataset[species] / denominator).data

        # Also calculate uncertainty if available
        species_uncertainty = f"unc_{species}"
        if species_uncertainty not in dataset:
            logger.warning(
                f"Uncertainty {species_uncertainty} not found in dataset."
                f" Filling with NaNS."
            )
            dataset[species_uncertainty] = np.full(
                dataset["esa_step"].data.shape, np.nan
            )
        else:
            dataset[species_uncertainty].data = (
                dataset[species_uncertainty] / denominator
            ).data

    return dataset


def process_lo_species_intensity(
    dataset: xr.Dataset,
    species_list: list,
    geometric_factors: xr.DataArray,
    efficiency: pd.DataFrame,
    positions: list,
) -> xr.Dataset:
    """
    Process the lo-species L2 dataset to calculate species intensities.

    Parameters
    ----------
    dataset : xarray.Dataset
        The L2 dataset to process.
    species_list : list
        List of species variable names to calculate intensity.
    geometric_factors : xarray.DataArray
        The geometric factors array with shape (epoch, esa_steps).
    efficiency : pandas.DataFrame
        The efficiency lookup table.
    positions : list
        A list of position indices to select from the geometric factor and
        efficiency lookup tables.

    Returns
    -------
    xarray.Dataset
        The updated L2 dataset with species intensities calculated.
    """
    # Calculate the species intensities using the provided geometric factors and
    # efficiency.
    dataset = calculate_intensity(
        dataset,
        species_list,
        geometric_factors,
        efficiency,
        positions,
        average_across_positions=True,
    )
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_variable_attrs("codice", "l2-lo-species")
    if positions == SOLAR_WIND_POSITIONS:
        species_attrs = cdf_attrs.get_variable_attributes("lo-sw-species-attrs")
        unc_attrs = cdf_attrs.get_variable_attributes("lo-sw-species-unc-attrs")
    elif positions == PUI_POSITIONS:
        species_attrs = cdf_attrs.get_variable_attributes("lo-pui-species-attrs")
        unc_attrs = cdf_attrs.get_variable_attributes("lo-pui-species-unc-attrs")
    else:
        species_attrs = cdf_attrs.get_variable_attributes("lo-species-attrs")
        unc_attrs = cdf_attrs.get_variable_attributes("lo-species-unc-attrs")

    # update species attrs
    for species in species_list:
        attrs = unc_attrs if "unc" in unc_attrs else species_attrs
        # Replace {species} and {direction} in attrs
        attrs = apply_replacements_to_attrs(attrs, {"species": species})
        dataset[species].attrs.update(attrs)

    return dataset


def process_lo_angular_intensity(
    dataset: xr.Dataset,
    species_list: list,
    geometric_factors: xr.DataArray,
    efficiency: pd.DataFrame,
    positions: list,
) -> xr.Dataset:
    """
    Process the lo-species L2 dataset to calculate angular intensities.

    Parameters
    ----------
    dataset : xarray.Dataset
        The L2 dataset to process.
    species_list : list
        List of species variable names to calculate intensity.
    geometric_factors : xarray.DataArray
        The geometric factors array with shape (epoch, esa_steps).
    efficiency : pandas.DataFrame
        The efficiency lookup table.
    positions : list
        A list of position indices to select from the geometric factor and
        efficiency lookup tables.

    Returns
    -------
    xarray.Dataset
        The updated L2 dataset with angular intensities calculated.
    """
    # Calculate the angular intensities using the provided geometric factors and
    # efficiency.
    dataset = calculate_intensity(
        dataset,
        species_list,
        geometric_factors,
        efficiency,
        positions,
        average_across_positions=False,
    )

    # transform positions to elevation angles
    if positions == SW_POSITIONS:
        pos_to_el = LO_POSITION_TO_ELEVATION_ANGLE["sw"]
        position_index_to_adjust = 0
        direction = "Sunward"
    elif positions == NSW_POSITIONS:
        pos_to_el = LO_POSITION_TO_ELEVATION_ANGLE["nsw"]
        position_index_to_adjust = 9
        direction = "Non-Sunward"
    else:
        raise ValueError("Unknown positions for elevation angle mapping.")

    # Create a new coordinate for elevation_angle based on inst_az
    dataset = dataset.assign_coords(
        elevation_angle=(
            "inst_az",
            [pos_to_el[pos] for pos in dataset["inst_az"].data],
        )
    )
    # add uncertainties to species list
    species_list = species_list + [f"unc_{var}" for var in species_list]

    # Take the mean across elevation angles and restore the original dimension order
    dataset_converted = (
        dataset[species_list]
        .groupby("elevation_angle")
        .sum(keep_attrs=True, skipna=False)  # One position should always contain zeros
        # so sum is safe
        # Restore original dimension order because groupby moves the grouped
        # dimension to the front
        .transpose("epoch", "esa_step", "spin_sector", "elevation_angle", ...)
    )
    # Create a new coordinate for spin angle based on spin_sector
    # Use equation from section 11.2.2 of algorithm document
    dataset = dataset.assign_coords(
        spin_angle=("spin_sector", dataset["spin_sector"].data * 15.0 + 7.5)
    )
    dataset = dataset.drop_vars(species_list).merge(dataset_converted)

    # Positions 0 and 10 only observe half of the 24 spins for each esa step.
    # To account for this, we replicate the counts observed in position 0 and 10 for
    # each esa step to either spin angles 0-11 or 12-23, depending on the pixel
    # orientation (A/B). See section 11.2.2 of the CoDICE algorithm document
    # Use the variable "half_spin_per_esa_step" to determine the pixel orientations.
    # When the half spin number is even, the configuration is A and when the half spin
    # is odd, the configuration is B.
    # TODO handle when half_spin_per_esa_step changes in the middle of the dataset
    half_spin_per_esa_step = dataset["half_spin_per_esa_step"].data[0]
    # only consider valid half spin values
    valid_half_spin = half_spin_per_esa_step != HALF_SPIN_FILLVAL
    a_inds = np.nonzero(valid_half_spin & (half_spin_per_esa_step % 2 == 0))[0]
    b_inds = np.nonzero(valid_half_spin & (half_spin_per_esa_step % 2 == 1))[0]

    position_index = position_index_to_adjust
    for species in species_list:
        # Create a copy of the dataset to avoid modifying the original
        species_data = dataset[species].data.copy()
        # Determine the correct spin indices based on the position
        spin_sectors = dataset["spin_sector"].data
        spin_inds_1 = np.where(spin_sectors >= 12)[0]
        spin_inds_2 = np.where(spin_sectors < 12)[0]

        # if position_index is 9, swap the spin indices
        if position_index == 9:
            spin_inds_1, spin_inds_2 = spin_inds_2, spin_inds_1

        # Assign the values to the correct positions and spin sectors
        dataset[species].values[
            :, a_inds[:, np.newaxis], spin_inds_1, position_index
        ] = species_data[:, a_inds[:, np.newaxis], spin_inds_2, position_index]

        dataset[species].values[
            :, b_inds[:, np.newaxis], spin_inds_2, position_index
        ] = species_data[:, b_inds[:, np.newaxis], spin_inds_1, position_index]

    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_variable_attrs("codice", "l2-lo-angular")
    species_attrs = cdf_attrs.get_variable_attributes("lo-angular-attrs")
    unc_attrs = cdf_attrs.get_variable_attributes("lo-angular-unc-attrs")

    # update species attrs
    for species in species_list:
        attrs = unc_attrs if "unc" in species else species_attrs
        # Replace {species} and {direction} in attrs
        attrs = apply_replacements_to_attrs(
            attrs, {"species": species, "direction": direction}
        )
        dataset[species].attrs.update(attrs)

    # make sure elevation_angle is a coordinate and has the right attrs
    dataset["elevation_angle"].attrs.update(
        cdf_attrs.get_variable_attributes("elevation_angle", check_schema=False)
    )
    dataset["elevation_angle_label"] = xr.DataArray(
        dataset["elevation_angle"].data.astype(str),
        dims=("elevation_angle",),
        attrs=cdf_attrs.get_variable_attributes(
            "elevation_angle_label", check_schema=False
        ),
    )
    # update spin angle attributes
    dataset["spin_angle"].attrs = cdf_attrs.get_variable_attributes(
        "spin_angle", check_schema=False
    )
    # update spin sector attributes
    dataset["spin_sector"].attrs = cdf_attrs.get_variable_attributes(
        "spin_sector", check_schema=False
    )

    return dataset


def process_hi_omni(dependencies: ProcessingInputCollection) -> xr.Dataset:
    """
    Process the hi-omni L1B dataset to calculate omni-directional intensities.

    See section 11.1.3 of the CoDICE algorithm document for details.

    The formula for omni-directional intensities is::

        l1B species data / (
            geometric_factor * number_of_ssd * efficiency * energy_passband
        )

    Geometric factor is constant for all species which is 0.013.
    Number of SSD is constant for all species which is 12.
    Efficiency is provided in a CSV file for each species and energy bin.
    Energy passband is calculated from L1B variables energy_bin_minus + energy_bin_plus

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        The collection of processing input files.

    Returns
    -------
    xarray.Dataset
        The updated L2 dataset with omni-directional intensities calculated.
    """
    l1b_file = dependencies.get_file_paths(descriptor="hi-omni")[0]
    l1b_dataset = load_cdf(l1b_file)

    # Read the efficiencies data from the CSV file
    efficiencies_file = dependencies.get_file_paths(descriptor="l2-hi-omni-efficiency")[
        0
    ]
    efficiencies_df = pd.read_csv(efficiencies_file)
    # Omni product has 8 species and each species has different shape.
    # Eg.
    #   h - (epoch, 15)
    #   c - (epoch, 18)
    #   uh - (epoch, 5)
    #   etc.
    # Because of that, we need to loop over each species and calculate
    # omni-directional intensities separately.
    # Read geometric factor. It is labeled as GF in the CSV file
    geometric_factor = efficiencies_df[efficiencies_df["species"] == "GF"].values[0][-1]
    for species in HI_OMNI_VARIABLE_NAMES:
        # replace '_' with '-' to match CSV species naming
        species_csv_name = species.replace("_", "-")
        species_data = efficiencies_df[efficiencies_df["species"] == species_csv_name]
        # Read current species' efficiency
        species_efficiencies = species_data["average_efficiency"].values[np.newaxis, :]
        # Calculate energy passband from L1B data
        energy_passbands = (
            l1b_dataset[f"energy_{species}_plus"]
            + l1b_dataset[f"energy_{species}_minus"]
        ).values[np.newaxis, :]
        # Calculate omni-directional intensities
        omni_direction_intensities = l1b_dataset[species] / (
            geometric_factor * species_efficiencies * energy_passbands
        )
        # Store by replacing existing species data with omni-directional intensities
        l1b_dataset[species].values = omni_direction_intensities

        # Calculate uncertainty if available
        species_uncertainty = f"unc_{species}"
        if species_uncertainty in l1b_dataset:
            omni_uncertainties = l1b_dataset[species_uncertainty] / (
                geometric_factor * species_efficiencies * energy_passbands
            )
            # Store by replacing existing uncertainty data with omni-directional
            # uncertainties
            l1b_dataset[species_uncertainty].values = omni_uncertainties

    # TODO: this may go away once Joey and I fix L1B CDF
    # Update global CDF attributes
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l2-hi-omni")
    l1b_dataset.attrs = cdf_attrs.get_global_attributes("imap_codice_l2_hi-omni")

    # TODO: ask Joey to add attrs for epoch_delta_plus and epoch_delta_minus
    # and update dimension to be 'epoch' in L1B data
    for variable in l1b_dataset.data_vars:
        if variable in ["epoch_delta_plus", "epoch_delta_minus", "data_quality"]:
            l1b_dataset[variable].attrs = cdf_attrs.get_variable_attributes(
                variable, check_schema=False
            )
        else:
            l1b_dataset[variable].attrs = cdf_attrs.get_variable_attributes(
                variable, check_schema=False
            )

    # Add these new coordinates
    new_coords = {
        "energy_h": l1b_dataset["energy_h"],
        "energy_h_label": xr.DataArray(
            l1b_dataset["energy_h"].values.astype(str),
            dims=("energy_h",),
            attrs=cdf_attrs.get_variable_attributes(
                "energy_h_label", check_schema=False
            ),
        ),
        "energy_he3": l1b_dataset["energy_he3"],
        "energy_he3_label": xr.DataArray(
            l1b_dataset["energy_he3"].values.astype(str),
            dims=("energy_he3",),
            attrs=cdf_attrs.get_variable_attributes(
                "energy_he3_label", check_schema=False
            ),
        ),
        "energy_he4": l1b_dataset["energy_he4"],
        "energy_he4_label": xr.DataArray(
            l1b_dataset["energy_he4"].values.astype(str),
            dims=("energy_he4",),
            attrs=cdf_attrs.get_variable_attributes(
                "energy_he4_label", check_schema=False
            ),
        ),
        "energy_c": l1b_dataset["energy_c"],
        "energy_c_label": xr.DataArray(
            l1b_dataset["energy_c"].values.astype(str),
            dims=("energy_c",),
            attrs=cdf_attrs.get_variable_attributes(
                "energy_c_label", check_schema=False
            ),
        ),
        "energy_o": l1b_dataset["energy_o"],
        "energy_o_label": xr.DataArray(
            l1b_dataset["energy_o"].values.astype(str),
            dims=("energy_o",),
            attrs=cdf_attrs.get_variable_attributes(
                "energy_o_label", check_schema=False
            ),
        ),
        "energy_ne_mg_si": l1b_dataset["energy_ne_mg_si"],
        "energy_ne_mg_si_label": xr.DataArray(
            l1b_dataset["energy_ne_mg_si"].values.astype(str),
            dims=("energy_ne_mg_si",),
            attrs=cdf_attrs.get_variable_attributes(
                "energy_ne_mg_si_label", check_schema=False
            ),
        ),
        "energy_fe": l1b_dataset["energy_fe"],
        "energy_fe_label": xr.DataArray(
            l1b_dataset["energy_fe"].values.astype(str),
            dims=("energy_fe",),
            attrs=cdf_attrs.get_variable_attributes(
                "energy_fe_label", check_schema=False
            ),
        ),
        "energy_uh": l1b_dataset["energy_uh"],
        "energy_uh_label": xr.DataArray(
            l1b_dataset["energy_uh"].values.astype(str),
            dims=("energy_uh",),
            attrs=cdf_attrs.get_variable_attributes(
                "energy_uh_label", check_schema=False
            ),
        ),
        "energy_junk": l1b_dataset["energy_junk"],
        "energy_junk_label": xr.DataArray(
            l1b_dataset["energy_junk"].values.astype(str),
            dims=("energy_junk",),
            attrs=cdf_attrs.get_variable_attributes(
                "energy_junk_label", check_schema=False
            ),
        ),
        "epoch": xr.DataArray(
            l1b_dataset["epoch"].data,
            dims=("epoch",),
            attrs=cdf_attrs.get_variable_attributes("epoch", check_schema=False),
        ),
    }
    l1b_dataset = l1b_dataset.assign_coords(new_coords)

    return l1b_dataset


def process_hi_sectored(dependencies: ProcessingInputCollection) -> xr.Dataset:
    """
    Process the hi-omni L1B dataset to calculate omni-directional intensities.

    See section 11.1.2 of the CoDICE algorithm document for details.

    The formula for omni-directional intensities is::

        l1b species data / (geometric_factor * efficiency * energy_passband)

    Geometric factor is constant for all species and is 0.013.
    Efficiency is provided in a CSV file for each species and energy bin and
    position.
    Energy passband is calculated from energy_bin_minus + energy_bin_plus

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        The collection of processing input files.

    Returns
    -------
    xarray.Dataset
        The updated L2 dataset with omni-directional intensities calculated.
    """
    file_path = dependencies.get_file_paths(descriptor="hi-sectored")[0]
    l1b_dataset = load_cdf(file_path)

    # Update global CDF attributes
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l2-hi-sectored")

    # Overwrite L1B variable attributes with L2 variable attributes
    l2_dataset = xr.Dataset(
        coords={
            "spin_sector": l1b_dataset["spin_sector"],
            "spin_sector_label": xr.DataArray(
                l1b_dataset["spin_sector"].values.astype(str),
                dims=("spin_sector",),
                attrs=cdf_attrs.get_variable_attributes(
                    "spin_sector_label", check_schema=False
                ),
            ),
            "energy_h": l1b_dataset["energy_h"],
            "energy_h_label": xr.DataArray(
                l1b_dataset["energy_h"].values.astype(str),
                dims=("energy_h",),
                attrs=cdf_attrs.get_variable_attributes(
                    "energy_h_label", check_schema=False
                ),
            ),
            "energy_he3he4": l1b_dataset["energy_he3he4"],
            "energy_he3he4_label": xr.DataArray(
                l1b_dataset["energy_he3he4"].values.astype(str),
                dims=("energy_he3he4",),
                attrs=cdf_attrs.get_variable_attributes(
                    "energy_he3he4_label", check_schema=False
                ),
            ),
            "energy_cno": l1b_dataset["energy_cno"],
            "energy_cno_label": xr.DataArray(
                l1b_dataset["energy_cno"].values.astype(str),
                dims=("energy_cno",),
                attrs=cdf_attrs.get_variable_attributes(
                    "energy_cno_label", check_schema=False
                ),
            ),
            "energy_fe": l1b_dataset["energy_fe"],
            "energy_fe_label": xr.DataArray(
                l1b_dataset["energy_fe"].values.astype(str),
                dims=("energy_fe",),
                attrs=cdf_attrs.get_variable_attributes(
                    "energy_fe_label", check_schema=False
                ),
            ),
            "epoch": l1b_dataset["epoch"],
            "elevation_angle": xr.DataArray(
                HI_L2_ELEVATION_ANGLE,
                dims=("elevation_angle",),
                attrs=cdf_attrs.get_variable_attributes(
                    "elevation_angle", check_schema=False
                ),
            ),
            "elevation_angle_label": xr.DataArray(
                HI_L2_ELEVATION_ANGLE.astype(str),
                dims=("elevation_angle",),
                attrs=cdf_attrs.get_variable_attributes(
                    "elevation_angle_label", check_schema=False
                ),
            ),
        },
        attrs=cdf_attrs.get_global_attributes("imap_codice_l2_hi-sectored"),
    )

    efficiencies_file = dependencies.get_file_paths(
        descriptor="l2-hi-sectored-efficiency"
    )[0]

    # Calculate sectored intensities
    efficiencies_df = pd.read_csv(efficiencies_file)
    # Similar to hi-omni, each species has different shape.
    # Because of that, we need to loop over each species and calculate
    # sectored intensities separately.
    for species in HI_SECTORED_VARIABLE_NAMES:
        # Efficiencies from dataframe maps to different dimension in L1B data.
        # For example:
        #   l1b species 'h' has shape:
        #       (epoch, 8, 12, 12) -> (time, energy, spin_sector, inst_az)
        #   efficiencies 'h' has shape after reading from CSV:
        #       (8, 12) -> (energy, inst_az)
        #       NOTE: 12 here maps to last 12 in above l1b dimension.
        # Because of this, it's easier to work with the data in xarray.
        # Xarray automatically aligns dimensions and coordinates, making it easier
        # to work with multi-dimensional data. Thus, we convert the efficiencies
        # to xarray.DataArray with dimensions (energy, inst_az)
        species_data = efficiencies_df[efficiencies_df["species"] == species].values
        species_efficiencies = xr.DataArray(
            species_data[:, 2:].astype(
                float
            ),  # Skip first two columns (species, energy_bin)
            dims=(f"energy_{species}", "inst_az"),
            coords=l1b_dataset[[f"energy_{species}", "inst_az"]],
        )
        # Read geometric factor. It is labeled as GF in the CSV file
        geometric_factor = efficiencies_df[efficiencies_df["species"] == "GF"].values
        geometric_factor_da = xr.DataArray(
            geometric_factor[0, 2:].astype(
                np.float64
            ),  # Skip first two columns (species, energy_bin)
            dims="inst_az",
            coords=l1b_dataset[["inst_az"]],
        )
        # energy_passbands has shape:
        #   (8,) -> (energy)
        energy_passbands = xr.DataArray(
            l1b_dataset[f"energy_{species}_minus"]
            + l1b_dataset[f"energy_{species}_plus"],
            dims=(f"energy_{species}",),
            coords=l2_dataset[[f"energy_{species}"]],
            name="passband",
        )

        sectored_intensities = l1b_dataset[species] / (
            geometric_factor_da * species_efficiencies * energy_passbands
        )

        # Replace existing species data with omni-directional intensities
        l2_dataset[species] = xr.DataArray(
            sectored_intensities.data,
            dims=("epoch", f"energy_{species}", "spin_sector", "elevation_angle"),
            attrs=cdf_attrs.get_variable_attributes(species, check_schema=False),
        )
        # Calculate uncertainty if available
        species_uncertainty = f"unc_{species}"
        if species_uncertainty in l1b_dataset:
            sectored_uncertainties = l1b_dataset[species_uncertainty] / (
                geometric_factor_da * species_efficiencies * energy_passbands
            )
            l2_dataset[species_uncertainty] = xr.DataArray(
                sectored_uncertainties.data,
                dims=("epoch", f"energy_{species}", "spin_sector", "elevation_angle"),
                attrs=cdf_attrs.get_variable_attributes(
                    species_uncertainty, check_schema=False
                ),
            )

    # Calculate spin angle
    # Formula:
    #   θ_(k,n) = (θ_(k,0)+30°* n)  mod 360°
    # where
    #   n is size of L2_HI_SECTORED_ANGLE, 0 to 11,
    #   k is size of inst_az from l1b, 0 to 11,
    # Calculate spin angle by adding a base angle from L2_HI_SECTORED_ANGLE
    # for each SSD index and then adding multiple of 30 degrees for each elevation.
    # Then mod by 360 to keep it within 0-360 range.
    elevation_angles = np.arange(len(l2_dataset["elevation_angle"].values)) * 30.0
    spin_angle = (L2_HI_SECTORED_ANGLE[:, np.newaxis] + elevation_angles) % 360.0

    # Add spin angle variable using the new elevation_angle dimension
    l2_dataset["spin_angle"] = (("spin_sector", "elevation_angle"), spin_angle)
    l2_dataset["spin_angle"].attrs = cdf_attrs.get_variable_attributes(
        "spin_angle", check_schema=False
    )

    # Now carry over other variables from L1B to L2 dataset
    for variable in l1b_dataset.data_vars:
        if variable.startswith("epoch_") and variable != "epoch":
            # get attrs with just that name
            l2_dataset[variable] = xr.DataArray(
                l1b_dataset[variable].data,
                dims=("epoch",),
                attrs=cdf_attrs.get_variable_attributes(variable, check_schema=False),
            )
        elif variable.startswith("energy_"):
            l2_dataset[variable] = xr.DataArray(
                l1b_dataset[variable].data,
                dims=(f"energy_{variable.split('_')[1]}",),
                attrs=cdf_attrs.get_variable_attributes(variable, check_schema=False),
            )
        elif variable == "data_quality":
            l2_dataset[variable] = l1b_dataset[variable]
            l2_dataset[variable].attrs.update(
                cdf_attrs.get_variable_attributes(variable, check_schema=False)
            )

    l2_dataset["epoch"].attrs.update(
        cdf_attrs.get_variable_attributes("epoch", check_schema=False)
    )
    return l2_dataset


def process_lo_direct_events(dependencies: ProcessingInputCollection) -> xr.Dataset:
    """
    Process the lo-direct-events L1A dataset to convert variables to physical units.

    See section 11.2.1 of the CoDICE algorithm document for details.

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        The collection of processing input files.

    Returns
    -------
    xarray.Dataset
        The updated L2 dataset with variables converted to physical units.
    """
    file_path = dependencies.get_file_paths(descriptor="lo-direct-events")[0]
    l1a_dataset = load_cdf(file_path)

    # Update global CDF attributes
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l2-lo-direct-events")
    energy_table, energy_bins = get_lo_de_energy_luts(dependencies)
    # Convert from position to elevation angle in degrees relative to the spacecraft
    # axis
    l2_dataset = l1a_dataset
    # Create a new coordinate for elevation_angle based on inst_az
    pos_to_els = (
        LO_POSITION_TO_ELEVATION_ANGLE["sw"] | LO_POSITION_TO_ELEVATION_ANGLE["nsw"]
    )
    elevation_angle_shape = l2_dataset["position"].shape
    elevation_angle = np.array(
        [pos_to_els.get(pos, np.nan) for pos in l2_dataset["position"].values.flat]
    ).reshape(elevation_angle_shape)
    l2_dataset["elevation_angle"] = (
        l2_dataset["position"].dims,
        elevation_angle.astype(np.float32),
    )
    # Convert spin_sector to spin_angle in degrees
    # Use equation from section 11.2.2 of algorithm document
    # Shift all spin sectors for all positions 13 - 24 adding 12 and mod 24
    original_spin_sector = l2_dataset["spin_sector"].values
    l2_dataset["spin_sector"] = xr.where(
        (l2_dataset["position"] >= 13) & (l2_dataset["position"] <= 24),
        (l2_dataset["spin_sector"] + 12) % 24,
        l2_dataset["spin_sector"],
    )
    l2_dataset["spin_angle"] = l2_dataset["spin_sector"].astype(np.float32) * 15.0 + 7.5

    # Set spin angle and sector to NaN for invalid positions (>23)
    l2_dataset["spin_angle"] = xr.where(
        (original_spin_sector > 23), np.nan, l2_dataset["spin_angle"]
    )
    l2_dataset["spin_sector"] = xr.where(
        (original_spin_sector > 23), np.nan, l2_dataset["spin_sector"]
    )
    # convert apd energy to physical units
    # Set the gain labels based on gain values
    gains = l2_dataset["gain"].values.ravel()
    apd_ids = l2_dataset["apd_id"].values.ravel()
    apd_energy = l2_dataset["apd_energy"].values.ravel()
    apd_energy_shape = l2_dataset["apd_energy"].shape

    # The energy table lookup columns are ordered by apd_id and gain
    # E.g. APD-1-LG, APD-1-HG, ..., APD-29-LG
    # So we can get the col index like so: ind = apd_id * 2 + gain
    col_inds = apd_ids * 2 + gains
    # Get a mask of valid indices
    valid_mask = (
        (apd_energy < energy_table.shape[0])
        & (col_inds < energy_table.shape[1])
        & (apd_ids > 0)
    )
    # Initialize output array with NaNs
    energy_bins_inds = np.full(apd_energy.shape, np.nan)
    energy_kev = np.full(apd_energy.shape, np.nan)
    # The rows are apd_energy bins
    energy_bins_inds[valid_mask] = energy_table[
        apd_energy[valid_mask], col_inds[valid_mask]
    ]
    energy_kev[valid_mask] = energy_bins[energy_bins_inds[valid_mask].astype(int)]

    l2_dataset["apd_energy"].data = (
        np.array(energy_kev).astype(np.float32).reshape(apd_energy_shape)
    )

    # Calculate TOF in nanoseconds
    tof_bit_to_ns = get_mpq_calc_tof_conversion_vals(dependencies)
    tof_bits = l2_dataset["tof"].values.flatten()
    # Create output array
    tof_ns = np.full(tof_bits.shape, np.nan, dtype=np.float64)
    # Get only valid TOF bits between 0 and 1023
    valid_mask = (tof_bits >= 0) & (tof_bits < 1024)
    tof_ns[valid_mask] = tof_bit_to_ns[tof_bits[valid_mask]]
    # Reshape back to original shape
    l2_dataset["tof"].data = tof_ns.astype(np.float32).reshape(l2_dataset["tof"].shape)

    # Convert energy step to energy in keV
    esa_kev = get_mpq_calc_energy_conversion_vals(dependencies)
    energy_steps = l2_dataset["energy_step"].values.flatten()
    # Create output array
    kev = np.full(energy_steps.shape, np.nan, dtype=np.float64)
    # Get only valid energy_steps between 0 and 128
    valid_mask = (energy_steps >= 0) & (energy_steps < 128)
    kev[valid_mask] = esa_kev[energy_steps[valid_mask]]
    # Reshape back to original shape
    l2_dataset["energy_per_charge"] = (
        l2_dataset["energy_step"].dims,
        kev.astype(np.float32).reshape(l2_dataset["energy_step"].shape),
    )
    # Drop unused variables
    vars_to_drop = ["spare", "sw_bias_gain_mode", "st_bias_gain_mode", "k_factor"]
    l2_dataset = l2_dataset.drop_vars(vars_to_drop)
    # Update variable attributes
    l2_dataset.attrs.update(
        cdf_attrs.get_global_attributes("imap_codice_l2_lo-direct-events")
    )
    for var in l2_dataset.data_vars:
        l2_dataset[var].attrs.update(cdf_attrs.get_variable_attributes(var))
    # Update coord attributes
    l2_dataset["priority"].attrs.update(
        cdf_attrs.get_variable_attributes("priority", check_schema=False)
    )
    l2_dataset["event_num"].attrs.update(
        cdf_attrs.get_variable_attributes("event_num", check_schema=False)
    )
    l2_dataset["epoch"] = xr.DataArray(
        l2_dataset["epoch"].data,
        dims="epoch",
        attrs=cdf_attrs.get_variable_attributes("epoch", check_schema=False),
    )
    # Add labels
    l2_dataset["event_num_label"] = xr.DataArray(
        l2_dataset["event_num"].values.astype(str).astype("<U5"),
        dims=("event_num",),
        attrs=cdf_attrs.get_variable_attributes("event_num_label", check_schema=False),
    )
    l2_dataset["priority_label"] = xr.DataArray(
        l2_dataset["priority_label"].values.astype("<U1"),
        dims=("priority",),
        attrs=cdf_attrs.get_variable_attributes("priority_label", check_schema=False),
    )

    return l2_dataset


def process_hi_direct_events(dependencies: ProcessingInputCollection) -> xr.Dataset:
    """
    Process the hi-direct-events L1A dataset to convert variables to physical units.

    See section 11.2.1 of the CoDICE algorithm document for details.

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        The collection of processing input files.

    Returns
    -------
    xarray.Dataset
        The updated L2 dataset with variables converted to physical units.
    """
    file_path = dependencies.get_file_paths(descriptor="hi-direct-events")[0]
    l1a_dataset = load_cdf(file_path)

    # Update global CDF attributes
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l2-hi-direct-events")

    l2_dataset = l1a_dataset
    # Load energy table and tof table needed for conversions
    energy_table, tof_table = get_hi_de_luts(dependencies)
    # Initialize nan array for calculations
    nan_array = np.full(l2_dataset["ssd_id"].shape, np.nan)
    # Convert from position to elevation angle in degrees relative to the spacecraft
    # axis
    ssd_id = l2_dataset["ssd_id"].values
    valid_ssd = (ssd_id <= 15) & (ssd_id >= 0)

    elevation = nan_array.copy()
    elevation[valid_ssd] = SSD_ID_TO_ELEVATION[ssd_id[valid_ssd]]
    l2_dataset["elevation_angle"] = xr.DataArray(
        data=elevation.astype(np.float32), dims=l2_dataset["ssd_id"].dims
    )
    # Calculate ssd energy in meV
    gain = l2_dataset["gain"].values
    ssd_energy = l2_dataset["ssd_energy"].values
    valid_mask = (
        (np.isin(gain, list(GAIN_ID_TO_STR.keys())))
        & valid_ssd
        & (ssd_energy != len(energy_table))
    )
    # The columns are organized in order of id and gains
    # E.g. ssd 0 - LG, ssd 0 - MG, ssd 0 - HG, ssd 1 - LG, ssd 1 - MG, ssd 1 - HG, ...
    cols = ssd_id * 3 + (gain - 1)
    ssd_energy_converted = nan_array.copy()
    ssd_energy_converted[valid_mask] = energy_table[
        ssd_energy[valid_mask], cols[valid_mask]
    ]
    l2_dataset["ssd_energy"].data = ssd_energy_converted.astype(np.float32)

    # Convert spin_sector to spin_angle in degrees
    theta_angles = nan_array.copy()
    theta_angles[valid_ssd] = SSD_ID_TO_SPIN_ANGLE[ssd_id[valid_ssd]]
    l2_dataset["spin_angle"] = (
        (theta_angles + 15.0 * l2_dataset["spin_sector"]) % 360.0
    ).astype(np.float32)

    # Calculate TOF in ns
    tof = l2_dataset["tof"].values
    # Get valid TOF indices for lookup
    valid_tof_mask = tof < tof_table.shape[0]
    tof_ns = nan_array.copy()
    # Get tof values in ns from first column of tof_table
    tof_ns[valid_tof_mask] = tof_table[tof[valid_tof_mask], 0]
    l2_dataset["tof"] = xr.DataArray(
        data=tof_ns,
        dims=l2_dataset["tof"].dims,
    ).astype(np.float32)

    # Calculate energy per nuc
    energy_nuc = nan_array.copy()
    # Get value from second column of tof_table (E/n (MeV/n))
    energy_nuc[valid_tof_mask] = tof_table[tof[valid_tof_mask], 1]
    l2_dataset["energy_per_nuc"] = xr.DataArray(
        data=energy_nuc,
        dims=l2_dataset["tof"].dims,
    ).astype(np.float32)
    # Drop unused variables
    vars_to_drop = ["spare", "sw_bias_gain_mode", "st_bias_gain_mode"]
    l2_dataset = l2_dataset.drop_vars(vars_to_drop)
    # Update variable attributes
    l2_dataset.attrs.update(
        cdf_attrs.get_global_attributes("imap_codice_l2_hi-direct-events")
    )
    for var in l2_dataset.data_vars:
        l2_dataset[var].attrs.update(cdf_attrs.get_variable_attributes(var))
    # Update coord attributes
    l2_dataset["priority"].attrs.update(
        cdf_attrs.get_variable_attributes("priority", check_schema=False)
    )
    l2_dataset["event_num"].attrs.update(
        cdf_attrs.get_variable_attributes("event_num", check_schema=False)
    )
    l2_dataset["epoch"] = xr.DataArray(
        l2_dataset["epoch"].data,
        dims="epoch",
        attrs=cdf_attrs.get_variable_attributes("epoch", check_schema=False),
    )
    # Add labels
    l2_dataset["event_num_label"] = xr.DataArray(
        l2_dataset["event_num"].values.astype(str).astype("<U5"),
        dims=("event_num",),
        attrs=cdf_attrs.get_variable_attributes("event_num_label", check_schema=False),
    )
    l2_dataset["priority_label"] = xr.DataArray(
        l2_dataset["priority_label"].values.astype("<U1"),
        dims=("priority",),
        attrs=cdf_attrs.get_variable_attributes("priority_label", check_schema=False),
    )

    return l2_dataset


def process_codice_l2(
    descriptor: str, dependencies: ProcessingInputCollection
) -> xr.Dataset:
    """
    Will process CoDICE l1 data to create l2 data products.

    Parameters
    ----------
    descriptor : str
        The descriptor for the CoDICE L1 file to process.
    dependencies : ProcessingInputCollection
        Collection of processing inputs such as ancillary data files.

    Returns
    -------
    l2_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    # This should get science files since ancillary or spice doesn't have data_type
    # as data level.
    file_path = dependencies.get_file_paths(descriptor=descriptor)[0]
    # Now form product name from descriptor
    descriptor = ScienceFilePath(file_path).descriptor
    dataset_name = f"imap_codice_l2_{descriptor}"
    # TODO: update list of datasets that need geometric factors (if needed)
    # Compute geometric factors needed for intensity calculations
    if dataset_name in [
        "imap_codice_l2_lo-sw-species",
        "imap_codice_l2_lo-nsw-species",
        "imap_codice_l2_lo-nsw-angular",
        "imap_codice_l2_lo-sw-angular",
    ]:
        cdf_attrs = ImapCdfAttributes()
        cdf_attrs.add_instrument_global_attrs("codice")

        l2_dataset = load_cdf(file_path).copy()

        geometric_factor_lookup = get_geometric_factor_lut(dependencies)
        efficiency_lookup = get_efficiency_lut(dependencies)
        geometric_factors = compute_geometric_factors(
            l2_dataset, geometric_factor_lookup
        )

        if dataset_name == "imap_codice_l2_lo-sw-species":
            # Filter the efficiency lookup table for solar wind efficiencies
            efficiencies = efficiency_lookup[efficiency_lookup["product"] == "sw"]
            # Calculate the pickup ion sunward solar wind intensities using equation
            # described in section 11.2.3 of algorithm document.
            l2_dataset = process_lo_species_intensity(
                l2_dataset,
                LO_SW_PICKUP_ION_SPECIES_VARIABLE_NAMES,
                geometric_factors,
                efficiencies,
                PUI_POSITIONS,
            )
            # Calculate the sunward solar wind species intensities using equation
            # described in section 11.2.3 of algorithm document.
            l2_dataset = process_lo_species_intensity(
                l2_dataset,
                LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES,
                geometric_factors,
                efficiencies,
                SOLAR_WIND_POSITIONS,
            )
            l2_dataset.attrs.update(
                cdf_attrs.get_global_attributes("imap_codice_l2_lo-sw-species")
            )
        elif dataset_name == "imap_codice_l2_lo-nsw-species":
            # Filter the efficiency lookup table for non-solar wind efficiencies
            efficiencies = efficiency_lookup[efficiency_lookup["product"] == "nsw"]
            # Calculate the non-sunward species intensities using equation
            # described in section 11.2.3 of algorithm document.
            l2_dataset = process_lo_species_intensity(
                l2_dataset,
                LO_NSW_SPECIES_VARIABLE_NAMES,
                geometric_factors,
                efficiencies,
                NSW_POSITIONS,
            )
            l2_dataset.attrs.update(
                cdf_attrs.get_global_attributes("imap_codice_l2_lo-nsw-species")
            )
        elif dataset_name == "imap_codice_l2_lo-sw-angular":
            efficiencies = efficiency_lookup[efficiency_lookup["product"] == "sw"]
            # Calculate the sunward solar wind angular intensities using equation
            # described in section 11.2.2 of algorithm document.
            l2_dataset = process_lo_angular_intensity(
                l2_dataset,
                LO_SW_ANGULAR_VARIABLE_NAMES,
                geometric_factors,
                efficiencies,
                SW_POSITIONS,
            )
            l2_dataset.attrs.update(
                cdf_attrs.get_global_attributes("imap_codice_l2_lo-sw-angular")
            )
        if dataset_name == "imap_codice_l2_lo-nsw-angular":
            # Calculate the non sunward angular intensities
            efficiencies = efficiency_lookup[efficiency_lookup["product"] == "nsw"]
            l2_dataset = process_lo_angular_intensity(
                l2_dataset,
                LO_NSW_ANGULAR_VARIABLE_NAMES,
                geometric_factors,
                efficiencies,
                NSW_POSITIONS,
            )
            l2_dataset.attrs.update(
                cdf_attrs.get_global_attributes("imap_codice_l2_lo-nsw-angular")
            )
        # Drop vars not needed in L2
        l2_dataset = l2_dataset.drop_vars(
            [
                "acquisition_time_per_esa_step",
                "rgfo_half_spin",
                "half_spin_per_esa_step",
                "energy_table",
            ]
        )

    if dataset_name in [
        "imap_codice_l2_hi-counters-singles",
        "imap_codice_l2_hi-counters-aggregated",
        "imap_codice_l2_lo-counters-singles",
        "imap_codice_l2_lo-counters-aggregated",
        "imap_codice_l2_lo-sw-priority",
        "imap_codice_l2_lo-nsw-priority",
    ]:
        # No changes needed. Just save to an L2 CDF file.
        # TODO: May not even need L2 files for these products
        pass

    elif dataset_name == "imap_codice_l2_hi-direct-events":
        # Convert the following data variables to physical units using
        # calibration data:
        #    - ssd_energy
        #    - tof
        #    - elevation_angle
        #    - spin_angle
        # These converted variables are *in addition* to the existing L1 variables
        # The other data variables require no changes
        # See section 11.1.2 of algorithm document
        l2_dataset = process_hi_direct_events(dependencies)

    elif dataset_name == "imap_codice_l2_hi-sectored":
        # Convert the sectored count rates using equation described in section
        # 11.1.3 of algorithm document.
        l2_dataset = process_hi_sectored(dependencies)

    elif dataset_name == "imap_codice_l2_hi-omni":
        # Calculate the omni-directional intensity for each species using
        # equation described in section 11.1.4 of algorithm document
        # hopefully this can also apply to hi-ialirt
        l2_dataset = process_hi_omni(dependencies)

    elif dataset_name == "imap_codice_l2_lo-direct-events":
        # Convert the following data variables to physical units using
        # calibration data:
        #    - apd_energy
        #    - elevation_angle
        #    - tof
        #    - spin_sector
        #    - esa_step
        # These converted variables are *in addition* to the existing L1 variables
        # The other data variables require no changes
        # See section 11.1.2 of algorithm document
        l2_dataset = process_lo_direct_events(dependencies)

    # logger.info(f"\nFinal data product:\n{l2_dataset}\n")

    return l2_dataset
