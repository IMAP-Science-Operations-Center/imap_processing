"""Keys for I-ALiRT data products."""

import numpy as np

from imap_processing.codice.constants import (
    HI_IALIRT_REF_SPIN_ANGLE,
)

IALIRT_DIMS = {
    # H intensities in 15 energy ranges and binned into 4 azimuths and 4 spin angle bins
    "codice_hi_h": [
        "codice_hi_epoch",
        "codice_hi_energy_center",
        "codice_hi_spin_sector",
        "codice_hi_elevation",
    ],
    # C/O abundance ratio
    "codice_lo_c_over_o_abundance": ["codice_lo_epoch"],
    # Mg/O abundance ratio
    "codice_lo_mg_over_o_abundance": ["codice_lo_epoch"],
    # Fe/O abundance ratio
    "codice_lo_fe_over_o_abundance": ["codice_lo_epoch"],
    # C+6/C+5 charge state ratio
    "codice_lo_c_plus_6_over_c_plus_5": ["codice_lo_epoch"],
    # O+7/O+6 charge state ratio
    "codice_lo_o_plus_7_over_o_plus_6": ["codice_lo_epoch"],
    # Fe low/Fe high charge state ratio
    "codice_lo_fe_low_over_fe_high": ["codice_lo_epoch"],
    # Low energy (>0.5 MeV) electrons (A-side)
    "hit_e_a_side_low_en": ["hit_epoch"],
    # Medium energy (<1 MeV) electrons (A-side)
    "hit_e_a_side_med_en": ["hit_epoch"],
    # Low energy (>0.5 MeV) electrons (B-side)
    "hit_e_b_side_low_en": ["hit_epoch"],
    # Medium energy (<1 MeV) electrons (B-side)
    "hit_e_b_side_med_en": ["hit_epoch"],
    # Low energy (6 to 8 MeV) protons (Omnidirectional)
    "hit_h_omni_low_en": ["hit_epoch"],
    # Medium energy (12 to 15 MeV) protons (Omnidirectional)
    "hit_h_omni_med_en": ["hit_epoch"],
    # Low energy (6 to 8 MeV/nuc) He (Omnidirectional)
    "hit_he_omni_low_en": ["hit_epoch"],
    # High energy (15 to 70 MeV/nuc) He (Omnidirectional)
    "hit_he_omni_high_en": ["hit_epoch"],
    # Magnitude of the magnetic field vector
    "mag_B_magnitude": ["mag_epoch"],
    # Elevation angle (θ) of the magnetic field in GSE coordinates
    "mag_theta_B_GSE": ["mag_epoch"],
    # Azimuth angle (φ) of the magnetic field in GSE coordinates
    "mag_phi_B_GSE": ["mag_epoch"],
    # Elevation angle (θ) of the magnetic field in GSM coordinates
    "mag_theta_B_GSM": ["mag_epoch"],
    # Azimuth angle (φ) of the magnetic field in GSM coordinates
    "mag_phi_B_GSM": ["mag_epoch"],
    # Magnetic field vector in GSE coordinates
    "mag_B_GSE": ["mag_epoch", "B_GSE_labels"],
    # Magnetic field vector in GSM coordinates
    "mag_B_GSM": ["mag_epoch", "B_GSM_labels"],
    # Magnetic field vector in RTN coordinates
    "mag_B_RTN": ["mag_epoch", "B_RTN_labels"],
    # Pseudo density of solar wind protons
    "swapi_pseudo_proton_density": ["swapi_epoch"],
    # Pseudo speed of solar wind protons in solar inertial frame
    "swapi_pseudo_proton_speed": ["swapi_epoch"],
    # Pseudo temperature of solar wind protons in plasma frame
    "swapi_pseudo_proton_temperature": ["swapi_epoch"],
    # SWE Normalized Counts
    "swe_normalized_counts": ["swe_epoch", "swe_electron_energy"],
    # SWE Counterstreaming flag
    "swe_counterstreaming_electrons": ["swe_epoch"],
    # Spacecraft position in GSE coordinates
    "sc_position_GSE": ["ephemeris_epoch", "sc_GSE_position_labels"],
    # Spacecraft velocity in GSE coordinates
    "sc_velocity_GSE": ["ephemeris_epoch", "sc_GSE_velocity_labels"],
    # Spacecraft position in GSM coordinates
    "sc_position_GSM": ["ephemeris_epoch", "sc_GSM_position_labels"],
    # Spacecraft velocity in GSM coordinates
    "sc_velocity_GSM": ["ephemeris_epoch", "sc_GSM_velocity_labels"],
}

IALIRT_DTYPES = {
    # H intensities in 15 energy ranges and binned into 4 azimuths and 4 spin angle bins
    "codice_hi_h": np.float32,
    # CoDICE-Lo abundance / ratios
    "codice_lo_c_over_o_abundance": np.float32,
    "codice_lo_mg_over_o_abundance": np.float32,
    "codice_lo_fe_over_o_abundance": np.float32,
    "codice_lo_c_plus_6_over_c_plus_5": np.float32,
    "codice_lo_o_plus_7_over_o_plus_6": np.float32,
    "codice_lo_fe_low_over_fe_high": np.float32,
    # HIT scalars
    "hit_e_a_side_low_en": np.uint32,
    "hit_e_a_side_med_en": np.uint32,
    "hit_e_b_side_low_en": np.uint32,
    "hit_e_b_side_med_en": np.uint32,
    "hit_h_omni_low_en": np.uint32,
    "hit_h_omni_med_en": np.uint32,
    "hit_he_omni_low_en": np.uint32,
    "hit_he_omni_high_en": np.uint32,
    # MAG
    "mag_epoch": np.int64,  # if you are treating this as a data variable
    "mag_B_magnitude": np.float32,
    "mag_B_RTN": np.float32,
    "mag_B_GSE": np.float32,
    "mag_theta_B_GSE": np.float32,
    "mag_phi_B_GSE": np.float32,
    "mag_B_GSM": np.float32,
    "mag_theta_B_GSM": np.float32,
    "mag_phi_B_GSM": np.float32,
    # SWAPI
    "swapi_pseudo_proton_density": np.float32,
    "swapi_pseudo_proton_speed": np.float32,
    "swapi_pseudo_proton_temperature": np.float32,
    # SWE
    "swe_normalized_counts": np.int64,
    "swe_counterstreaming_electrons": np.uint8,
    # Spacecraft vectors
    "sc_position_GSM": np.float32,
    "sc_velocity_GSM": np.float32,
    "sc_position_GSE": np.float32,
    "sc_velocity_GSE": np.float32,
}

hit_restricted_fields = {
    "hit_e_a_side_high_en",
    "hit_e_b_side_high_en",
    "hit_h_a_side_high_en",
    "hit_h_b_side_high_en",
}

codice_hi_energy_center = [
    0.02378414,
    0.03363586,
    0.04756828,
    0.06727171,
    0.09513657,
    0.13454343,
    0.19027314,
    0.26908685,
    0.38054628,
    0.53817371,
    0.76109255,
    1.07634741,
    1.52218511,
    2.15269482,
    3.04437021,
]
codice_hi_energy_minus = [
    0.00378414,
    0.00535159,
    0.00756828,
    0.01070317,
    0.01513657,
    0.02140634,
    0.03027314,
    0.04281268,
    0.06054628,
    0.08562537,
    0.12109255,
    0.17125073,
    0.24218511,
    0.34250146,
    0.48437021,
]
codice_hi_energy_plus = [
    0.00450013,
    0.00636414,
    0.00900026,
    0.01272829,
    0.01800052,
    0.02545657,
    0.03600103,
    0.05091315,
    0.07200206,
    0.10182629,
    0.14400413,
    0.20365259,
    0.28800825,
    0.40730518,
    0.57601651,
]

# Calculate spin angle
# Formula:
#   θ_(g,n) = (θ_(g,0)+90°* n)  mod 360°
# where
#   n is number of sectored angles, 0 to 3,
#   g is size of the group (inst_az), 0 to 3,
HI_IALIRT_SPIN_ANGLE = (
    HI_IALIRT_REF_SPIN_ANGLE[:, np.newaxis] + np.array([0, 1, 2, 3]) * 90
) % 360.0

swe_energy = [
    100.4,
    140.0,
    194.0,
    270.0,
    376.0,
    523.0,
    727.0,
    1011.0,
]
