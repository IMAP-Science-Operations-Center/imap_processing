"""Keys for I-ALiRT data products."""

# ---- the actual mapping ----
IALIRT_DIMS = {
    # H intensities in 15 energy ranges and binned into 4 azimuths and 4 spin angle bins
    "codice_hi_h": [
        "codice_hi_epoch",
        "codice_hi_h_energy_range",
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
    "codice_lo_c_plus_6_over_c_plus_5_ratio": ["codice_lo_epoch"],
    # O+7/O+6 charge state ratio
    "codice_lo_o_plus_7_over_o_plus_6_ratio": ["codice_lo_epoch"],
    # Fe low/Fe high charge state ratio
    "codice_lo_fe_low_over_fe_high_ratio": ["codice_lo_epoch"],
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
    "swe_normalized_counts": ["swe_epoch", "swe_electron_energy_labels"],
    # SWE Counterstreaming flag
    "swe_counterstreaming_electrons": ["swe_epoch"],
    # Spacecraft position in GSE coordinates
    "sc_position_GSE": ["spacecraft_epoch", "sc_GSE_labels"],
    # Spacecraft velocity in GSE coordinates
    "sc_velocity_GSE": ["spacecraft_epoch", "sc_GSE_labels"],
    # Spacecraft position in GSM coordinates
    "sc_position_GSM": ["spacecraft_epoch", "sc_GSM_labels"],
    # Spacecraft velocity in GSM coordinates
    "sc_velocity_GSM": ["spacecraft_epoch", "sc_GSM_labels"],
}

codice_energy_bounds = [
    (0.020, 0.028),
    (0.028, 0.040),
    (0.040, 0.057),
    (0.057, 0.080),
    (0.080, 0.113),
    (0.113, 0.160),
    (0.160, 0.226),
    (0.226, 0.320),
    (0.320, 0.453),
    (0.453, 0.640),
    (0.640, 0.905),
    (0.905, 1.280),
    (1.280, 1.810),
    (1.810, 2.560),
    (2.560, 3.620),
]

swe_energy_labels = [
    "100.4",
    "140.0",
    "194.0",
    "270.0",
    "376.0",
    "523.0",
    "727.0",
    "1011.0",
]
