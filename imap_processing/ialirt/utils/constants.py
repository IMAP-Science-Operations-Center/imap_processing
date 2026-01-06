"""Keys for I-ALiRT data products."""

IALIRT_KEYS = [
    # H intensities in 15 energy ranges and binned into 4 azimuths and 4 spin angle bins
    "codice_hi_h",
    # C/O abundance ratio
    "codice_lo_c_over_o_abundance",
    # Mg/O abundance ratio
    "codice_lo_mg_over_o_abundance",
    # Fe/O abundance ratio
    "codice_lo_fe_over_o_abundance",
    # C+6/C+5 charge state ratio
    "codice_lo_c_plus_6_over_c_plus_5_ratio",
    # O+7/O+6 charge state ratio
    "codice_lo_o_plus_7_over_o_plus_6_ratio",
    # Fe low/Fe high charge state ratio
    "codice_lo_fe_low_over_fe_high_ratio",
    # Low energy (>0.5 MeV) electrons (A-side)
    "hit_e_a_side_low_en",
    # Medium energy (<1 MeV) electrons (A-side)
    "hit_e_a_side_med_en",
    # Low energy (>0.5 MeV) electrons (B-side)
    "hit_e_b_side_low_en",
    # Medium energy (<1 MeV) electrons (B-side)
    "hit_e_b_side_med_en",
    # Low energy (6 to 8 MeV) protons (Omnidirectional)
    "hit_h_omni_low_en",
    # Medium energy (12 to 15 MeV) protons (Omnidirectional)
    "hit_h_omni_med_en",
    # Low energy (6 to 8 MeV/nuc) He (Omnidirectional)
    "hit_he_omni_low_en",
    # High energy (15 to 70 MeV/nuc) He (Omnidirectional)
    "hit_he_omni_high_en",
    # MAG instrument epoch
    "mag_epoch",
    # Magnitude of the magnetic field vector
    "mag_B_magnitude",
    # Magnetic field vector in RTN coordinates
    "mag_B_RTN",
    # Magnetic field vector in GSE coordinates
    "mag_B_GSE",
    # Elevation angle (θ) of the magnetic field in GSE coordinates
    "mag_theta_B_GSE",
    # Azimuth angle (φ) of the magnetic field in GSE coordinates
    "mag_phi_B_GSE",
    # Magnetic field vector in GSM coordinates
    "mag_B_GSM",
    # Elevation angle (θ) of the magnetic field in GSM coordinates
    "mag_theta_B_GSM",
    # Azimuth angle (φ) of the magnetic field in GSM coordinates
    "mag_phi_B_GSM",
    # Pseudo density of solar wind protons
    "swapi_pseudo_proton_density",
    # Pseudo speed of solar wind protons in solar inertial frame
    "swapi_pseudo_proton_speed",
    # Pseudo temperature of solar wind protons in plasma frame
    "swapi_pseudo_proton_temperature",
    # SWE Normalized Counts
    "swe_normalized_counts",
    # SWE Counterstreaming flag
    "swe_counterstreaming_electrons",
    # Spacecraft position in GSM coordinates
    "sc_position_GSM",
    # Spacecraft velocity in GSM coordinates
    "sc_velocity_GSM",
    # Spacecraft position in GSE coordinates
    "sc_position_GSE",
    # Spacecraft velocity in GSE coordinates
    "sc_velocity_GSE",
]
