"""Keys for I-ALiRT data products."""

IALIRT_KEYS = [
    # H intensities in 15 energy ranges and binned into 4 azimuths and 4 spin angle bins
    "codicehi_h",
    # C/O abundance ratio
    "codicelo_c_over_o_abundance",
    # Mg/O abundance ratio
    "codicelo_mg_over_o_abundance",
    # Fe/O abundance ratio
    "codicelo_fe_over_o_abundance",
    # C+6/C+5 charge state ratio
    "codicelo_c_plus_6_over_c_plus_5_ratio",
    # O+7/O+6 charge state ratio
    "codicelo_o_plus_7_over_o_plus_6_ratio",
    # Fe low/Fe high charge state ratio
    "codicelo_fe_low_over_fe_high_ratio",
    # Low energy (~300 keV) electrons (A-side)
    "hit_e_a_side_low_en",
    # Medium energy (~3 MeV) electrons (A-side)
    "hit_e_a_side_med_en",
    # High energy (>3 MeV) electrons (A-side)
    "hit_e_a_side_high_en",
    # Low energy (~300 keV) electrons (B-side)
    "hit_e_b_side_low_en",
    # Medium energy (~3 MeV) electrons (B-side)
    "hit_e_b_side_med_en",
    # High energy (>3 MeV) electrons (B-side)
    "hit_e_b_side_high_en",
    # Medium energy (12 to 70 MeV) protons (Omnidirectional)
    "hit_h_omni_med_en",
    # High energy (>70 MeV) protons (A-side)
    "hit_h_a_side_high_en",
    # High energy (>70 MeV) protons (B-side)
    "hit_h_b_side_high_en",
    # Low energy (6 to 8 MeV/nuc) He (Omnidirectional)
    "hit_he_omni_low_en",
    # High energy (15 to 70 MeV/nuc) He (Omnidirectional)
    "hit_he_omni_high_en",
    # Magnetic field vector in GSE coordinates
    "mag_4s_b_gse",
    # Magnetic field vector in GSM coordinates
    "mag_4s_b_gsm",
    # Magnetic field vector in RTN coordinates
    "mag_4s_b_rtn",
    # Azimuth angle (φ) of the magnetic field in GSM coordinates
    "mag_phi_4s_b_gsm",
    # Elevation angle (θ) of the magnetic field in GSM coordinates
    "mag_theta_4s_b_gsm",
    # Pseudo density of solar wind protons
    "swapi_pseudo_proton_density",
    # Pseudo speed of solar wind protons in solar inertial frame
    "swapi_pseudo_proton_speed",
    # Pseudo temperature of solar wind protons in plasma frame
    "swapi_pseudo_proton_temperature",
    # SWE Normalized Counts - Half Cycle 1
    "swe_normalized_counts_half_1",
    # SWE Normalized Counts - Half Cycle 2
    "swe_normalized_counts_half_2",
    # SWE Counterstreaming flag
    "swe_counterstreaming_electrons",
]
