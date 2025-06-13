from imap_processing import imap_module_directory

# ruff: noqa: E501
EXTERNAL_TEST_DATA = [

    # CoDICE
    ("imap_codice_l0_raw_20241110_v001.pkts", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l0_raw_20241110_v001.pkts"),
    ("imap_codice_l1a_hi-counters-aggregated_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hi-counters-aggregated_20241110_v999.cdf"),
    ("imap_codice_l1a_hi-counters-singles_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hi-counters-singles_20241110_v999.cdf"),
    ("imap_codice_l1a_hi-ialirt_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hi-ialirt_20241110_v999.cdf"),
    ("imap_codice_l1a_hi-omni_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hi-omni_20241110_v999.cdf"),
    ("imap_codice_l1a_hi-pha_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hi-pha_20241110_v999.cdf"),
    ("imap_codice_l1a_hi-priority_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hi-priority_20241110_v999.cdf"),
    ("imap_codice_l1a_hi-sectored_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hi-sectored_20241110_v999.cdf"),
    ("imap_codice_l1a_hskp_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hskp_20241110_v999.cdf"),
    ("imap_codice_l1a_lo-counters-aggregated_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-counters-aggregated_20241110_v999.cdf"),
    ("imap_codice_l1a_lo-counters-singles_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-counters-singles_20241110_v999.cdf"),
    ("imap_codice_l1a_lo-ialirt_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-ialirt_20241110_v999.cdf"),
    ("imap_codice_l1a_lo-nsw-angular_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-nsw-angular_20241110_v999.cdf"),
    ("imap_codice_l1a_lo-nsw-priority_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-nsw-priority_20241110_v999.cdf"),
    ("imap_codice_l1a_lo-nsw-species_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-nsw-species_20241110_v999.cdf"),
    ("imap_codice_l1a_lo-pha_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-pha_20241110_v999.cdf"),
    ("imap_codice_l1a_lo-sw-angular_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-sw-angular_20241110_v999.cdf"),
    ("imap_codice_l1a_lo-sw-priority_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-sw-priority_20241110_v999.cdf"),
    ("imap_codice_l1a_lo-sw-species_20241110_v999.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-sw-species_20241110_v999.cdf"),

    ("imap_codice_l1a_hi-counters-aggregated_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hi-counters-aggregated_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_lo-counters-singles_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-counters-singles_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_hi-counters-singles_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hi-counters-singles_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_lo-ialirt_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-ialirt_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_hi-ialirt_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hi-ialirt_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_lo-nsw-angular_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-nsw-angular_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_hi-omni_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hi-omni_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_lo-nsw-priority_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-nsw-priority_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_hi-pha_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hi-pha_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_lo-nsw-species_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-nsw-species_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_hi-priorities_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hi-priorities_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_lo-pha_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-pha_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_hi-sectored_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hi-sectored_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_lo-sw-angular_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-sw-angular_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_hskp_20241110193622_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_hskp_20241110193622_v0.0.0.cdf"),
    ("imap_codice_l1a_lo-sw-priority_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-sw-priority_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_lo-counters-aggregated_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-counters-aggregated_20241110193700_v0.0.0.cdf"),
    ("imap_codice_l1a_lo-sw-species_20241110193700_v0.0.0.cdf", imap_module_directory / "tests" / "codice" / "data" / "imap_codice_l1a_lo-sw-species_20241110193700_v0.0.0.cdf"),

    # Hi
    ("imap_hi_l1a_45sensor-de_20250415_v999.cdf", imap_module_directory / "tests" / "hi" / "data" / "l1" / "imap_hi_l1a_45sensor-de_20250415_v999.cdf"),
    ("imap_hi_l1b_45sensor-de_20250415_v999.cdf", imap_module_directory / "tests" / "hi" / "data" / "l1" / "imap_hi_l1b_45sensor-de_20250415_v999.cdf"),
    ("imap_hi_l1c_45sensor-pset_20250415_v999.cdf", imap_module_directory / "tests" / "hi" / "data" / "l1" / "imap_hi_l1c_45sensor-pset_20250415_v999.cdf"),

    # I-ALiRT
    ("apid_478.bin", imap_module_directory / "tests" / "ialirt" / "data" / "l0" / "apid_478.bin"),

    # IDEX
    ("idex_l1a_validation_file.h5", imap_module_directory / "tests" / "idex" / "test_data" / "idex_l1a_validation_file.h5"),
    ("idex_l1b_validation_file.h5", imap_module_directory / "tests" / "idex" / "test_data" / "idex_l1b_validation_file.h5"),
    ("IMAP-Ultra45_r1_L1_V0_shortened.csv", imap_module_directory / "tests" / "ultra" / "data" / "l1" / "IMAP-Ultra45_r1_L1_V0_shortened.csv"),

    # Ultra
    ("imap_ultra_l0_raw_20260924_v001.pkts", imap_module_directory / "tests" / "ultra" / "data" / "l0" / "imap_ultra_l0_raw_20260924_v001.pkts"),
    ("imap_ultra_l1b_45sensor-de_20240207_v999.cdf", imap_module_directory / "tests" / "ultra" / "data" / "l1" / "imap_ultra_l1b_45sensor-de_20240207_v999.cdf"),
    ("ultra-90_raw_event_data_shortened.csv", imap_module_directory / "tests" / "ultra" / "data" / "l1" / "ultra-90_raw_event_data_shortened.csv"),
    ("imap_ultra_l1c-90sensor-efficiencies_20250101_v000.csv", imap_module_directory / "tests" / "ultra" / "data" / "l1" / "imap_ultra_l1c-90sensor-efficiencies_20250101_v000.csv"),
    ("imap_ultra_l1c-90sensor-gf_20250101_v000.csv", imap_module_directory / "tests" / "ultra" / "data" / "l1" / "imap_ultra_l1c-90sensor-gf_20250101_v000.csv"),
    ("imap_ultra_l1c-90sensor-dps-exposure_20250101_v000.csv", imap_module_directory / "tests" / "ultra" / "data" / "l1" / "imap_ultra_l1c-90sensor-dps-exposure_20250101_v000.csv"),
    ("imap_ultra_l1b-45sensor-logistic-interpolation_20250101_v000.csv", imap_module_directory / "tests" / "ultra" / "data" / "l1" / "imap_ultra_l1b-45sensor-logistic-interpolation_20250101_v000.csv")
]  # fmt: skip
