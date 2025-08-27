"""Contains the source and destination paths of external test data.

This module defines the source of external test data, as stored in S3, and the
destination of where that test data should be downloaded to during GitHub
Actions builds. This is defined in a tuple (i.e. ``(<source>, <destination>)``),
where ``<destination>`` is relative to ``imap_module_directory/tests/``.
"""

# ruff: noqa: E501
EXTERNAL_TEST_DATA = [

    # CoDICE
    ("imap_codice_l0_raw_20241110_v001.pkts", "codice/data/"),
    ("imap_codice_l1a_hi-counters-aggregated_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_hi-counters-singles_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_hi-ialirt_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_hi-omni_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_hi-direct-events_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_hi-priority_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_hi-sectored_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_hskp_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_lo-counters-aggregated_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_lo-counters-singles_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_lo-ialirt_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_lo-nsw-angular_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_lo-nsw-priority_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_lo-nsw-species_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_lo-direct-events_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_lo-sw-angular_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_lo-sw-priority_20241110_v001.cdf", "codice/data/"),
    ("imap_codice_l1a_lo-sw-species_20241110_v001.cdf", "codice/data/"),

    ("imap_codice_l1a_hi-counters-aggregated_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_lo-counters-singles_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_hi-counters-singles_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_lo-ialirt_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_hi-ialirt_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_lo-nsw-angular_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_hi-omni_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_lo-nsw-priority_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_hi-direct-events_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_lo-nsw-species_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_hi-priorities_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_lo-direct-events_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_hi-sectored_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_lo-sw-angular_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_hskp_20241110193622_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_lo-sw-priority_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_lo-counters-aggregated_20241110193900_v0.0.2.cdf", "codice/data/validation/"),
    ("imap_codice_l1a_lo-sw-species_20241110193900_v0.0.2.cdf", "codice/data/validation/"),

    # Hi
    ("imap_hi_l1a_45sensor-de_20250415_v999.cdf", "hi/data/l1/"),
    ("imap_hi_l1b_45sensor-de_20250415_v999.cdf", "hi/data/l1/"),
    ("imap_hi_l1b_90sensor-hk_20241105-repoint00099_v001.cdf", "hi/data/l1/"),
    ("imap_hi_l1a_90sensor-de_20241105-repoint00099_v001.cdf", "hi/data/l1/"),
    ("imap_hi_l1c_45sensor-pset_20250415_v999.cdf", "hi/data/l1/"),

    # I-ALiRT
    ("apid_478.bin", "ialirt/data/l0/"),

    # IDEX
    ("idex_l1a_validation_file.h5", "idex/test_data/"),
    ("idex_l1b_validation_file.h5", "idex/test_data/"),
    ("imap_idex_l2a-calibration-curve-yield-params_20250101_v001.csv", "idex/test_data/"),
    ("imap_idex_l2a-calibration-curve-t-rise_20250101_v001.csv", "idex/test_data/"),

    # Ultra
    ("FM90_Startup_20230711T081655.CCSDS", "ultra/data/l0/"),
    ("IMAP-Ultra45_r1_L1_V0_shortened.csv", "ultra/data/l1/"),
    ("FM45_UltraFM45Extra_TV_Tests_2024-01-22T0930_20240122T093008.CCSDS", "ultra/data/l0/"),
    ("ultra45_raw_sc_rawnrgevnt_19840122_00.csv", "ultra/data/l0/"),
    ("ultra45_raw_sc_enaphxtofhnrgimg_FM45_UltraFM45Extra_TV_Tests_2024-01-22T0930_20240122T093008.csv",
     "ultra/data/l0/"),
    ("ultra45_raw_sc_imgpriority1evnt_FM45_UltraFM45Extra_TV_Tests_2024-01-22T0930_20240122T093008.csv", "ultra/data/l0/"),
    ("ultra45_raw_sc_imgpriority2evnt_FM45_UltraFM45Extra_TV_Tests_2024-01-22T0930_20240122T093008.csv", "ultra/data/l0/"),
    ("ultra45_raw_sc_imgpriority3evnt_FM45_UltraFM45Extra_TV_Tests_2024-01-22T0930_20240122T093008.csv", "ultra/data/l0/"),
    ("ultra45_raw_sc_imgpriority4evnt_FM45_UltraFM45Extra_TV_Tests_2024-01-22T0930_20240122T093008.csv", "ultra/data/l0/"),
    ("ultra45_raw_sc_ultranrgrates_FM45_UltraFM45_Functional_"
     "2024-01-22T0105_20240122T010548.csv", "ultra/data/l0/"),
    ("ultra45_raw_sc_ionexhtimeimg_FM45_UltraFM45Extra_TV_Tests_2024-01-22T0930_20240122T093008.csv", "ultra/data/l0/"),
    ("ultra45_raw_sc_ultraenaphxtofhtimeresimg_FM45_UltraFM45Extra_TV_Tests_2024-01-22T0930_20240122T093008.csv", "ultra/data/l0/"),
    ("ultra45_raw_sc_enaextofhangimg_FM45_UltraFM45Extra_TV_Tests_2024-01-22T0930_20240122T093008.csv", "ultra/data/l0/"),
    ("ultra45_raw_sc_ionextofhnrgimg_FM45_UltraFM45Extra_TV_Tests_2024-01-22T0930_20240122T093008.csv", "ultra/data/l0/"),
    ("imap_ultra_l0_raw_20260924_v001.pkts", "ultra/data/l0/"),
    ("imap_ultra_l1b_45sensor-de_20240207_v999.cdf", "ultra/data/l1/"),
    ("ultra-90_raw_event_data_shortened.csv", "ultra/data/l1/"),
    ("imap_ultra_l1c-90sensor-efficiencies_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1c-90sensor-gf_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1c-90sensor-dps-exposure_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-45sensor-logistic-interpolation_20250101_v000.csv",
     "ultra/data/l1/"),
    ("imap_ultra_l1b-sensor-gf-blades_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-sensor-gf-noblades_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-45sensor-leftslit-lookup_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-45sensor-rightslit-lookup_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-45sensor-tdc-norm-lookup_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-45sensor-back-pos-lookup_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-egynorm-lookup_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-yadjust-lookup_20250101_v001.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-45sensor-sptpphcorr_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-45sensor-spbtphcorr_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-90sensor-sptpphcorr_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-90sensor-spbtphcorr_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-90sensor-scattering-calibration-data_20250101_v000.csv", "ultra/data/l1/"),
    ("ultra45_raw_sc_ultrarawimg_withFSWccs_FM45_40P_Phi28p5_BeamCal_LinearScan_"
     "phi2850_theta-000_20240207T102740_revised20250724.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-45sensor-tofxeflat_20250101_v000.pgm", "ultra/data/l1/"),
    ("imap_ultra_l1b-45sensor-tofxemedium_20250101_v000.pgm", "ultra/data/l1/"),
    ("imap_ultra_l1b-45sensor-tofxesteep_20250101_v000.pgm", "ultra/data/l1/"),
    ("imap_ultra_l1b-90sensor-tofxeflat_20250101_v000.pgm", "ultra/data/l1/"),
    ("imap_ultra_l1b-90sensor-tofxemedium_20250101_v000.pgm", "ultra/data/l1/"),
    ("imap_ultra_l1b-90sensor-tofxesteep_20250101_v000.pgm", "ultra/data/l1/"),
    ("imap_ultra_l1b-tofxph_20250101_v000.pgm","ultra/data/l1/"),
    ("imap_ultra_l1c-90sensor-sc-pointing-theta-n32-test_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1c-90sensor-sc-pointing-phi-n32-test_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1c-90sensor-sc-pointing-index-n32-test_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1c-90sensor-sc-pointing-bsf-n32-test_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-scattering-thresholds-per-energy_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b_45sensor-de_20240207-repoint99999_v999.cdf","ultra/data/l1/"),
    ("imap_ultra_l1c-45sensor-nominal-for-lookup_20250101_v000.csv", "ultra/data/l1/"),

    # MAG
    ("mag-l1b-l1c-t013-magi-burst-in.csv",
     "mag/validation/L1c/T013/",
     ),
    (
        "mag-l1b-l1c-t013-mago-burst-in.csv",
        "mag/validation/L1c/T013/",
    ),
    (
        "mag-l1b-l1c-t014-mago-burst-in.csv",
        "mag/validation/L1c/T014/",
    ),
    (
        "mag-l1b-l1c-t014-magi-burst-in.csv",
        "mag/validation/L1c/T014/",
    ),
    (
        "mag-l1b-l1c-t015-mago-burst-in.csv",
        "mag/validation/L1c/T015/",
    ),
    (
        "mag-l1b-l1c-t016-mago-burst-in.csv",
        "mag/validation/L1c/T016/",
    ),
    (
        "imap_mag_l2_burst_20250506_v007.csv",
        "mag/validation/L2/T021/",
    ),
    (
        "mag-l1bc-l2-t021-mago-burst-in.csv",
        "mag/validation/L2/T021/",
    ),
    (
        "imap_mag_l2_burst-offsets_20250506_v006.cdf",
        "mag/validation/L2/T021/",
    ),
    (
        "mag-l1bc-l2-t021-magi-burst-in.csv",
        "mag/validation/L2/T021/",
    ),

    # SWE
    ("swe_l0_unpacked-data_20240510_v001_VALIDATION_L1A.dat", "swe/l1_validation/"),
    ("swe_l0_unpacked-data_20240510_v001_VALIDATION_L1B_v3.dat", "swe/l1_validation/"),
    ("swe_l0_unpacked-data_20240510_v001_VALIDATION_L2_v0B_15.dat", "swe/l2_validation/"),
    ("swe_l0_unpacked-data_20240510_v001_VALIDATION_L2_bins_v1F_15.dat", "swe/l2_validation/"),
    ("swe_l0_unpacked-data_20240510_v001_VALIDATION_L2_bins_v0F_15.dat", "swe/l2_validation/"),
    ("swe_l0_unpacked-data_20240510_v001_VALIDATION_L2_bins_v1H_14_6.dat", "swe/l2_validation/"),
    ("swe_l0_unpacked-data_20240510_v001_VALIDATION_L2_bins_v0H_14_6.dat", "swe/l2_validation/"),
]  # fmt: skip
