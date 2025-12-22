"""Contains the source and destination paths of external test data.

This module defines the source of external test data, as stored in S3, and the
destination of where that test data should be downloaded to during GitHub
Actions builds. This is defined in a tuple (i.e. ``(<source>, <destination>)``),
where ``<destination>`` is relative to ``imap_module_directory/tests/``.
"""

# ruff: noqa: E501
from imap_processing.tests.codice.conftest import (
    VALIDATION_FILE_DATE,
    VALIDATION_FILE_VERSION,
)

EXTERNAL_TEST_DATA = [

    # CoDICE
    # L0 data
    ("imap_codice_l0_lo-sw-species_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_lo-nsw-species_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_lo-sw-angular_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_lo-nsw-angular_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_lo-nsw-priority_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_lo-sw-priority_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_lo-counters-aggregated_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_lo-counters-singles_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_lo-ialirt_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_lo-direct-events_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_hi-ialirt_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_hi-counters-aggregated_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_hi-counters-singles_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_hi-omni_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_hi-sectored_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_hi-priority_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_l0_hi-direct-events_20250814_v001.pkts", "codice/data/l1a_input/"),
    ("imap_codice_hskp_20250814_v001.pkts", "codice/data/l1a_input/"),

    # L1A LUT
    ("imap_codice_l1a-sci-lut_20251007_v004.json", "codice/data/l1a_lut/"),

    # L1A validation data
    (f"imap_codice_l1a_hi-counters-aggregated_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_hi-counters-singles_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_hi-direct-events_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_hi-ialirt_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_hi-omni_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_hi-sectored_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_hi-priority_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_hi-sectored_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_lo-counters-aggregated_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_lo-counters-singles_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_lo-direct-events_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_lo-ialirt_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_lo-nsw-priority_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_lo-nsw-angular_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_lo-sw-angular_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_lo-sw-priority_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_lo-nsw-species_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    (f"imap_codice_l1a_lo-sw-species_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1a_validation"),
    # L1B Input data is same as L1A validation data

    # L1B validation data
    (f"imap_codice_l1b_hi-counters-aggregated_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_hi-counters-singles_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_hi-ialirt_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_hi-omni_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_hi-priority_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_hi-sectored_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_lo-counters-aggregated_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_lo-counters-singles_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_lo-ialirt_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_lo-nsw-angular_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_lo-nsw-priority_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_lo-sw-angular_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_lo-sw-priority_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_lo-nsw-species_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_lo-sw-species_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_lo-nsw-angular_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    (f"imap_codice_l1b_lo-sw-angular_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l1b_validation"),
    # L2 LUT input data
    ("imap_codice_l2-hi-omni-efficiency_20251008_v001.csv", "codice/data/l2_lut/"),
    ("imap_codice_l2-hi-sectored-efficiency_20251008_v001.csv", "codice/data/l2_lut/"),
    ("imap_codice_l2-hi-ialirt-efficiency_20251008_v001.csv", "codice/data/l2_lut/"),
    ("imap_codice_l2-lo-gfactor_20251008_v001.csv", "codice/data/l2_lut/"),
    ("imap_codice_l2-lo-efficiency_20251008_v001.csv", "codice/data/l2_lut/"),
    ("imap_codice_l2-hi-tof-table_20250101_v001.csv", "codice/data/l2_lut/"),
    ("imap_codice_l2-hi-energy-table_20250101_v001.csv", "codice/data/l2_lut/"),
    ("imap_codice_l2-lo-onboard-energy-bins_20250101_v001.csv", "codice/data/l2_lut/"),
    ("imap_codice_l2-lo-onboard-energy-table_20250101_v001.csv", "codice/data/l2_lut/"),
    ("imap_codice_l2-lo-onboard-mpq-cal_20250101_v001.csv", "codice/data/l2_lut/"),

    # L2 Validation data
    (f"imap_codice_l2_hi-omni_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l2_validation/"),
    (f"imap_codice_l2_hi-sectored_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l2_validation/"),
    (f"imap_codice_l2_lo-nsw-angular_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l2_validation/"),
    (f"imap_codice_l2_lo-sw-angular_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l2_validation/"),
    (f"imap_codice_l2_lo-nsw-species_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l2_validation/"),
    (f"imap_codice_l2_lo-sw-species_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l2_validation/"),
    (f"imap_codice_l2_lo-direct-events_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l2_validation/"),
    (f"imap_codice_l2_hi-direct-events_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l2_validation/"),
    (f"imap_codice_l2_hi-ialirt_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l2_validation/"),
    (f"imap_codice_l2_lo-ialirt_{VALIDATION_FILE_DATE}_{VALIDATION_FILE_VERSION}.cdf", "codice/data/l2_validation/"),

    # Hi
    ("imap_hi_l1a_45sensor-de_20250415_v999.cdf", "hi/data/l1/"),
    ("imap_hi_l1b_45sensor-de_20250415_v999.cdf", "hi/data/l1/"),
    ("imap_hi_l1b_90sensor-hk_20241105-repoint00099_v001.cdf", "hi/data/l1/"),
    ("imap_hi_l1a_90sensor-de_20241105-repoint00099_v001.cdf", "hi/data/l1/"),
    ("imap_hi_l1c_45sensor-pset_20250415_v999.cdf", "hi/data/l1/"),

    # I-ALiRT
    ("apid_478.bin", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_40_25", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_41_26", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_42_27", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_43_28", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_44_29", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_45_30", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_46_31", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_47_32", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_48_33", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_49_34", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_50_35", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_51_36", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_52_37", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_53_38", "ialirt/data/l0/"),
    ("iois_1_packets_2025_284_05_54_39", "ialirt/data/l0/"),
    ("iois_1_packets_2025_344_05_57_56", "ialirt/data/l0/"),
    ("imap_recon_od005_20250925_20251014_v01.bsp", "spice/test_data/"),
    ("imap_2025_283_2025_284_001.ah.bc", "spice/test_data/"),

    # IDEX
    ("idex_l1a_validation_file.h5", "idex/test_data/"),
    ("idex_l1b_validation_file.h5", "idex/test_data/"),
    ("imap_idex_l2a-calibration-curve-yield-params_20250101_v001.csv", "idex/test_data/"),
    ("imap_idex_l2a-calibration-curve-t-rise_20250101_v001.csv", "idex/test_data/"),

    # Lo
    ("imap_lo_l1c_pset_20260101-repoint01261_v001.cdf", "lo/test_cdfs"),

    # Pointing Attitude Kernel
    ("imap_2025_338_2025_339_001.ah.bc", "spice/test_data/"),
    ("imap_2025_339_2025_339_001.ah.bc", "spice/test_data/"),
    ("imap_2025_339_2025_340_001.ah.bc", "spice/test_data/"),

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
    ("ultra45_l1b_raw_sc_ionextofhtimeimg_20240122_00_SDCStyle.csv", "ultra/data/l0/"),
    ("ultra45_l1b_raw_sc_ionextofhnrgimg_20240122_00_SDCStyle.csv", "ultra/data/l0/"),
    ("ultra45_l1b_raw_sc_enaextofhangimg_20240122_00_SDCStyle.csv", "ultra/data/l0/"),
    ("ultra45_l1b_raw_sc_enaphxtofhtimeimg_20240122_00_SDCStyle.csv", "ultra/data/l0/"),
    ("ultra45_l1b_raw_sc_enaphxtofhnrgimg_20240122_00_SDCStyle.csv", "ultra/data/l0/"),
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
    ("imap_ultra_l1c-90sensor-sc-pointing-theta-test_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1c-90sensor-sc-pointing-phi-test_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1c-90sensor-sc-pointing-index-test_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1c-90sensor-sc-pointing-bsf-test_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b-scattering-thresholds-per-energy_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1b_45sensor-de_20240207-repoint99999_v999.cdf","ultra/data/l1/"),
    ("imap_ultra_l1c-45sensor-nominal-for-lookup_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1c-45sensor-static-dead-times_20250101_v000.csv", "ultra/data/l1/"),
    ("imap_ultra_l1c-90sensor-static-dead-times_20250101_v000.csv", "ultra/data/l1/"),

    # l2
    ("imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv", "ultra/data/l2/"),

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
