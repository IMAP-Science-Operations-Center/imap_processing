from imap_processing import imap_module_directory

TEST_DATA_PATH = imap_module_directory / "tests" / "codice" / "data"

TEST_L0_FILE = TEST_DATA_PATH / "imap_codice_l0_raw_20241110_v001.pkts"

TEST_L1A_FILES = [
    TEST_DATA_PATH / "imap_codice_l1a_hskp_20100101_v001.pkts",
    TEST_DATA_PATH / "imap_codice_l1a_hi-counters-aggregated_20240429_v001.pkts",
    TEST_DATA_PATH / "imap_codice_l1a_hi-counters-singles_20240429_v001.pkts",
    TEST_DATA_PATH / "imap_codice_l1a_hi-omni_20240429_v001.pkts",
    TEST_DATA_PATH / "imap_codice_l1a_hi-sectored_20240429_v001.pkts",
    TEST_DATA_PATH / "imap_codice_l1a_lo-counters-aggregated_20240429_v001.pkts",
    TEST_DATA_PATH / "imap_codice_l1a_lo-counters-singles_20240429_v001.pkts",
    TEST_DATA_PATH / "imap_codice_l1a_lo-sw-angular_20240429_v001.pkts",
    TEST_DATA_PATH / "imap_codice_l1a_lo-nsw-angular_20240429_v001.pkts",
    TEST_DATA_PATH / "imap_codice_l1a_lo-sw-priority_20240429_v001.pkts",
    TEST_DATA_PATH / "imap_codice_l1a_lo-nsw-priority_20240429_v001.pkts",
    TEST_DATA_PATH / "imap_codice_l1a_lo-sw-species_20240429_v001.pkts",
    TEST_DATA_PATH / "imap_codice_l1a_lo-nsw-species_20240429_v001.pkts",
]

# Placeholder for validation data files
VALIDATION_DATA = [
    TEST_DATA_PATH / "validation_hskp.cdf",
    TEST_DATA_PATH / "validation_hi-counters-aggregated.cdf",
    TEST_DATA_PATH / "validation_hi-counters-singles.cdf",
    TEST_DATA_PATH / "validation_hi-omni.cdf",
    TEST_DATA_PATH / "validation_hi-sectored.cdf",
    TEST_DATA_PATH / "validation_hi-pha.cdf",
    TEST_DATA_PATH / "validation_lo-counters-aggregated.cdf",
    TEST_DATA_PATH / "validation_lo-counters-singles.cdf",
    TEST_DATA_PATH / "validation_lo-sw-angular.cdf",
    TEST_DATA_PATH / "validation_lo-nsw-angular.cdf",
    TEST_DATA_PATH / "validation_lo-sw-priority.cdf",
    TEST_DATA_PATH / "validation_lo-nsw-priority.cdf",
    TEST_DATA_PATH / "validation_lo-sw-species.cdf",
    TEST_DATA_PATH / "validation_lo-nsw-species.cdf",
    TEST_DATA_PATH / "validation_lo-pha.cdf",
]
