from imap_processing import imap_module_directory

TEST_L0_DATA_PATH = imap_module_directory / "tests" / "codice" / "data"
TEST_L0_FILE = TEST_L0_DATA_PATH / "imap_codice_l0_raw_20241110_v001.pkts"
TEST_L1A_DATA_PATH = (
    imap_module_directory
    / "codice"
    / "data"
    / "imap"
    / "codice"
    / "l1a"
    / "2024"
    / "11"
)


TEST_L1A_FILES = [
    TEST_L1A_DATA_PATH / "imap_codice_l1a_hskp_20241110_v001.cdf",
    TEST_L1A_DATA_PATH / "imap_codice_l1a_hi-counters-aggregated_20241110_v001.cdf",
    TEST_L1A_DATA_PATH / "imap_codice_l1a_hi-counters-singles_20241110_v001.cdf",
    TEST_L1A_DATA_PATH / "imap_codice_l1a_hi-omni_20241110_v001.cdf",
    TEST_L1A_DATA_PATH / "imap_codice_l1a_hi-priority_20241110_v001.cdf",
    TEST_L1A_DATA_PATH / "imap_codice_l1a_hi-sectored_20241110_v001.cdf",
    TEST_L1A_DATA_PATH / "imap_codice_l1a_lo-counters-aggregated_20241110_v001.cdf",
    TEST_L1A_DATA_PATH / "imap_codice_l1a_lo-counters-singles_20241110_v001.cdf",
    TEST_L1A_DATA_PATH / "imap_codice_l1a_lo-sw-angular_20241110_v001.cdf",
    TEST_L1A_DATA_PATH / "imap_codice_l1a_lo-nsw-angular_20241110_v001.cdf",
    TEST_L1A_DATA_PATH / "imap_codice_l1a_lo-sw-priority_20241110_v001.cdf",
    TEST_L1A_DATA_PATH / "imap_codice_l1a_lo-nsw-priority_20241110_v001.cdf",
    TEST_L1A_DATA_PATH / "imap_codice_l1a_lo-sw-species_20241110_v001.cdf",
    TEST_L1A_DATA_PATH / "imap_codice_l1a_lo-nsw-species_20241110_v001.cdf",
]

# ruff: noqa
VALIDATION_DATA = [
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_hi-ialirt_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_lo-ialirt_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_hskp_20241110193622_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_lo-counters-aggregated_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_lo-counters-singles_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_lo-sw-priority_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_lo-nsw-priority_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_lo-sw-species_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_lo-nsw-species_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_lo-sw-angular_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_lo-nsw-angular_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_hi-counters-aggregated_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_hi-counters-singles_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_hi-omni_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_hi-sectored_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_hi-priorities_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_lo-pha_20241110193700_v0.0.0.cdf",
    TEST_L0_DATA_PATH / "validation" / "imap_codice_l1a_hi-pha_20241110193700_v0.0.0.cdf",
]  # fmt: skip
