import requests

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

# ruff: noqa
VALIDATION_DATA = [
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_hi-ialirt_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_lo-ialirt_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_hskp_20241110193622_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_lo-counters-aggregated_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_lo-counters-singles_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_lo-sw-priority_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_lo-nsw-priority_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_lo-sw-species_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_lo-nsw-species_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_lo-sw-angular_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_lo-nsw-angular_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_hi-counters-aggregated_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_hi-counters-singles_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_hi-omni_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_hi-sectored_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_hi-priority_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_lo-pha_20241110193700_v0.0.0.cdf",
    TEST_DATA_PATH / "validation" / "imap_codice_l1a_hi-pha_20241110193700_v0.0.0.cdf",
]  # fmt: skip

# Download CoDICE L0 test data and write it to the appropriate directory
response = requests.get(
    "https://api.dev.imap-mission.com/download/test_data/imap_codice_l0_raw_20241110_v001.pkts"
)
if response.status_code == 200:
    with open(TEST_L0_FILE, "wb") as file:
        file.write(response.content)
    print(f"Downloaded file: {TEST_L0_FILE}")
else:
    print(f"Failed to download file: {response.status_code}")
