import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.ena_maps import ena_maps

@pytest.fixture
def pset():
    dataset = load_cdf(
        imap_module_directory / "tests/lo/test_cdfs/imap_lo_l1c_pset_20250415_v001.cdf"
    )
    return dataset

@pytest.mark.external_kernel
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_hflux_map(pset):


    fields = ["h_counts", "exposure_time"]
    for field in fields:
        pset[field] = pset[field].transpose("epoch", "dim2", "dim0", "dim1")

    print("PSET", pset.data_vars)

    ena_maps.RectangularSkyMap()