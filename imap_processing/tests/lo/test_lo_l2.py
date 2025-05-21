import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.lo.l2 import lo_l2


@pytest.fixture
def pset():
    dataset = load_cdf(
        imap_module_directory / "tests/lo/test_cdfs/imap_lo_l1c_pset_20250415_v001.cdf"
    )
    return dataset


@pytest.mark.external_kernel
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_hflux_map(pset):
    # Arrange
    pset = {"imap_lo_l1c_pset": pset}

    # Act
    hflux_map = lo_l2.lo_l2(pset, [])

    hflux_map[0]["h_flux"].mean(dim=("epoch", "energy")).plot(
        x="longitude",
        y="latitude",
    )
    print(hflux_map)
