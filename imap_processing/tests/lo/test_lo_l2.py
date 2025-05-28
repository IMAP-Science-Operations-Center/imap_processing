import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from imap_processing.lo.l2 import lo_l2

# @pytest.fixture
# def pset():
#     dataset = load_cdf(
#         imap_module_directory / "tests/lo/test_cdfs/imap_lo_l1c_pset_20250415_v001.cdf"
#     )
#     return dataset


@pytest.fixture
def pset():
    h_counts = np.zeros((1, 3600, 40, 7))
    h_counts[:, :, 0:10, :] = 1

    exposure_time = np.ones((1, 3600, 40, 7))

    dataset = xr.Dataset(
        {
            "h_counts": (("epoch", "longitude", "latitude", "energy"), h_counts),
            "exposure_time": (
                ("epoch", "longitude", "latitude", "energy"),
                exposure_time,
            ),
        },
        coords={
            "epoch": [8.1794907049e17],
            "longitude": [i for i in range(3600)],
            "latitude": [i for i in range(40)],
            "energy": [i for i in range(1, 8)],
        },
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
    fig = plt.figure()
    hflux_map[0]["h_flux"][0, 0].fillna(0).plot(
        x="longitude",
        y="latitude",
    )
    fig.savefig("/Users/seho5886/Desktop/lo_rect_map_h_flux.png")
    print(hflux_map)
