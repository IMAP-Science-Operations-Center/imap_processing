KPL/MK

Meta kernel that loads the arecibo configuration kernels into the sarbeta program.

\begindata

   PATH_SYMBOLS += ( 'GENERIC' )
   PATH_VALUES  += ( '/project/mini-rf/poc/LRO/data/storage/ancillary/spice/frames' )

   PATH_SYMBOLS += ( 'LRO' )
   PATH_VALUES  += ( '/project/mini-rf/poc/LRO/data/storage/ancillary/spice/frames' )

   KERNELS_TO_LOAD += ( '$GENERIC/arecibo.bsp'
                        '$GENERIC/arecibo.tf'
                        '$GENERIC/arecibo.ti'   )

   KERNELS_TO_LOAD += ( '$GENERIC/earthstns_itrf93_050714.bsp'
                        '$GENERIC/earth_topo_050714.tf'
                        '$GENERIC/dss13.ti'     )

   KERNELS_TO_LOAD += ( '$GENERIC/moon_pa_de421_1900-2050.bpc'
                        '$GENERIC/pck00010.tpc'
                        '$GENERIC/moon_080317.tf'
                        '$GENERIC/moon_assoc_me.tf'
                        '$GENERIC/earth_assoc_itrf93.tf'
                        '$GENERIC/de421.bsp'
                        '$GENERIC/naif0012.tls' )

   KERNELS_TO_LOAD += ( '$LRO/lro_frames_2010214_v01.tf'
                        '$LRO/lro_sandia.tf' )

\begintext
