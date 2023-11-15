KPL/MK

Meta kernel that loads the arecibo configuration kernels into the sarbeta program.

\begindata

   PATH_SYMBOLS += ( 'GENERIC' )
   PATH_VALUES  += ( '/project/mini-rf/poc/LRO/data/storage/ancillary/spice/frames' )

   PATH_SYMBOLS += ( 'IMAP' )
   PATH_VALUES  += ( '/project/sis/users/duttont1/software/imap/lib' )



   KERNELS_TO_LOAD += ( '$GENERIC/earthstns_itrf93_050714.bsp'
                        '$GENERIC/earth_topo_050714.tf' )

   KERNELS_TO_LOAD += ( '$GENERIC/moon_pa_de421_1900-2050.bpc'
                        '$GENERIC/pck00010.tpc'
                        '$GENERIC/moon_080317.tf'
                        '$GENERIC/moon_assoc_me.tf'
                        '$GENERIC/earth_assoc_itrf93.tf'
                        '$GENERIC/naif0012.tls' )

   KERNELS_TO_LOAD += ( '$IMAP/imap_frames_demo_0000.tf'
                        '$IMAP/imap_ultra_instrument_demo.ti'
                        '$IMAP/de440.bsp' )


\begintext
