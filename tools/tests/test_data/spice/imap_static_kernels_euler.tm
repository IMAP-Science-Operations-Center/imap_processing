KPL/MK

Meta kernel that loads the IMAP configuration kernels.

\begindata

   PATH_SYMBOLS += ( 'IMAP' )
   PATH_VALUES  += ( '.' )


   KERNELS_TO_LOAD += ( '$IMAP/naif0012.tls'
                        '$IMAP/imap_frames_demo_euler.tf'
                        '$IMAP/imap_ultra_instrument_demo.ti'
                        '$IMAP/de440.bsp' )


\begintext
