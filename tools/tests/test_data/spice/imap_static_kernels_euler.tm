KPL/MK

Meta kernel that loads the arecibo configuration kernels into the sarbeta program.

\begindata

   PATH_SYMBOLS += ( 'IMAP' )
   PATH_VALUES  += ( '/Users/lasa6858/Desktop/ultra/ultra_prototype_v1/kernels' )


   KERNELS_TO_LOAD += ( '$IMAP/naif0012.tls'
                        '$IMAP/imap_frames_demo_euler.tf' 
                        '$IMAP/imap_ultra_instrument_demo.ti'
                        '$IMAP/de440.bsp' )


\begintext
