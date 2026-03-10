KPL/FK

Interstellar Mapping and Acceleration Probe Frames Kernel
========================================================================

   This frames kernel contains the current set of coordinate frame
   definitions for the Interstellar Mapping and Acceleration Probe
   (IMAP) spacecraft, structures, and science instruments.

   This kernel also contains NAIF ID/name mapping for the IMAP
   instruments.


Version and Date
========================================================================

   The TEXT_KERNEL_ID stores version information of loaded project text
   kernels. Each entry associated with the keyword is a string that
   consists of four parts: the kernel name, version, entry date, and
   type. For example, the frames kernel might have an entry as follows:


           TEXT_KERNEL_ID += 'IMAP_FRAMES V1.0.0 2024-XXXX-NN FK'
                                  |         |        |        |
                                  |         |        |        |
              KERNEL NAME <-------+         |        |        |
                                            |        |        V
                             VERSION <------+        |   KERNEL TYPE
                                                     |
                                                     V
                                                ENTRY DATE


   Interstellar Mapping and Acceleration Probe Frames Kernel Version:

   \begindata

   TEXT_KERNEL_ID += 'IMAP_FRAMES V1.0.0 2024-XXXX-NN FK'

   \begintext

   Version 1.0.0 -- XXXX NN, 2024 -- Douglas Rodgers
                                     Lillian Nguyen
                                     Nicholas Dutton

      Initial complete release. Frame/Body codes for thrusters redefined

   Version 0.0.1 -- July 9, 2021 -- Ian Wick Murphy

      Modifying dart_008.tf to add basic IMAP frame components. This
      includes IMAP, IMAP_THRUSTER, and CK/SCLK IDs. Also adding a place
      holder for the IMAP-Lo instrument with the ID -43001 and IMAP_LO
      name. Future work includes adding more detailed instrument frames,
      and reaching out to mechanical for an "official" IMAP_SPACECRAFT
      frame definition.


References
========================================================================

   1.  "Frames Required Reading"

   2.  "Kernel Pool Required Reading"

   3.  "C-Kernel Required Reading"

   4.  "7516-9067: IMAP Mechanical Interface Control Document",
       Johns Hopkins Applied Physics Laboratory

   5.  "7516-9050: IMAP Coordinate Frame & Technical Definitions Doc.",
       Johns Hopkins Applied Physics Laboratory

   6.  "7516-0011: IMAP Mechanical Interface Control Drawing",
       [EXPORT CONTROLLED], Johns Hopkins Applied Physics Laboratory

   7.  "7523-0008: IMAP ULTRA Mechanical Interface Control Drawing",
       [EXPORT CONTROLLED], Johns Hopkins Applied Physics Laboratory

   8.  "058991000: IMAP SWAPI Mechanical Interface Control Drawing",
       Princeton University Space Physics

   9.  "GLOWS-CBK-DWG-2020-08-25-019-v4.4: IMAP GLOWS Mechanical
       Interface Control Drawing", Centrum Badag Kosmicznych, Polska
       Akademia Nauks

   10. Responses from IMAP instrument teams on their base frame axis
       definitions, received in email.

   11. "Euler angles", Wikimedia Foundation, 2024-04-22,
       https://en.wikipedia.org/wiki/Euler_angles

   12. "7516-9059: IMAP-Lo to Spacecraft Interface Control Document",
       [EXPORT CONTROLLED], Johns Hopkins Applied Physics Laboratory

   13. "DRAFT Rev H: IMAP-Lo Mechanical Interface Control Drawing",
       [EXPORT CONTROLLED],  Univ. of New Hampshire Space Science Center

   14. McComas et al, "IMAP: A New NASA Mission",
       Space Sci Rev (2018) 214:116

   15. "IMAP-HI SENSOR HEAD Mechanical Interface Control Drawing",
       [EXPORT CONTROLLED],  Los Alamos National Laboratory

   16. "IMAP-MAG-SENSOR Drawing Rev 6", Imperial College London


Contact Information
========================================================================

   Douglas Rodgers,  JHU/APL,  Douglas.Rodgers@jhuapl.edu

   Lillian Nguyen,   JHU/APL,  Lillian.Nguyen@jhuapl.edu

   Nicholas Dutton,  JHU/APL,  Nicholas.Dutton@jhuapl.edu

   Ian Wick Murphy,  JHU/APL,  Ian.Murphy@jhuapl.edu


Implementation Notes
========================================================================

   This file is used by the SPICE system as follows: programs that make
   use of this frame kernel must `load' the kernel, normally during
   program initialization. Loading the kernel associates the data items
   with their names in a data structure called the `kernel pool'. The
   SPICELIB routine FURNSH loads a kernel into the pool as shown below:

      FORTRAN: (SPICELIB)

         CALL FURNSH ( frame_kernel_name )

      C: (CSPICE)

         furnsh_c ( frame_kernel_name );

      IDL: (ICY)

         cspice_furnsh, frame_kernel_name

      MATLAB: (MICE)

         cspice_furnsh ( frame_kernel_name )

   This file was created and may be updated with a text editor or word
   processor.


Viewing ASCII Artwork
========================================================================

   Artwork must be viewed in a text editor with monospaced font and
   compact single-spaced lines. The following give the proper aspect
   ratio:

   Andale Regular
   Menlo Regular
   Courier New Regular
   PT Mono Regular

   The common monospaced font (at the time of writing) Monaco Regular
   gives an aspect ratio that is too tall. Other fonts undoubtedly
   will render the diagrams properly or improperly.

   As a guide, the following axis will be square when measured from the
   bottom of the lower-most vertical line to the end of each axis.

      |
      |
      |
      |_______


IMAP NAIF ID Codes -- Definitions
========================================================================

   This section contains name to NAIF ID mappings for the IMAP mission.
   Once the contents of this file are loaded into the KERNEL POOL, these
   mappings become available within SPICE, making it possible to use
   names instead of ID code in high level SPICE routine calls.

   \begindata

      NAIF_BODY_NAME   += ( 'IMAP' )
      NAIF_BODY_CODE   += ( -43    )

      NAIF_BODY_NAME   += ( 'IMAP_SPACECRAFT' )
      NAIF_BODY_CODE   += ( -43000            )

      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_A1' )
      NAIF_BODY_CODE   += ( -43010             )

      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_A2' )
      NAIF_BODY_CODE   += ( -43011             )

      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_A3' )
      NAIF_BODY_CODE   += ( -43012             )

      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_A4' )
      NAIF_BODY_CODE   += ( -43013             )

      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R1' )
      NAIF_BODY_CODE   += ( -43020             )

      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R2' )
      NAIF_BODY_CODE   += ( -43021             )

      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R3' )
      NAIF_BODY_CODE   += ( -43022             )

      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R4' )
      NAIF_BODY_CODE   += ( -43023             )

      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R5' )
      NAIF_BODY_CODE   += ( -43024             )

      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R6' )
      NAIF_BODY_CODE   += ( -43025             )

      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R7' )
      NAIF_BODY_CODE   += ( -43026             )

      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R8' )
      NAIF_BODY_CODE   += ( -43027             )

      NAIF_BODY_NAME   += ( 'IMAP_SUN_SENSOR_PZ' )
      NAIF_BODY_CODE   += ( -43030               )

      NAIF_BODY_NAME   += ( 'IMAP_SUN_SENSOR_MZ' )
      NAIF_BODY_CODE   += ( -43031               )

      NAIF_BODY_NAME   += ( 'IMAP_STAR_TRACKER_PX' )
      NAIF_BODY_CODE   += ( -43040                 )

      NAIF_BODY_NAME   += ( 'IMAP_STAR_TRACKER_MX' )
      NAIF_BODY_CODE   += ( -43041                 )

      NAIF_BODY_NAME   += ( 'IMAP_LOW_GAIN_ANTENNA' )
      NAIF_BODY_CODE   += ( -43050                  )

      NAIF_BODY_NAME   += ( 'IMAP_MED_GAIN_ANTENNA' )
      NAIF_BODY_CODE   += ( -43051                  )

      NAIF_BODY_NAME   += ( 'IMAP_LO_BASE'       )
      NAIF_BODY_CODE   += ( -43100               )

      NAIF_BODY_NAME   += ( 'IMAP_LO'            )
      NAIF_BODY_CODE   += ( -43101               )

      NAIF_BODY_NAME   += ( 'IMAP_LO_STAR_SENSOR' )
      NAIF_BODY_CODE   += ( -43102               )

      NAIF_BODY_NAME   += ( 'IMAP_HI_45' )
      NAIF_BODY_CODE   += ( -43150       )

      NAIF_BODY_NAME   += ( 'IMAP_HI_90' )
      NAIF_BODY_CODE   += ( -43175       )

      NAIF_BODY_NAME   += ( 'IMAP_ULTRA_45' )
      NAIF_BODY_CODE   += ( -43200          )

      NAIF_BODY_NAME   += ( 'IMAP_ULTRA_90' )
      NAIF_BODY_CODE   += ( -43225          )

      NAIF_BODY_NAME   += ( 'IMAP_MAG_BOOM' )
      NAIF_BODY_CODE   += ( -43250          )

      NAIF_BODY_NAME   += ( 'IMAP_MAG_I' )
      NAIF_BODY_CODE   += ( -43251       )

      NAIF_BODY_NAME   += ( 'IMAP_MAG_O' )
      NAIF_BODY_CODE   += ( -43251       )

      NAIF_BODY_NAME   += ( 'IMAP_SWE' )
      NAIF_BODY_CODE   += ( -43300     )

      NAIF_BODY_NAME   += ( 'IMAP_SWE_DETECTOR_P63' )
      NAIF_BODY_CODE   += ( -43301                  )

      NAIF_BODY_NAME   += ( 'IMAP_SWE_DETECTOR_P42' )
      NAIF_BODY_CODE   += ( -43302                  )

      NAIF_BODY_NAME   += ( 'IMAP_SWE_DETECTOR_P21' )
      NAIF_BODY_CODE   += ( -43303                  )

      NAIF_BODY_NAME   += ( 'IMAP_SWE_DETECTOR_000' )
      NAIF_BODY_CODE   += ( -43304                  )

      NAIF_BODY_NAME   += ( 'IMAP_SWE_DETECTOR_M21' )
      NAIF_BODY_CODE   += ( -43305                  )

      NAIF_BODY_NAME   += ( 'IMAP_SWE_DETECTOR_M42' )
      NAIF_BODY_CODE   += ( -43306                  )

      NAIF_BODY_NAME   += ( 'IMAP_SWE_DETECTOR_M63' )
      NAIF_BODY_CODE   += ( -43307                  )

      NAIF_BODY_NAME   += ( 'IMAP_SWAPI' )
      NAIF_BODY_CODE   += ( -433510      )

      NAIF_BODY_NAME   += ( 'IMAP_SWAPI_APERTURE_L' )
      NAIF_BODY_CODE   += ( -43351                  )

      NAIF_BODY_NAME   += ( 'IMAP_SWAPI_APERTURE_R' )
      NAIF_BODY_CODE   += ( -43352                  )

      NAIF_BODY_NAME   += ( 'IMAP_SWAPI_SUNGLASSES' )
      NAIF_BODY_CODE   += ( -43353                  )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE' )
      NAIF_BODY_CODE   += ( -43400        )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_01' )
      NAIF_BODY_CODE   += ( -43401                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_02' )
      NAIF_BODY_CODE   += ( -43402                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_03' )
      NAIF_BODY_CODE   += ( -43403                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_04' )
      NAIF_BODY_CODE   += ( -43404                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_05' )
      NAIF_BODY_CODE   += ( -43405                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_06' )
      NAIF_BODY_CODE   += ( -43406                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_07' )
      NAIF_BODY_CODE   += ( -43407                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_08' )
      NAIF_BODY_CODE   += ( -43408                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_09' )
      NAIF_BODY_CODE   += ( -43409                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_10' )
      NAIF_BODY_CODE   += ( -43410                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_11' )
      NAIF_BODY_CODE   += ( -43411                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_12' )
      NAIF_BODY_CODE   += ( -43412                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_13' )
      NAIF_BODY_CODE   += ( -43413                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_14' )
      NAIF_BODY_CODE   += ( -43414                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_15' )
      NAIF_BODY_CODE   += ( -43415                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_16' )
      NAIF_BODY_CODE   += ( -43416                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_17' )
      NAIF_BODY_CODE   += ( -43417                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_18' )
      NAIF_BODY_CODE   += ( -43418                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_19' )
      NAIF_BODY_CODE   += ( -43419                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_20' )
      NAIF_BODY_CODE   += ( -43420                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_21' )
      NAIF_BODY_CODE   += ( -43421                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_22' )
      NAIF_BODY_CODE   += ( -43422                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_23' )
      NAIF_BODY_CODE   += ( -43423                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_LO_APERTURE_24' )
      NAIF_BODY_CODE   += ( -43424                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_HI_APERTURE_01' )
      NAIF_BODY_CODE   += ( -43425                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_HI_APERTURE_02' )
      NAIF_BODY_CODE   += ( -43426                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_HI_APERTURE_03' )
      NAIF_BODY_CODE   += ( -43427                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_HI_APERTURE_04' )
      NAIF_BODY_CODE   += ( -43428                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_HI_APERTURE_05' )
      NAIF_BODY_CODE   += ( -43429                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_HI_APERTURE_06' )
      NAIF_BODY_CODE   += ( -43430                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_HI_APERTURE_07' )
      NAIF_BODY_CODE   += ( -43431                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_HI_APERTURE_08' )
      NAIF_BODY_CODE   += ( -43432                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_HI_APERTURE_09' )
      NAIF_BODY_CODE   += ( -43433                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_HI_APERTURE_10' )
      NAIF_BODY_CODE   += ( -43434                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_HI_APERTURE_11' )
      NAIF_BODY_CODE   += ( -43435                       )

      NAIF_BODY_NAME   += ( 'IMAP_CODICE_HI_APERTURE_12' )
      NAIF_BODY_CODE   += ( -43436                       )

      NAIF_BODY_NAME   += ( 'IMAP_HIT' )
      NAIF_BODY_CODE   += ( -43500     )

      NAIF_BODY_NAME   += ( 'IMAP_HIT_L1_APERTURE_01' )
      NAIF_BODY_CODE   += ( -43501                    )

      NAIF_BODY_NAME   += ( 'IMAP_HIT_L1_APERTURE_02' )
      NAIF_BODY_CODE   += ( -43502                    )

      NAIF_BODY_NAME   += ( 'IMAP_HIT_L1_APERTURE_03' )
      NAIF_BODY_CODE   += ( -43503                    )

      NAIF_BODY_NAME   += ( 'IMAP_HIT_L1_APERTURE_04' )
      NAIF_BODY_CODE   += ( -43504                    )

      NAIF_BODY_NAME   += ( 'IMAP_HIT_L1_APERTURE_05' )
      NAIF_BODY_CODE   += ( -43505                    )

      NAIF_BODY_NAME   += ( 'IMAP_HIT_L1_APERTURE_06' )
      NAIF_BODY_CODE   += ( -43506                    )

      NAIF_BODY_NAME   += ( 'IMAP_HIT_L1_APERTURE_07' )
      NAIF_BODY_CODE   += ( -43507                    )

      NAIF_BODY_NAME   += ( 'IMAP_HIT_L1_APERTURE_08' )
      NAIF_BODY_CODE   += ( -43508                    )

      NAIF_BODY_NAME   += ( 'IMAP_HIT_L1_APERTURE_09' )
      NAIF_BODY_CODE   += ( -43509                    )

      NAIF_BODY_NAME   += ( 'IMAP_HIT_L1_APERTURE_10' )
      NAIF_BODY_CODE   += ( -43510                    )

      NAIF_BODY_NAME   += ( 'IMAP_IDEX' )
      NAIF_BODY_CODE   += ( -43700      )

      NAIF_BODY_NAME   += ( 'IMAP_IDEX_DETECTOR' )
      NAIF_BODY_CODE   += ( -43701               )

      NAIF_BODY_NAME   += ( 'IMAP_IDEX_FULL_SCIENCE' )
      NAIF_BODY_CODE   += ( -43702                   )


      NAIF_BODY_NAME   += ( 'IMAP_GLOWS' )
      NAIF_BODY_CODE   += ( -43751       )

   \begintext

Removed by Tim Plummer due to missing frame definition
      NAIF_BODY_NAME   += ( 'IMAP_GLOWS_BASE' )
      NAIF_BODY_CODE   += ( -43750            )

IMAP NAIF ID Codes -- Definitions
========================================================================

   The ID codes -43900 to -43999 have been reserved for the IMAP dynamic
   frames kernel and are not utilized in this file.

   The following frames are defined in this kernel file:

      Frame Name                    Relative To        Type      NAIF ID
      ==========================    ===============    =======   =======

      Spacecraft (000-099)
      --------------------------
      IMAP_SPACECRAFT               J2000              CK        -43000
      IMAP_THRUSTER_A1              IMAP_SPACECRAFT    FIXED     -43010
      IMAP_THRUSTER_A2              IMAP_SPACECRAFT    FIXED     -43011
      IMAP_THRUSTER_A3              IMAP_SPACECRAFT    FIXED     -43012
      IMAP_THRUSTER_A4              IMAP_SPACECRAFT    FIXED     -43013
      IMAP_THRUSTER_R1              IMAP_SPACECRAFT    FIXED     -43020
      IMAP_THRUSTER_R2              IMAP_SPACECRAFT    FIXED     -43021
      IMAP_THRUSTER_R3              IMAP_SPACECRAFT    FIXED     -43022
      IMAP_THRUSTER_R4              IMAP_SPACECRAFT    FIXED     -43023
      IMAP_THRUSTER_R5              IMAP_SPACECRAFT    FIXED     -43024
      IMAP_THRUSTER_R6              IMAP_SPACECRAFT    FIXED     -43025
      IMAP_THRUSTER_R7              IMAP_SPACECRAFT    FIXED     -43026
      IMAP_THRUSTER_R8              IMAP_SPACECRAFT    FIXED     -43027
      IMAP_SUN_SENSOR_PZ            IMAP_SPACECRAFT    FIXED     -43030
      IMAP_SUN_SENSOR_MZ            IMAP_SPACECRAFT    FIXED     -43031
      IMAP_STAR_TRACKER_PX          IMAP_SPACECRAFT    FIXED     -43040
      IMAP_STAR_TRACKER_MX          IMAP_SPACECRAFT    FIXED     -43041
      IMAP_LOW_GAIN_ANTENNA         IMAP_SPACECRAFT    FIXED     -43050
      IMAP_MED_GAIN_ANTENNA         IMAP_SPACECRAFT    FIXED     -43051

      IMAP-Lo (100-149)
      --------------------------
      IMAP_LO_BASE                  IMAP_SPACECRAFT    FIXED     -43100
      IMAP_LO                       IMAP_LO_BASE       CK        -43101
      IMAP_LO_STAR_SENSOR           IMAP_LO            FIXED     -43102

      IMAP-Hi (150-199)
      --------------------------
      IMAP_HI_45                    IMAP_SPACECRAFT    FIXED     -43150
      IMAP_HI_90                    IMAP_SPACECRAFT    FIXED     -43151

      IMAP-Ultra (200-249)
      --------------------------
      IMAP_ULTRA_45                 IMAP_SPACECRAFT    FIXED     -43200
      IMAP_ULTRA_90                 IMAP_SPACECRAFT    FIXED     -43201

      MAG (250-299)
      --------------------------
      IMAP_MAG_BOOM                 IMAP_SPACECRAFT    FIXED     -43250
      IMAP_MAG_I                    IMAP_MAG_BOOM      FIXED     -43251
      IMAP_MAG_O                    IMAP_MAG_BOOM      FIXED     -43252

      SWE (300-349)
      --------------------------
      IMAP_SWE                      IMAP_SPACECRAFT    FIXED     -43300
      IMAP_SWE_DETECTOR_P63         IMAP_SWE           FIXED     -43301
      IMAP_SWE_DETECTOR_P42         IMAP_SWE           FIXED     -43302
      IMAP_SWE_DETECTOR_P21         IMAP_SWE           FIXED     -43303
      IMAP_SWE_DETECTOR_000         IMAP_SWE           FIXED     -43304
      IMAP_SWE_DETECTOR_M21         IMAP_SWE           FIXED     -43305
      IMAP_SWE_DETECTOR_M42         IMAP_SWE           FIXED     -43306
      IMAP_SWE_DETECTOR_M63         IMAP_SWE           FIXED     -43307

      SWAPI (350-399)
      --------------------------
      IMAP_SWAPI                    IMAP_SPACECRAFT    FIXED     -43350
      IMAP_SWAPI_APERTURE_L         IMAP_SWAPI         FIXED     -43351
      IMAP_SWAPI_APERTURE_R         IMAP_SWAPI         FIXED     -43352
      IMAP_SWAPI_SUNGLASSES         IMAP_SWAPI         FIXED     -43353

      CODICE (400-499)
      --------------------------
      IMAP_CODICE                   IMAP_SPACECRAFT    FIXED     -43400
      IMAP_CODICE_LO_APERTURE_01    IMAP_CODICE        FIXED     -43401
      IMAP_CODICE_LO_APERTURE_02    IMAP_CODICE        FIXED     -43402
      IMAP_CODICE_LO_APERTURE_03    IMAP_CODICE        FIXED     -43403
      IMAP_CODICE_LO_APERTURE_04    IMAP_CODICE        FIXED     -43404
      IMAP_CODICE_LO_APERTURE_05    IMAP_CODICE        FIXED     -43405
      IMAP_CODICE_LO_APERTURE_06    IMAP_CODICE        FIXED     -43406
      IMAP_CODICE_LO_APERTURE_07    IMAP_CODICE        FIXED     -43407
      IMAP_CODICE_LO_APERTURE_08    IMAP_CODICE        FIXED     -43408
      IMAP_CODICE_LO_APERTURE_09    IMAP_CODICE        FIXED     -43409
      IMAP_CODICE_LO_APERTURE_10    IMAP_CODICE        FIXED     -43410
      IMAP_CODICE_LO_APERTURE_11    IMAP_CODICE        FIXED     -43411
      IMAP_CODICE_LO_APERTURE_12    IMAP_CODICE        FIXED     -43412
      IMAP_CODICE_LO_APERTURE_13    IMAP_CODICE        FIXED     -43413
      IMAP_CODICE_LO_APERTURE_14    IMAP_CODICE        FIXED     -43414
      IMAP_CODICE_LO_APERTURE_15    IMAP_CODICE        FIXED     -43415
      IMAP_CODICE_LO_APERTURE_16    IMAP_CODICE        FIXED     -43416
      IMAP_CODICE_LO_APERTURE_17    IMAP_CODICE        FIXED     -43417
      IMAP_CODICE_LO_APERTURE_18    IMAP_CODICE        FIXED     -43418
      IMAP_CODICE_LO_APERTURE_19    IMAP_CODICE        FIXED     -43419
      IMAP_CODICE_LO_APERTURE_20    IMAP_CODICE        FIXED     -43420
      IMAP_CODICE_LO_APERTURE_21    IMAP_CODICE        FIXED     -43421
      IMAP_CODICE_LO_APERTURE_22    IMAP_CODICE        FIXED     -43422
      IMAP_CODICE_LO_APERTURE_23    IMAP_CODICE        FIXED     -43423
      IMAP_CODICE_LO_APERTURE_24    IMAP_CODICE        FIXED     -43424
      IMAP_CODICE_HI_APERTURE_01    IMAP_CODICE        FIXED     -43425
      IMAP_CODICE_HI_APERTURE_02    IMAP_CODICE        FIXED     -43426
      IMAP_CODICE_HI_APERTURE_03    IMAP_CODICE        FIXED     -43427
      IMAP_CODICE_HI_APERTURE_04    IMAP_CODICE        FIXED     -43428
      IMAP_CODICE_HI_APERTURE_05    IMAP_CODICE        FIXED     -43429
      IMAP_CODICE_HI_APERTURE_06    IMAP_CODICE        FIXED     -43430
      IMAP_CODICE_HI_APERTURE_07    IMAP_CODICE        FIXED     -43431
      IMAP_CODICE_HI_APERTURE_08    IMAP_CODICE        FIXED     -43432
      IMAP_CODICE_HI_APERTURE_09    IMAP_CODICE        FIXED     -43433
      IMAP_CODICE_HI_APERTURE_10    IMAP_CODICE        FIXED     -43434
      IMAP_CODICE_HI_APERTURE_11    IMAP_CODICE        FIXED     -43435
      IMAP_CODICE_HI_APERTURE_12    IMAP_CODICE        FIXED     -43436

      HIT (500-699)
      --------------------------
      IMAP_HIT                      IMAP_SPACECRAFT    FIXED     -43500
      IMAP_HIT_L1_APERTURE_01       IMAP_HIT           FIXED     -43501
      IMAP_HIT_L1_APERTURE_02       IMAP_HIT           FIXED     -43502
      IMAP_HIT_L1_APERTURE_03       IMAP_HIT           FIXED     -43503
      IMAP_HIT_L1_APERTURE_04       IMAP_HIT           FIXED     -43504
      IMAP_HIT_L1_APERTURE_05       IMAP_HIT           FIXED     -43505
      IMAP_HIT_L1_APERTURE_06       IMAP_HIT           FIXED     -43506
      IMAP_HIT_L1_APERTURE_07       IMAP_HIT           FIXED     -43507
      IMAP_HIT_L1_APERTURE_08       IMAP_HIT           FIXED     -43508
      IMAP_HIT_L1_APERTURE_09       IMAP_HIT           FIXED     -43509
      IMAP_HIT_L1_APERTURE_10       IMAP_HIT           FIXED     -43510

      IDEX (700-749)
      --------------------------
      IMAP_IDEX                     IMAP_SPACECRAFT    FIXED     -43700
      IMAP_IDEX_DETECTOR            IMAP_IDEX          FIXED     -43701
      IMAP_IDEX_FULL_SCIENCE        IMAP_IDEX          FIXED     -43702

      GLOWS (750-799)
      --------------------------
      IMAP_GLOWS_BASE               IMAP_SPACECRAFT    FIXED     -43750
      IMAP_GLOWS                    IMAP_GLOWS_BASE    FIXED     -43751


IMAP Frame Tree
========================================================================

   The diagram below illustrates the IMAP frame hierarchy:

   J2000
      |
      |<---ck
      |
      IMAP_SPACECRAFT
           |
           IMAP_THRUSTER_A1
           |
           |...
           |
           IMAP_THRUSTER_A4
           |
           IMAP_THRUSTER_R1
           |
           |...
           |
           IMAP_THRUSTER_R8
           |
           IMAP_SUN_SENSOR_PZ
           |
           IMAP_SUN_SENSOR_MZ
           |
           IMAP_STAR_TRACKER_PX
           |
           IMAP_STAR_TRACKER_MX
           |
           IMAP_LOW_GAIN_ANTENNA
           |
           IMAP_MED_GAIN_ANTENNA
           |
           IMAP_LO_BASE
           |    |
           |    |<---ck
           |    |
           |    IMAP_LO
           |         |
           |         IMAP_LO_STAR_SENSOR
           |
           IMAP_HI_45
           |
           IMAP_HI_90
           |
           IMAP_ULTRA_45
           |
           IMAP_ULTRA_90
           |
           IMAP_MAG_BOOM
           |    |
           |    IMAP_MAP_I
           |    |
           |    IMAP_MAP_O
           |
           IMAP_SWE
           |    |
           |    IMAP_SWE_DETECTOR_P63
           |    |
           |    IMAP_SWE_DETECTOR_P42
           |    |
           |    IMAP_SWE_DETECTOR_P21
           |    |
           |    IMAP_SWE_DETECTOR_000
           |    |
           |    IMAP_SWE_DETECTOR_M21
           |    |
           |    IMAP_SWE_DETECTOR_M42
           |    |
           |    IMAP_SWE_DETECTOR_M63
           |
           IMAP_SWAPI
           |    |
           |    IMAP_SWAPI_APERTURE_L
           |    |
           |    IMAP_SWAPI_APERTURE_R
           |    |
           |    IMAP_SWAPI_SUNGLASSES
           |
           IMAP_CODICE
           |    |
           |    IMAP_CODICE_LO_APERTURE_01
           |    |
           |    |...
           |    |
           |    IMAP_CODICE_LO_APERTURE_24
           |    |
           |    IMAP_CODICE_HI_APERTURE_01
           |    |
           |    |...
           |    |
           |    IMAP_CODICE_HI_APERTURE_12
           |
           IMAP_HIT
           |    |
           |    IMAP_HIT_L1_APERTURE_01
           |    |
           |    |...
           |    |
           |    IMAP_HIT_L1_APERTURE_10
           |
           IMAP_IDEX
           |    |
           |    IMAP_IDEX_DETECTOR
           |    |
           |    IMAP_IDEX_FULL_SCIENCE
           |
           IMAP_GLOWS_BASE
                |
                IMAP_GLOWS

IMAP Spacecraft Frame
========================================================================

   \begindata

        FRAME_IMAP_SPACECRAFT       = -43000
        FRAME_-43000_NAME           = 'IMAP_SPACECRAFT'
        FRAME_-43000_CLASS          = 3
        FRAME_-43000_CLASS_ID       = -43000
        FRAME_-43000_CENTER         = -43
        CK_-43000_SCLK              = -43
        CK_-43000_SPK               = -43

   \begintext


   The orientation of the spacecraft body frame with respect to an
   inertial frame, J2000 for IMAP, is provided by a C-kernel (see [3]
   for details).

   The spacecraft coordinate frames are defined by the IMAP control
   documents (see [4,5], NB, figure 2.2). There are two frames described
   there: Observatory Mechanical Design Reference Frame (most relevant)
   and Observatory Pointing and Dynamics Reference Frame (less relevant
   for this frame kernel).


   Observatory Mechanical Design Reference Frame (IMAP_SPACECRAFT)
   ---------------------------------------------------------------------

   If not explicitly stated, references to 'spacecraft mechanical frame'
   'spacecraft frame', or 'S/C frame' will refer to this frame.

   All instruments and component placements and orientations are defined
   using this coordinate frame reference.

   Origin:   Center of the launch vehicle adapter ring at the
             observatory/launch vehicle interface plane

   +Z axis:  Perpendicular to the launch vehicle interface plane pointed
             in the direction of the top deck (runs through the center
             of the central cylinder structure element)

   +Y axis:  Direction of the vector orthogonal to the +Z axis and
             parallel to the deployed MAG boom

   +X axis:  The third orthogonal axis defined using an X, Y, Z ordered
             right hand rule

   NB: The Observatory Pointing and Dynamics Reference Frame is also
   defined in [5]. It is identical to the observatory mechanical design
   reference frame, but with the origin translated to the observatory
   center of mass (which changes with boom deployment and fuel usage).
   The offset difference between the mechanical and dynamic frame is
   within the uncertainty range of the ephemeris, so the mechanical
   design frame is used here for definiteness.

   Three different views [5,6] of the spacecraft with labeled components
   are presented below for illustrative purposes.


   IMAP -Z Bottom View (Figure 3-2 in [5], G-G in [6] rotated 180°)
   ---------------------------------------------------------------------
                       ---------
                      | +X axis |         --------------------
                       ---------         | +Z axis facing Sun |
                           .             |      into page     |
                          /|\             --------------------
                           |
                           |
                           |
                                              _
                  HI 45    /`~~__HI 90          `+    direction of
                    , = .^ - /_  ``-.              '.    positive
                  .+     +     `^~/ ./                ~    rotation
                 ^ +     + . -- ' ``  \   _-~          \
            _   /   ',= '              \~'`   \ IMAP    \
    ULTRA /' '-_ .~ '                   \,.=.. \ LO    \|/
     90  /    ~          _,.,_         +      + \       '
        /  ,~'        +'       `'+     +      +  \
       / ~^        .'        , = .'.    '- =''  -``   ---------
       ^/         /  , = .  +     + \       \~'`     | +Y axis |----->
        |        .  +     + +     +  .       \        ---------     ___
        |        |  +     +  ' = '   |        \--------------------|   |
   SWAPI|        |   ' = ', - .      |        /--------------------|___|
     _+_:        '       +     +     '       /         MAG boom
    \_ __\__      \      +     +    /      /^*~,
      +  |  SWE    '.     ' = '   .'    ULTRA  /
          `~-'       '~..,___,..~'       45   /~,*
           _\                           /    /~,*`
        * / CODICE                      ^*._/  *` HIT
         *\ _/`.                       /
          *   / /~ _ _  ,.-^-., _ _ _ /
             '='       +       +
            GLOWS      +       +
                        '-.,.-'
                          IDEX


   IMAP +X Side View (F-F in [6])
   ---------------------------------------------------------------------
                       ---------
                      | +Z axis |
                       ---------               ---------------------
                           .                  | +X axis out of page |
                          /|\                  ---------------------
                           |      LGA
         __________________|______|^|_________                      ___
   SWAPI|__________________|__________________|====================|   |
    #|-|        |          |  .-==-,  |        /     MAG boom      '---'
    #|-| {|##   |          | /      \ |        |
       | {|##   |          |{  HI 90 }| IMAP LO|
       | {|##   |   _.._   | \      / |  _.,   |
       | ULTRA  | /      \ |  `-==-'  | / __`',|
       |  90    | \ HI 45/ |          | \ \_\ ;|
       |        |  '----`  |          |  ~._ + |
       '-------------------|----------/--------'
          |  |   \_________O_________/ | |        ---------------->
          |__|        -----------      /_\           ---------
          STAR       | S/C FRAME |     MGA          | +Y axis |
        TRACKERS     |  ORIGIN   |                   ---------
                      -----------


   IMAP -X Side View (C-C in [6])
   ---------------------------------------------------------------------
                                           ---------
                                          | +Z axis |
        -------------------                ---------
       | +X axis into page |                   .
        -------------------                   /|\
                                      LGA      |
    ___                      _________|^|______|__________________
   |   |====================|__________________|_____________ __ _|SWAPI
   '---'      MAG boom     \     __ |          |          | //  \ /--|#
                           |( )=|__||          |          | \\__/ \--|#
                           |  HIT   |         _|_  IDEX   | CODICE |
                           |        |      ,.' | '.,      |        |
                           |   ____ |     [  \ | /  ]     |     SWE|
                         ULTRA ##',',      |,.'|'.,|     GLOWS  (#)|
                           45  ####'.      +   |   +      \\(O) |-||
                           '----####/----- +   |   + --------------'
        <----------------        | | \______'-.O.-'______/   |  |
             ---------           /_\      -----------        |__|
            | +Y axis |          MGA     | S/C FRAME |       STAR
             ---------                   |  ORIGIN   |     TRACKERS
                                          -----------


   IMAP Component Location - Azimuth and Elevation
   ---------------------------------------------------------------------

   Payload and subsystem component locations are specified[5,6] in the
   Observatory Mechanical Design Reference Frame (described above).
   Locations are defined in azimuth and elevation (and resultant
   direction cosign matrices) of these angles[6] in the same reference
   frame. The azimuth and elevation angle diagram is provided below.

   In general, descriptions in this kernel treat the +Z direction as
   "up" and the -Z direction as "down." Locations referred to as "above"
   are generally closer to the Sun, and vice versa for "below." The
   "upper" side of the spacecraft is the plane of the solar panels,
   while the "lower" side may refer to the area near the adapter ring.
   If ambiguity could arise, more thorough descriptions will be used.


                                Toward Sun

                                 +Z axis
                                     .
                                     |
                                     .
                                     |
                                     .    Component
                                     |    Location/
                                     .   Orientation
                                     |        @
                      Toward         .      .'|
                       MAG           |    +`  |
                 .~ '` Boom     S/C  .  .` \  |
           .~ '`               FRAME |.`    : |
     / ~'`                    ORIGIN O      | |
    *---                     .~ '`    \  Elevation
                       .~ '`           \    | |
                 .~ '`                  \   ; |~
           .~ '\                         \ /  |  ^~
      +Y axis   \    \                    +   |     ^~
                 '.   '~,                  \  |        ^~
                   '~     Azimuth           \ |           ^~
                      '~.         `^~->      \|          -X axis
                          ' ~ ., _       _ ,.~
                                   ``'``


   IMAP Component Orientation - Azimuth and Elevation
   ---------------------------------------------------------------------

   In addition to the rotation matrices, azimuth and elevation are used
   to specify look direction (i.e., boresight) of the science payload
   components and thrusters. However, these two angles are not adequate
   to specify the complete orientation of the components--a secondary
   axis must be specified to complete the rotation.

   The look direction, D, in the frame of the spacecraft for azimuth, az
   and elevation, el, is:

      D = [ -cos(el) x sin(az), cos(el) x cos(az), sin(el) ]

   For all practical purposes, the look direction (primary axis)
   corresponds to one of the six axis-aligned directions of the local
   coordinate system of the instrument: X', Y', Z', -X', -Y', -Z'. While
   the azimuth/elevation of the instrument look direction is provided in
   the spacecraft MICD[4], the local coordinate axis in which it
   corresponds is provided in the instrument's MICD.

   The secondary axis, S, must be perpendicular to D for the following
   discussion. It will generally be specified in one of two ways:

      1) S is one of the six axis-aligned directions of the
         spacecraft coordinate system: X, Y, Z, -X, -Y, -Z

      2) S lies in the plane perpendicular to one of the axes of the
         spacecraft coordinate system: X, Y, Z, -X, -Y, -Z

   Similar to the look direction, this direction will then be assigned
   to correspond to one of the six instrument directions X', Y', Z',
   -X', -Y', -Z'.

   For definiteness, it is assumed that the third axes, N = D x S,
   completes the righthanded coordinate system.

   The rotation matrix specifying the component frame, X'Y'Z', in the
   spacecraft frame, XYZ, is:

                Ux    Uy    Uz

      [ X ]   [ R11   R12   R13 ] [ X']
      [   ]   [                 ] [   ]
      [ Y ] = [ R21   R22   R23 ] [ Y']
      [   ]   [                 ] [   ]
      [ Z ]   [ R31   R32   R33 ] [ Z']

   with Ux, Uy, Uz specifying the unit column vectors of the rotation.
   Because the primary and secondary axes, D and S, lie along the local
   axes of the instrument coordinate system (X'Y'Z'), they are simply
   the column vectors of the rotation matrix (assuming properly unit).


   IMAP Component Orientation - Euler Angles
   ---------------------------------------------------------------------

   When the orientation is not specified in azimuth/elevation, or the
   secondary is not well-defined, we try to deduce the most straight-
   forward definition using a simple secondary axis. Sometimes a
   single axis-aligned rotation applied BEFORE the general rotation
   allows a simple secondary axis to notionally be used to accurately
   define the coordinates; see Hi 45 or Hi 90 for this case.

   It is also possible to deduce the Euler angles to produce more
   precise rotation matrices. For most components, before final
   alignments are calculated, these angles are in whole degrees.
   (However, see Hi 45 for a counterexample).

   The spacecraft subsystems such as the star trackers have complete
   rotation matrices that fully define the orientation of each
   component. These matrices, while complete, are not conducive to
   visualizing the orientation of a component on the spacecraft bus.

   As it happens, when applied to rotations, the azimuth and elevation
   are nearly identitical to the first two Euler angles of the ZXZ
   intrinsic rotation. For the Euler angles (A, B, Y), this is defined
   as follows[11].

   Let xyz represent the coordinate axes of the fixed frame, and XYZ
   are the axes of the fully rotated frame expressed in the xyz frame.
   Three successive, ordered rotations about the axes are performed:

   1) Righthanded rotation about z by the angle A ∈ [-π, π); the rotated
      frame is defined x'y'z', with z' = z. The new frame x'y'z' is
      expressed in the coordinates of the original frame xyz.

   2) Righthanded rotation about x' by the angle B ∈ [0,π]; the rotated
      frame is defined x"y"z", with x" = x'. The new frame x"y"z" is
      expressed in the coordinates of the original frame xyz.

   3) Righthanded rotation about z" by the angle Y ∈ [-π,π); the rotated
      frame is defined XYZ, with Z = z".  The final frame XYZ is
      expressed in the coordinates of the original frame xyz.


                               Euler Angles
                          Intrinsic ZXZ Rotation

                                   z axis
                                     .
                                     |       Y axis
                                  _._.         /
                             , B `   |        /
                  Z axis  ,-`        .       /
                      ^, ^           |      /
                        ^,           .     /
                          ^,         |    /
                            ^,       .   /
                              ^,     |  /             _ X axis
                                ^,   . /        _ ~ ^
                                  ^, |/   _ ~ ^      ^
                                   .~ ~ ^             |
                             .~ '`    \ ^~            ;
                       .~ '` \         \   ^~         ;
                 .~ '`        ',        \     ^~     ,
           .~ '`                ` A      \       ^  Y
       x axis                       `^~-> \        , ~
                                           \     ~`    ^~
                                            \- ^          ^~
                                             \            y axis
                                              \
                                          x'=x" axis


   Comparing the two figures, we see that A = azimuth and B appears to
   coincide with elevation. However, while B lies on the range [0,π],
   conventionally, elevation ∈ [-π/2,π/2]. This range for elevation does
   not capture all possible orientations, e.g., a playing card facing
   upward cannot be placed facing downward with elevation ∈ [-π/2,π/2].

   So, we need to supplement the azimuth and elevation nomenclature with
   fully specified Euler angles.

   The technical documents [4,5,6] give rotation matrix elements to six
   decimal places, which is not sufficient for accurate pointing in the
   SPICE toolkit. The remedy to this inaccuracy is provided below.

   Given an insufficiently-accurate rotation matrix, M, with column
   vectors Vx, Vy, Vz:

            Vx    Vy    Vz

          [ M11   M12   M13 ]
          [                 ]
      M = [ M21   M22   M23 ]
          [                 ]
          [ M31   M32   M33 ]

   A rotation matrix, R, with column unit vectors Ux, Uy, Uz:

            Ux    Uy    Uz

          [ R11   R12   R13 ]
          [                 ]
      R = [ R21   R22   R23 ]
          [                 ]
          [ R31   R32   R33 ]

   is calculated so that column vectors are orthonormal to within double
   precision accuracy (an operation SPICE calls "sharpening"):

      Uz = Vz / |Vz|

      Uy = Uz x (Vx / |Vx|)

      Ux = Uy x Uz

   These calculations are done outside of the SPICE library, but using
   numerically stable algorithms as SPICE does. Sharpening by starting
   with the X or Y direction, as opposed to Z, can be accomplished by
   cyclically permuting x,y,z above. SPICE, for example, starts with X.

   With a precise (though not necessarily accurate) rotation matrix,
   the instrinsic ZXZ Euler angles (A, B, Y) are calculated:

      A' = atan2(R13, -R23)
                   ______________
      B' = atan2(\/ 1 - R33 x R33 , R33)

      Y' = atan2(R31, R32)

   These values are rounded to regain the assumed original orientation:

      A = round(A') to nearest 1/1000th degree

      B = round(B') to nearest 1/1000th degree

      Y = round(Y') to nearest 1/1000th degree

   And finally, the rotation matrix elements are recalculated:

      R11 =  c1 x c3 - s1 x c2 x s3

      R21 =  s1 x c3 + c1 x c2 x s3

      R31 =  s2 x s3

      R12 = -c1 x s3 - s1 x c2 x c3

      R22 = -s1 x s3 + c1 x c2 x c3

      R32 =  s2 x c3

      R13 =  s1 x s2

      R23 = -c1 x s2

      R33 =  c2

   where:

      c1 = cos(A)

      s1 = sin(A)

      c2 = cos(B)

      s2 = sin(B)

      c3 = cos(Y)

      s3 = sin(Y)

   When B = 0, the angles A and Y are degenerate; Y = 0 in this case.

   In the subsequent frames defined below, when Euler angles (A, B, Y)
   are referenced without further discussion, they will refer to the
   Euler angles as defined here. Otherwise, definitions will be given
   inline with the discussion.


   When Look Direction is Well-Defined
   ---------------------------------------------------------------------

   When the look direction is well-defined, but the secondary axis is
   not, we replace the column of the imprecise rotation matrix with
   the exact look direction, and proceed with the calculations above.


IMAP Thruster Frames
========================================================================

   There are four axial (A) thrusters and eight radial (R) thrusters on
   IMAP[6]. The table below shows the thruster positions defined in the
   spacecraft frame[6], at the intersection of the thrust axis and the
   nozzle exit plane. The unit direction vectors listed in the table
   below point in the direction of the thruster exhaust. The positional
   information may be captured in the IMAP structure SPK, while the
   orientation information is captured here.


   Thruster         ID      X (mm)    Y (mm)    Z (mm)   UnitDir (X,Y,Z)
   ---------------- ------  --------  --------  -------  ---------------
   IMAP_THRUSTER_A1 -43010   1007.28    516.50  1312.40  (   0,  0,  1 )
   IMAP_THRUSTER_A2 -43011  -1007.28   -516.50  1312.40  (   0,  0,  1 )
   IMAP_THRUSTER_A3 -43012  -1007.28   -516.50   101.77  (   0,  0, -1 )
   IMAP_THRUSTER_A4 -43013   1007.28    516.50   101.77  (   0,  0, -1 )
   IMAP_THRUSTER_R1 -43020   -126.90   1237.78   841.12  (-0.5, 0.866,0)
   IMAP_THRUSTER_R2 -43021    126.90  -1237.78   841.12  ( 0.5,-0.866,0)
   IMAP_THRUSTER_R3 -43022  -1008.49    728.79   841.12  (-0.5, 0.866,0)
   IMAP_THRUSTER_R4 -43023   1008.49   -728.79   841.12  ( 0.5,-0.866,0)
   IMAP_THRUSTER_R5 -43024   -126.90   1237.78   447.42  (-0.5, 0.866,0)
   IMAP_THRUSTER_R6 -43025    126.90  -1237.78   447.42  ( 0.5,-0.866,0)
   IMAP_THRUSTER_R7 -43026  -1008.49    728.79   447.42  (-0.5, 0.866,0)
   IMAP_THRUSTER_R8 -43027   1008.49   -728.79   447.42  ( 0.5,-0.866,0)


   Thruster Locations and Directions
   ---------------------------------------------------------------------

   The four axial thrusters[6] are directed along the spacecraft Z axis,
   with A1,A2 located on the +Z side of the spacecraft and A3,A4 located
   on the -Z side. A1,A2 fire in the +Z direction, while A3,A4 fire in
   the -Z direction. A1 and A4 are aligned in the Z direction, while
   A2 and A3 are aligned but on the opposite side of the S/C as A1/A4.

   The eight radial thrusters[6] are grouped into four pairs (R1/R5,
   R2/R6, R3/R7, R4/R8); each pair is aligned along the Z direction and
   fire in the same direction. There are two distinct firing directions,
   all perpendicular to the spacecraft Z axis: R1/R5 & R3/R7 fire toward
   the +Y direction (with a slight -X component), while R2/R6 & R4/R8
   fire in the -Y direction (with a slight +X component). Thrusters
   R1-R4 are located above the center of mass (towards the Sun), while
   thrusters R5-R8 are located below the center of mass (away from the
   Sun). The table below shows the azimuth of location and direction of
   radial thrusters calculated from using thruster table above.


                       Location Azim   Direction Azim
                       --------------  --------------
                R1/R5           5.85°           30.0°
                R2/R6   180° +  5.85°    180° + 30.0°
                R3/R7          54.15°           30.0°
                R4/R8   180° + 54.15°    180° + 30.0°


                               +X axis       +Z axis facing Sun
                                   .              into page
                                  /|\
                                   |
                                   |
                                   |             A1 (on +Z side)
                                                 A4 (on -Z side)
           R4/R8 Dir               /`~~__       /
            '~._            , = .^ - /_  ``-.  /
                /~._      .+     +     `^~/ .\/
            30°|    '~.          + . -- ' `` @\   _-~
           - - + - - - -# R4/R8                \~'`   \
                  /' '-_ .                      \,.=.. \
                 /    ~          _,.,_         +      + \
   R2/R6 Dir    /  ,~'        +'       `'+     +      +  \
    '~._       / ~^        .'        , = .'.    '- =''  -``
        /~._   ^/         /  , = .  +     + \       \~'`
    30°|    '~. |        .  +     + +     +  .       \    +Y axis ----->
   - - + - - - -|# R2/R6 |  +     +  ' = '   |        \
                |        |   ' = ', - .      | R1/R5 #._- - - - - + - -
             _+_:        '       +     +     '       /  '~._      |
            \_ __\__      \      +     +    /      /^*~,    '~._ /  30°
              +  |   \     '.     ' = '   .'      /    /        '~.
                  `~-'       '~..,___,..~'       /    /~,*     R1/R5 Dir
                   _\                           /    /~,*`
                * /  \                          ^*._/  *`
                 *\ _/`.                R3/R7 #/._- - - - - + - -
                  *   / /\@_ _  ,.-^-., _ _ _ /   '~._      |
                     '='  |    +       +              '~._ /  30°
                          |    +       +                  '~.
                          |     '-.,.-'                  R3/R7 Dir
                          |
                          A2 (on +Z side)
                          A3 (on -Z side)


   Axial Thruster Frames
   ---------------------------------------------------------------------

   Each axial thruster has a frame defined so that the thruster exhaust
   exits in the +Z' direction. The +Y' axis is chosen to lie in the
   direction of the MAG boom. X' = Y' x Z' completes the frame.

      [X]      [  1   0   0 ] [X']
      [Y]    = [  0   1   0 ] [Y']
      [Z]S/C   [  0   0   1 ] [Z']Axial Thrusters A1,A2

      [X]      [ -1   0   0 ] [X']
      [Y]    = [  0   1   0 ] [Y']
      [Z]S/C   [  0   0  -1 ] [Z']Axial Thrusters A3,A4


                               Axial Thruster
                             Exhaust Direction

                                  +Z' axis
                                     |
                                     |
                                 _. -|- ._
                               ,'    |    ',
                               ,     |     ,
                               | -.,_|_,.- |
                               '           '
                                '         '
                                 ;       ;
                                  ;     ;
                                  :     ;
                                   ,   ,         Toward
                                   ',_,'     ^~      MAG
                             .~ '`      ^~      ^~      Boom
                       .~ '`               ^~      ^~
                 .~ '`                        ^~      ^~
           .~ '`                                 ^~      ^~ \
      +X' axis                                      ^~     --*
                                                       ^~
                                                          ^~
                                                         +Y' axis


   \begindata

   FRAME_IMAP_THRUSTER_A1      = -43010
   FRAME_-43010_NAME           = 'IMAP_THRUSTER_A1'
   FRAME_-43010_CLASS          = 4
   FRAME_-43010_CLASS_ID       = -43010
   FRAME_-43010_CENTER         = -43
   TKFRAME_-43010_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43010_SPEC         = 'MATRIX'
   TKFRAME_-43010_MATRIX       = (  1,
                                    0,
                                    0,
                                    0,
                                    1,
                                    0,
                                    0,
                                    0,
                                    1  )

   FRAME_IMAP_THRUSTER_A2      = -43011
   FRAME_-43011_NAME           = 'IMAP_THRUSTER_A2'
   FRAME_-43011_CLASS          = 4
   FRAME_-43011_CLASS_ID       = -43011
   FRAME_-43011_CENTER         = -43
   TKFRAME_-43011_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43011_SPEC         = 'MATRIX'
   TKFRAME_-43011_MATRIX       = (  1,
                                    0,
                                    0,
                                    0,
                                    1,
                                    0,
                                    0,
                                    0,
                                    1  )

   FRAME_IMAP_THRUSTER_A3      = -43012
   FRAME_-43012_NAME           = 'IMAP_THRUSTER_A3'
   FRAME_-43012_CLASS          = 4
   FRAME_-43012_CLASS_ID       = -43012
   FRAME_-43012_CENTER         = -43
   TKFRAME_-43012_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43012_SPEC         = 'MATRIX'
   TKFRAME_-43012_MATRIX       = ( -1,
                                    0,
                                    0,
                                    0,
                                    1,
                                    0,
                                    0,
                                    0,
                                   -1  )

   FRAME_IMAP_THRUSTER_A4      = -43013
   FRAME_-43013_NAME           = 'IMAP_THRUSTER_A4'
   FRAME_-43013_CLASS          = 4
   FRAME_-43013_CLASS_ID       = -43013
   FRAME_-43013_CENTER         = -43
   TKFRAME_-43013_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43013_SPEC         = 'MATRIX'
   TKFRAME_-43013_MATRIX       = ( -1,
                                    0,
                                    0,
                                    0,
                                    1,
                                    0,
                                    0,
                                    0,
                                   -1  )

   \begintext


   Radial Thrusters
   ---------------------------------------------------------------------

   Each radial thruster has a frame defined so that the thruster exhaust
   exits in the +Y' direction. The +Z' axis is chosen to lie along the
   spacecraft +Z axis (toward Sun). X' = Y' x Z' completes the frame.

      [X]      [  cos( 30) -sin( 30)  0 ] [X']
      [Y]    = [  sin( 30)  cos( 30)  0 ] [Y']
      [Z]S/C   [      0         0     1 ] [Z']Rad. Thrusters R1,R3,R5,R7

      [X]      [  cos(210) -sin(210)  0 ] [X']
      [Y]    = [  sin(210)  cos(210)  0 ] [Y']
      [Z]S/C   [      0         0     1 ] [Z']Rad. Thrusters R2,R4,R6,R8


                                 Toward Sun

                                  +Z' axis
                                     .
                                     |
                                     .
                                     |
                                     .
                                     |
                                     .
    Radial Thruster                  |
   Exhaust Direction                 .
                                     |
           .~ '`                     .
    /.~ '`       _,,~ ~ ~ ~ ~ ~ ~ ~  |
   *--        .;-.                  \ ~
            ,'    '.                ~   ^~
            ;       \            ~'        ^~
            |   .~ '`:         ~'             ^~
           .~ '`     |      ~'                   ^~
      ~ '`   \       ; _ ~'                         ^~
   +Y' axis   '.,_._;-'                                ^~
                                                          ^~
                                                         -X' axis


   \begindata

   FRAME_IMAP_THRUSTER_R1      = -43020
   FRAME_-43020_NAME           = 'IMAP_THRUSTER_R1'
   FRAME_-43020_CLASS          = 4
   FRAME_-43020_CLASS_ID       = -43020
   FRAME_-43020_CENTER         = -43
   TKFRAME_-43020_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43020_SPEC         = 'MATRIX'
   TKFRAME_-43020_MATRIX       = (  0.86602540378443865,
                                    0.50000000000000000,
                                    0.00000000000000000,
                                   -0.50000000000000000,
                                    0.86602540378443865,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    1.00000000000000000  )

   FRAME_IMAP_THRUSTER_R2      = -43021
   FRAME_-43021_NAME           = 'IMAP_THRUSTER_R1'
   FRAME_-43021_CLASS          = 4
   FRAME_-43021_CLASS_ID       = -43021
   FRAME_-43021_CENTER         = -43
   TKFRAME_-43021_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43021_SPEC         = 'MATRIX'
   TKFRAME_-43021_MATRIX       = ( -0.86602540378443865,
                                   -0.50000000000000000,
                                    0.00000000000000000,
                                    0.50000000000000000,
                                   -0.86602540378443865,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    1.00000000000000000  )

   FRAME_IMAP_THRUSTER_R3      = -43022
   FRAME_-43022_NAME           = 'IMAP_THRUSTER_R3'
   FRAME_-43022_CLASS          = 4
   FRAME_-43022_CLASS_ID       = -43022
   FRAME_-43022_CENTER         = -43
   TKFRAME_-43022_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43022_SPEC         = 'MATRIX'
   TKFRAME_-43022_MATRIX       = (  0.86602540378443865,
                                    0.50000000000000000,
                                    0.00000000000000000,
                                   -0.50000000000000000,
                                    0.86602540378443865,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    1.00000000000000000  )

   FRAME_IMAP_THRUSTER_R4      = -43023
   FRAME_-43023_NAME           = 'IMAP_THRUSTER_R4'
   FRAME_-43023_CLASS          = 4
   FRAME_-43023_CLASS_ID       = -43023
   FRAME_-43023_CENTER         = -43
   TKFRAME_-43023_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43023_SPEC         = 'MATRIX'
   TKFRAME_-43023_MATRIX       = ( -0.86602540378443865,
                                   -0.50000000000000000,
                                    0.00000000000000000,
                                    0.50000000000000000,
                                   -0.86602540378443865,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    1.00000000000000000  )

   FRAME_IMAP_THRUSTER_R5      = -43024
   FRAME_-43024_NAME           = 'IMAP_THRUSTER_R5'
   FRAME_-43024_CLASS          = 4
   FRAME_-43024_CLASS_ID       = -43024
   FRAME_-43024_CENTER         = -43
   TKFRAME_-43024_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43024_SPEC         = 'MATRIX'
   TKFRAME_-43024_MATRIX       = (  0.86602540378443865,
                                    0.50000000000000000,
                                    0.00000000000000000,
                                   -0.50000000000000000,
                                    0.86602540378443865,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    1.00000000000000000  )

   FRAME_IMAP_THRUSTER_R6      = -43025
   FRAME_-43025_NAME           = 'IMAP_THRUSTER_R6'
   FRAME_-43025_CLASS          = 4
   FRAME_-43025_CLASS_ID       = -43025
   FRAME_-43025_CENTER         = -43
   TKFRAME_-43025_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43025_SPEC         = 'MATRIX'
   TKFRAME_-43025_MATRIX       = ( -0.86602540378443865,
                                   -0.50000000000000000,
                                    0.00000000000000000,
                                    0.50000000000000000,
                                   -0.86602540378443865,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    1.00000000000000000  )

   FRAME_IMAP_THRUSTER_R7      = -43026
   FRAME_-43026_NAME           = 'IMAP_THRUSTER_R7'
   FRAME_-43026_CLASS          = 4
   FRAME_-43026_CLASS_ID       = -43026
   FRAME_-43026_CENTER         = -43
   TKFRAME_-43026_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43026_SPEC         = 'MATRIX'
   TKFRAME_-43026_MATRIX       = (  0.86602540378443865,
                                    0.50000000000000000,
                                    0.00000000000000000,
                                   -0.50000000000000000,
                                    0.86602540378443865,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    1.00000000000000000  )

   FRAME_IMAP_THRUSTER_R8      = -43027
   FRAME_-43027_NAME           = 'IMAP_THRUSTER_R6'
   FRAME_-43027_CLASS          = 4
   FRAME_-43027_CLASS_ID       = -43027
   FRAME_-43027_CENTER         = -43
   TKFRAME_-43027_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43027_SPEC         = 'MATRIX'
   TKFRAME_-43027_MATRIX       = ( -0.86602540378443865,
                                   -0.50000000000000000,
                                    0.00000000000000000,
                                    0.50000000000000000,
                                   -0.86602540378443865,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    1.00000000000000000  )

   \begintext


IMAP Digital Sun Sensor and Star Tracker Frames
========================================================================

   There are two digital sun sensors (DSS)[6]: one on the +Z side of the
   spacecraft pointing in +Z direction, and one on the -Z side pointing
   mostly in the radial direction with a 30° tilt in the -Z direction.
   They are approximated aligned along the spacecraft Z axis, though the
   origins are offset from absolute alignment by a few centimeters (see
   table below). Azimuthally, the sun sensors are located near the SWAPI
   instrument approximately 18° off of the Y-Z plane.

   There are two star trackers mounted adjacent to each other on the
   underside of the spacecraft close to the -Z digital star sensor[6].
   Their boresights are generally downward (towards -Z), with an angular
   separation of 24°. One is angled toward the +X direction, the other
   angled towards the -X direction.

   Positional information may be captured in the IMAP structure SPK,
   while the orientation information is captured here.


   Digital Sun Sensor    ID      X (mm)    Y (mm)    Z (mm)    Loc. Azim
   --------------------  ------  --------  --------  --------  ---------
   IMAP_SUN_SENSOR_PZ    -43030   -364.22  -1121.90   1301.67   162.014°
   IMAP_SUN_SENSOR_MZ    -43031   -379.11  -1167.77     72.89   162.014°


   Digital Star Tracker  ID      X (mm)    Y (mm)    Z (mm)    Loc. Azim
   --------------------  ------  --------  --------  --------  ---------
   IMAP_STAR_TRACKER_PX  -43040    -45.75   -906.66    159.88   177.111°
   IMAP_STAR_TRACKER_MX  -43041   -188.05   -881.57    142.79   167.959°


   #####################################################################
   #                     /                            _-        __.----#
   #                   ,'                           ~`     _.~^'       #
   #                  /                           ~`    ,~^            #
   #                ,'   +Z axis facing Sun     .`    .^       +X axis #
   #               /        into page          /    .^               . #
   #              |                           :    /_,-----,_       /|\#
   #              |                          ~    ~`         ^.      | #
   #              |                         ^    ^             ^_    | #
   #              |                        /    /                ,   | #
   #              |                       ,    ,                 ;   | #
   #              |                       ;    ;                  }  | #
   #              |           ___        :    :                   ~ ___#
   # -Y axis   ___|         .`   `.      |    |                   }/  _#
   # <------  |===|        ;+X Star;     |    |.                 ;/ (` #
   #              |        ;Tracker;     |    |'                ; \ (,_#
   #              |        `,     ,`     |    | ',             ,   \___#
   #              |          '---'       :    :   '-.,_____,.-`     _,~#
   #              |              _,;@     ;    ;                 ,"    #
   #     /|       |        @*^^'`         :    :                ;      #
   #  /^' {    _,;|            ,---,       ;    ;              ^       #
   # \    *^^'`   |          .^     ^.      ~    ~            {        #
   # |  SWAPI     {  _,      |-X Star|       \    \           |        #
   #  \          _,;*^ \     .Tracker.        \    *          {        #
   #  |    *^^'`   \ -Z DSS   ^.___.^          ^,   `~_       \        #
   #  \     }       \   _}                       ^_    "~_     ^,      #
   #    ^^'"\\       \*^                           ^,     '-_    ~_    #
   #                  \   (+Z DSS not visible)       "~_      " -, '-  #
   #####################################################################


   Digital Sun Sensors (DSS)
   ---------------------------------------------------------------------

   Each DSS has a frame defined so that the look-direction is along the
   +Z' axis. The digital image rows and columns are aligned with the X'
   and Y' axes of the frame.


                             DSS Look Direction
                                Local Frame

                                  +Z' axis
                                     |
                                     |
                                     |
                                     |
                                     |
                                     |
                                   .~|'`^~
                             .~ '`   |     ^~
                       .~ '`   __,=# |   ,_   ^~
                 .~ '`   __,=#^^^    |@   ^%,_   ^~
                ~    ,=#^^^          |       ^%,_   ^~
                | ^~              ,.~^~         ^%,_   ^~
                |    ^~     ,.~ '`      ^~         ^%    ,^
                |     ,.^~' @              ^~       .~ '` |
                ^~.''`     ^~                @^~ '`       |
           .~ '``  ^~         ^~         .~ '`   ^~       |
      +X' axis        ^~         ^~.~ '`            ^~.~ '`
                         ^~        |            .~ '`  ^~
                            ^~     |      .~ '`           ^~
                               ^~  |.~ '`                +Y' axis


   The rotation matrices orienting each DSS on the spacecraft are
   given by [6]:

      [X]      [   0.951057   0.309017   0.000000  ] [X']
      [Y]    = [  -0.309017   0.951057   0.000000  ] [Y']
      [Z]S/C   [   0.000000   0.000000   1.000000  ] [Z'] +Z DSS

      [X]      [   0.951078  -0.154380  -0.267616  ] [X']
      [Y]    = [  -0.308952  -0.475579  -0.823640  ] [Y']
      [Z]S/C   [  -0.000116   0.866025  -0.500000  ] [Z'] -Z DSS

   Using the method described in a previous section, the Euler angles
   rounded to 1/1000th of a degree are:

      +Z DSS: (A, B, Y)  =  (  -18.000°,   0.000°,  0.000°  )

      -Z DSS: (A, B, Y)  =  (  -18.000°, 120.000°, -0.008°  )

   Using the formulas described in the Euler angles section above, the
   rotation matrices have been recalculated to double precision.


   \begindata

   FRAME_IMAP_SUN_SENSOR_PZ    = -43030
   FRAME_-43030_NAME           = 'IMAP_SUN_SENSOR_PZ'
   FRAME_-43030_CLASS          = 4
   FRAME_-43030_CLASS_ID       = -43030
   FRAME_-43030_CENTER         = -43
   TKFRAME_-43030_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43030_SPEC         = 'MATRIX'
   TKFRAME_-43030_MATRIX       = (  0.95105651629515350,
                                   -0.30901699437494734,
                                    0.00000000000000000,
                                    0.30901699437494734,
                                    0.95105651629515350,
                                    0.00000000000000000,
                                   -0.00000000000000000,
                                   -0.00000000000000000,
                                    1.00000000000000000  )

   FRAME_IMAP_SUN_SENSOR_MZ    = -43031
   FRAME_-43031_NAME           = 'IMAP_SUN_SENSOR_MZ'
   FRAME_-43031_CLASS          = 4
   FRAME_-43031_CLASS_ID       = -43031
   FRAME_-43031_CENTER         = -43
   TKFRAME_-43031_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43031_SPEC         = 'MATRIX'
   TKFRAME_-43031_MATRIX       = (  0.95107808048040110,
                                   -0.30895059509261280,
                                   -0.00012091995722272,
                                   -0.15437570314113858,
                                   -0.47557140042407403,
                                    0.86602539534263330,
                                   -0.26761656732981740,
                                   -0.82363910354633210,
                                   -0.49999999999999983  )

   \begintext


   Star Trackers
   ---------------------------------------------------------------------

   Each star tracker has a frame defined so that the look-direction is
   along the +Z' axis. The digital image rows and columns are aligned
   with the X' and Y' axes of the frame.


                         Star Tracker Look Direction
                                Local Frame

                                  +Z' axis

                                     |
                                     |
                                     |
                                     |
                                 _. -|- ._
                              ,'     |     ',
                             |  .~ '` ^~    ,|
                          .~ '` ~ .,_ _,.^~' |
                    .~ '`    |              ^~
              .~ '`          |,             ,| ^~
          +X' axis             ' -.,_ _,.- '      ^~
                               |           |         ^~
                               |           |            ^~
                               |           |          +Y' axis
                                '-.,_ _,.-'



   When oriented on the spacecraft:

   - The tracker X' axis mostly points towards the spacecraft -X axis
   - The tracker Y' axis mostly points towards the spacecraft +Y axis
   - The tracker Z' axis mostly points towards the spacecraft -Z axis


   #####################################################################
   #                                                           {  {    #
   #                                                           )  )    #
   #                                                           @  @    #
   #                                                          {  {     #
   #                 _,~--~,_                                 |  |     #
   #               ,"        ",                           ,-----,'     #
   #             ;              ;                         |     |      #
   #  +X Star   /                \                        |     |      #
   #  Tracker  {              __,.-  +Y'                  '-----'      #
   #           |         ..-^"    |:                          |  |     #
   #           {         ;        ;}                          |  |     #
   #           {\         ;      / }                          {  {     #
   #            {^,       :    ,^ ;                            @  @    #
   #             . ~_      ; _~  ,`                            |  |    #
   #              `,  '~--~"   ,^ "'                           |  |    #
   #              '"^--,__ `  '   "^                           {  {    #
   #              `^       +X'    `"`                           )  )   #
   #               "'              ^'                           |  |   #
   #               ^'              '~                           {  {   #
   #                ^,          __,,.~*^#                        )  )  #
   #                ', _,.~-'^'`__,,.~*^#                        |  |  #
   #             #-*~^'_,.~-'^'`    '"                           {  {  #
   #             #-*~^'              "^                           @  @ #
   #                 '"              `"`                          |  | #
   #                 `^               ^`                          {  { #
   #                  "`    _,~^^^~-.,'^                           )  )#
   #                  ^' _-" _,~--~,_ ".'                          (   #
   #                  '^/  ,"        ", \`                          \  #
   #                   , ;              ;',                          \ #
   #                   |/                \|                            #
   #                   {              __,.-  +Y'      Spacecraft Axes  #
   #         -X Star   |         ..-^"    |                            #
   #         Tracker   {         ;        }             +X             #
   #                    \         ;      /               |             #
   #                     ^,       :    ,^                |             #
   #                       ~_      ; _~                  |             #
   #                          '~--~"                     |             #
   #                               `                     x-------- +Y  #
   #                               +X'                +Z into          #
   #                                                    Page           #
   #####################################################################


   The rotation matrices orienting each star tracker on the spacecraft
   are given by [6]:

      [X]      [  -0.963287   0.173648   0.204753  ] [X']
      [Y]    = [   0.169854   0.984808  -0.036104  ] [Y']
      [Z]S/C   [  -0.207912   0.000000  -0.978148  ] [Z']+X Star Tracker


      [X]      [  -0.963287   0.173648  -0.204753  ] [X']
      [Y]    = [   0.169854   0.984808   0.036104  ] [Y']
      [Z]S/C   [   0.207912   0.000000  -0.978148  ] [Z']-X Star Tracker

   Using the method described in a previous section, the Euler angles
   rounded to 1/1000th of a degree are:

      +X Star Tracker: (A, B, Y)  =  (   80.000°, 168.000°, -90.000°  )

      -X Star Tracker: (A, B, Y)  =  ( -100.000°, 168.000°,  90.000°  )

   Use the formulas described in the Euler angles section above, the
   rotation matrices have been recalculated to double precision.


   \begindata

   FRAME_IMAP_STAR_TRACKER_PX  = -43040
   FRAME_-43040_NAME           = 'IMAP_STAR_TRACKER_PX'
   FRAME_-43040_CLASS          = 4
   FRAME_-43040_CLASS_ID       = -43040
   FRAME_-43040_CENTER         = -43
   TKFRAME_-43040_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43040_SPEC         = 'MATRIX'
   TKFRAME_-43040_MATRIX       = ( -0.96328734079294150,
                                    0.16985354835670569,
                                   -0.20791169081775915,
                                    0.17364817766693050,
                                    0.98480775301220800,
                                    0.00000000000000001,
                                    0.20475304505920630,
                                   -0.03610348622615415,
                                   -0.97814760073380570 )

   FRAME_IMAP_STAR_TRACKER_MX  = -43041
   FRAME_-43041_NAME           = 'IMAP_STAR_TRACKER_MX'
   FRAME_-43041_CLASS          = 4
   FRAME_-43041_CLASS_ID       = -43041
   FRAME_-43041_CENTER         = -43
   TKFRAME_-43041_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43041_SPEC         = 'MATRIX'
   TKFRAME_-43041_MATRIX       = ( -0.96328734079294150,
                                    0.16985354835670533,
                                    0.20791169081775915,
                                    0.17364817766693014,
                                    0.98480775301220800,
                                    0.00000000000000001,
                                   -0.20475304505920630,
                                    0.03610348622615410,
                                   -0.97814760073380570 )

   \begintext


IMAP Antenna Frames
========================================================================

   There are two antennas on the spacecraft. The low gain antenna (LGA)
   is located on the +Z side of the spacecraft pointing toward +Z, while
   the medium gain antenna (MGA) is located on the -Z side pointing in
   the -Z direction.


                                           ---------
                                          | +Z axis |
        -------------------                ---------
       | +X axis into page |        #-----#    .
        -------------------         | LGA |   /|\
                                    #-----#    |
    ___                      _________|^|______|__________________
   |   |====================|__________________|_____________ __ _|SWAPI
   '---'      MAG boom     \     __ |          |          | //  \ /--|#
                           |( )=|__||          |          | \\__/ \--|#
                           |  HIT   |         _|_  IDEX   | CODICE |
                           |        |      ,.' | '.,      |        |
                           |   ____ |     [  \ | /  ]     |     SWE|
                         ULTRA ##',',      |,.'|'.,|     GLOWS  (#)|
                           45  ####'.      +   |   +      \\(O) |-||
                           '----####/----- +   |   + --------------'
        <----------------        | | \______'-.O.-'______/   |  |
             ---------           /_\      -----------        |__|
            | +Y axis |        #-----#   | S/C FRAME |       STAR
             ---------         | MGA |   |  ORIGIN   |     TRACKERS
                               #-----#    -----------


   #####################################################################
   #            .-----------------------------------------------------.#
   #            |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|#
   #            |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |#
   #   ,,    _,~'-----|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|#
   #   \ \" ' _,~|___ |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|#
   #    \ \  "        |  |  |  |  |  | SOLAR PANELS |  |  |  |  |  |  |#
   #     \ \:         |--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|#
   #      \,'         |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|#
   #  HIT                |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |#
   #                     |--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|#
   #                     |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|#
   #                       \       ___  |  |  |  |  |  |  |  |  |  |  |#
   #     THRUSTER R3 -->  ,~\     |# #| |--|--|--|--|--|--|--|--|--|--|#
   #                         ^,   |# #| |__|__|__|__|__|__|__|__|__|__|#
   #                           ^~---|---|  |  |  |  |  |  |  |  |  |  |#
   # Spacecraft Axes                |   '-----------------------------'#
   #                                |       ^/~.,             ,.~\^    #
   #            +X               #-----#    { *  `"*,_____,*"`  * }    #
   #                             # LGA #    { *       | |       * }    #
   #             |               #-----#     \ *      | |      * /     #
   #             |                            ~. *    | |    * .~      #
   #             |                              "|~####|####~|"        #
   #             |                                                     #
   #  +Y --------+                                   IDEX              #
   #          +Z out                                                   #
   #          of page                                                  #
   #####################################################################


   #####################################################################
   #        /       #####~._    half of             ~`     _.~^'       #
   #       /       #########~._ ULTRA 45          ~`    ,~^_           #
   #    HIT        ,###########/                .`    .^  ~            #
   #  (just out   /   ########/                /    .^  ,`             #
   #   of view)  ,                            :    /   ,               #
   #            /                            ~    ~`   |               #
   #           ,                            ^    ^     ,               #
   #          /                            /    /       ,              #
   #         ,                            ,    ,         ,             #
   #        /        +Z into      __      ;    ;          -            #
   #       ,           page     .`##`.   :    :            `- . , _ ___#
   #     |/    +Y ------x       ;#**#;   |    |                    /  _#
   #     |\             |       `.##.`   |    |        ,.----.,   / (` #
   #       '            |         |      |    |     _~`        `~_\ (,_#
   #        \           |      #-----#   |    |    ~              ~\___#
   #         '                 # MGA #   :    :  ,`                `,  #
   #          \        +X      #-----#    ;    ;,                    , #
   #           '                          :    :|                    | #
   #            \   _.-----.               ;    ;                    , #
   #             '~         '^,             ~    ~                  ,  #
   #              -|  IMAP   / \             \    \                ,   #
   #             ' |   LO   |   '             \    *              -    #
   #            |   '        ;   \             ^,   `~_       _,.`     #
   #            |    ;        :,_ .              ^_    "~_ ~ ^         #
   #            '     ;           |                ^,     '-_          #
   #             \     -         ;                   "~_      " -,     #
   #####################################################################


   The LGA frame is coincident with the spacecraft XYZ axis, while the
   MGA secondary axis is chosen so that Y' coincides with spacecraft Y.
   This selection is identical to the axial thrusters A3,A4.

      [X]      [  1   0   0 ] [X']
      [Y]    = [  0   1   0 ] [Y']
      [Z]S/C   [  0   0   1 ] [Z']Low Gain Antenna

      [X]      [ -1   0   0 ] [X']
      [Y]    = [  0   1   0 ] [Y']
      [Z]S/C   [  0   0  -1 ] [Z']Medium Gain Antenna


   \begindata

   FRAME_IMAP_LOW_GAIN_ANTENNA = -43050
   FRAME_-43050_NAME           = 'IMAP_LOW_GAIN_ANTENNA'
   FRAME_-43050_CLASS          = 4
   FRAME_-43050_CLASS_ID       = -43050
   FRAME_-43050_CENTER         = -43
   TKFRAME_-43050_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43050_SPEC         = 'MATRIX'
   TKFRAME_-43050_MATRIX       = (  1,
                                    0,
                                    0,
                                    0,
                                    1,
                                    0,
                                    0,
                                    0,
                                    1  )

   FRAME_IMAP_MED_GAIN_ANTENNA = -43051
   FRAME_-43051_NAME           = 'IMAP_MED_GAIN_ANTENNA'
   FRAME_-43051_CLASS          = 4
   FRAME_-43051_CLASS_ID       = -43051
   FRAME_-43051_CENTER         = -43
   TKFRAME_-43051_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43051_SPEC         = 'MATRIX'
   TKFRAME_-43051_MATRIX       = ( -1,
                                    0,
                                    0,
                                    0,
                                    1,
                                    0,
                                    0,
                                    0,
                                   -1  )

   \begintext


IMAP-Lo Frames
========================================================================

   IMAP-Lo is a single-pixel energetic neutral atom (ENA) imager mounted
   on a pivot platform and equipped with a star sensor that pivots with
   the ENA sensor [12,13]. The instrument is mounted for imaging in the
   radial direction of the rotating spacecraft with the pivot allowing
   orientation of the boresight from a polar angle of 60° (slightly
   towards the Sun) to 180° (directed away from the Sun).


                       ---------
                      | +Z axis |
                       ---------               ---------------------
                           .                  | +X axis out of page |
                          /|\                  ---------------------
                           |      LGA
         __________________|______|^|_________                      ___
   SWAPI|__________________|__________________|====================|   |
    #|-|        |          |  .-==-,  |        /     MAG boom      '---'
    #|-| {|##   |          | /      \ |        |
       | {|##   |          |{  HI 90 }| IMAP LO|      _. IMAP LO
       | {|##   |   _.._   | \      / |  _.,   |  _.-'  BORESIGHT
       | ULTRA  | /      \ |  `-==-'  | / __`'_.-'
       |  90    | \ HI 45/ |          | \ \.-';|
       |        |  '----`  |          |  ~._.+ |
       '-------------------|----------/--------'
          |  |   \_________O_________/ | |        ---------------->
          |__|        -----------      /_\           ---------
          STAR       | S/C FRAME |     MGA          | +Y axis |
        TRACKERS     |  ORIGIN   |                   ---------
                      -----------


                             IMAP-Lo Local Frame

                 Pivot           +Z' axis
                    Angle             |
                            ,.~'^ ^ ^-|
                        .-'`          |
                      .`   _~-,  Star Sensor
                    .`    | ** \___ _ |
      Boresight  | /       \_-'`~~~~~~`'-.- -
             .   |/___    ,^~~~~~~%#####  ',  '.
               `'.       ^~~~~~~%%######### `   '.
                   `'.  /~~~~~~, - - ~~~#####\    .
                       /. ~~~ /       `.~~%###,    .
                      .~~~`'./          .~~###     .
                      .~~~~   `'.       |~~~%#"    .`.
                      "~~~~%|     O     :~~~~ '    .  .
                      |~~~ # .         /~~~~~ |    .  \
                      |~~~%##`.       /~~~~~ /     .  |
                       \~~%### ~`- -'~~~~~~ /     .   .
                       +,~%######~~~~~~~~ ,-   ~@@@~  .
                       | ' ~ ######%%%%_,^      ,~@@@~ Rotation Axis
                       '.    - .%##%.-        .'    . ^~.
                   .~ '` `.                 .'    .`      ^~.
             .~ '`         ' .         _ .'     .`            ^~.
       .~ '`                   ` '.''``      ,.`             +X' axis
    -Y' axis                       `-.,,, . `


   The local IMAP-Lo base frame is defined so the sensor pivots about
   the +X' axis. When the pivot angle is 90°, the boresight is aligned
   with the local -Y' axis. The +Z' axis, from which the pivot angle is
   measured, aligns with the spacecraft +Z axis.

   The boresight look-direction is defined for the azimuth-elevation:

      LO (azim, elev) = ( +330°, -90° to +30° )

   At 0° elevation (90° polar angle), the boresight direction and
   primary axis in the spacecraft frame of reference is:

      D = -Y' = [ -cos(0) x sin(330), cos(0) x cos(330), sin(0) ]

   The secondary axis is the +X' local axis, perpendicular to both
   the boresight direction D and the spacecraft -Z axis:

      S = +X' = D x -Z = Y' x [ 0, 0, 1 ]

   The tertiary axis is:

      N = D x S = Y' x ( Y' x [ 0, 0, 1 ] )

   The rotation matrix formed using the column vectors is:

      R = [ +S, -D, +N ]

   From the spacecraft MICD[6], the single-precision rotation matrices
   orienting IMAP-Lo on the spacecraft:

      [X]      [  -0.866025  -0.500000   0.000000  ] [X']
      [Y]    = [   0.500000  -0.866025   0.000000  ] [Y']
      [Z]S/C   [   0.000000   0.000000   1.000000  ] [Z']IMAP-Lo

   consistent with calculating the matrix R to single precision.

   For reference, the ZYZ intrinsic Euler angles orienting X'Y'Z' in
   the spacecraft XYZ coordinate system are:

      IMAP-Lo: (A, B, Y)  =  (  150.000°,   0.000°,  0.000°  )

   Using the formulas described in the Euler angles section above, the
   rotation matrix generated from these Euler angles is consistent with
   the rotation matrix using the azimuth/elevation look direction.


   IMAP-Lo Orientation
   ---------------------------------------------------------------------

   The orientation of IMAP-Lo must be specified in a separate C-kernel.
   To facilitate this specification, a base frame representing the fixed
   transformation of the local X'Y'Z' frame to the spacecraft frame has
   been provided.

   Ideally, the C-kernel will simply specify transformation within the
   local IMAP-Lo frame, and be generated using only the pivot angle.
   The implementation of this is outside the scope of this kernel.


   \begindata

   FRAME_IMAP_LO_BASE          = -43100
   FRAME_-43100_NAME           = 'IMAP_LO_BASE'
   FRAME_-43100_CLASS          = 4
   FRAME_-43100_CLASS_ID       = -43100
   FRAME_-43100_CENTER         = -43
   TKFRAME_-43100_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43100_SPEC         = 'MATRIX'
   TKFRAME_-43100_MATRIX       = ( -0.86602540378443865,
                                    0.50000000000000000,
                                    0.00000000000000000,
                                   -0.50000000000000000,
                                   -0.86602540378443865,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    0.00000000000000000,
                                    1.00000000000000000 )

   FRAME_IMAP_LO               = -43101
   FRAME_-43101_NAME           = 'IMAP_LO'
   FRAME_-43101_CLASS          = 3
   FRAME_-43101_CLASS_ID       = -43101
   FRAME_-43101_CENTER         = -43

   FRAME_IMAP_LO_STAR_SENSOR   = -43102
   FRAME_-43102_NAME           = 'IMAP_LO_STAR_SENSOR'
   FRAME_-43102_CLASS          = 4
   FRAME_-43102_CLASS_ID       = -43102
   FRAME_-43102_CENTER         = -43
   TKFRAME_-43102_RELATIVE     = 'IMAP_LO'
   TKFRAME_-43102_SPEC         = 'MATRIX'
   TKFRAME_-43102_MATRIX       = (  1,
                                    0,
                                    0,
                                    0,
                                    1,
                                    0,
                                    0,
                                    0,
                                    1 )

   \begintext


IMAP-Hi Frames
========================================================================

   IMAP-Hi consists of two identical, single-pixel high energy neutral
   atom (ENA) imagers. Hi 90 is oriented with its boresight
   perpendicular to the spacecraft spin axis, while Hi 45 is radially
   outward but with the boresight angled 45° from the -Z axis.

                       ---------
                      | +X axis |
                       ---------          --------------------
                           .             | +Z axis facing Sun |
       HI 45 BORESIGHT    /|\            |      into page     |
                  .        |        .     --------------------
                  " .~15°~.| .~15°~."
                   \       |       / HI 90 BORESIGHT
                    ,      |      ,
                     ;    /`~~__ ,             `+    direction of
                    , = .^ - /_  ``-.              '.    positive
                  .+     +     `^~/ ./                ~    rotation
                 ^ +     + . -- ' ``  \   _-~          \
            _   /   ',= '              \~'`   \ IMAP    \
    ULTRA /' '-_ .~ '                   \,.=.. \ LO    \|/
     90  /    ~          _,.,_         +      + \       '
        /  ,~'        +'       `'+     +      +  \
       / ~^        .'        , = .'.    '- =''  -``   ---------
       ^/         /  , = .  +     + \       \~'`     | +Y axis |----->
        |        .  +     + +     +  .       \        ---------     ___
        |        |  +     +  ' = '   |        \--------------------|   |
   SWAPI|        |   ' = ', - .      |        /--------------------|___|
     _+_:        '       +     +     '       /         MAG boom
    \_ __\__      \      +     +    /      /^*~,
      +  |  SWE    '.     ' = '   .'    ULTRA  /
          `~-'       '~..,___,..~'       45   /~,*
           _\                           /    /~,*`
        * / CODICE                      ^*._/  *` HIT
         *\ _/`.                       /
          *   / /~ _ _  ,.-^-., _ _ _ /
             '='       +       +
            GLOWS      +       +
                        '-.,.-'
                          IDEX


                       ---------
                      | +Z axis |
                       ---------               ---------------------
                           .                  | +X axis out of page |
                          /|\                  ---------------------
                           |      LGA
         __________________|______|^|_________                      ___
   SWAPI|__________________|__________________|====================|   |
    #|-|        |          |  .-==-,  |        /     MAG boom      '---'
    #|-| {|##   |          | /      \ |        |
       | {|##   |          |{  HI 90 }| IMAP LO|
       | {|##   |   _.._   | \      / |  _.,   |
       | ULTRA  | /      \ |  `-==-'  | / __`',|
       |  90    | \ HI 45/ |          | \ \_\ ;|
       |        |  '----`  |          |  ~._ + |
       '-------------------|----------/--------'
          |  |   \_________O_________/ | |        ---------------->
          |__|        -----------      /_\           ---------
          STAR       | S/C FRAME |     MGA          | +Y axis |
        TRACKERS     |  ORIGIN   |                   ---------
                      -----------


   #####################################################################
   #______________________________________________                     #
   #                   /  _       |        ||     |        IMAP HI 90  #
   #----~.            /  |_| O o  |        ||     |====      hidden    #
   #     ULTRA 90    /\   x  x =  |        ||     |        behind s/c  #
   #                /  \__________|        ||     || <---- struct here #
   #   ##### -- ####           |           ||     |]                   #
   #  ## % ## / \###\         /    ___     ||     |}  HI 90 Boresight  #
   # /## % ##\--|####|       |____|*#*|    ||     |}_________________  #
   # |## % ##|--|####|       |    |*#*|    ||     |}                   #
   # |## % ##|  |####|       |     ---     ||     |}                   #
   # |## % ##|--|####|       |             ||     |]      +Z           #
   # |## % ##|--|####|        \            ||     ||                   #
   # \## % ##/  |####|         |           ||     .        |           #
   #  ## % ## \ /###/          |           ||   .'.        |           #
   #   ##### -- ####           |           || .  /,        |           #
   #---------------            |           |.`   _~        x------ +X  #
   #                           |          ,`  ,~`  `~    +Y into       #
   #              ______      /         .`  ~`  _    \     page        #
   #  .=.=.=.=.  |( ) ()|    /     ___ *  -' ~' `',  |                 #
   #  | | | | |  |( ) ()|    |____|*#*|| ~ .`_ _  / ~                  #
   #__#_#_#_#_#__|______|         |*#*||` / //// / ~                   #
   #----------------|            .-----  /  ` ` ' ~                    #
   #  |_ _ _|   |   |          .'_   / '`.,_ ,~' ~.                    #
   #  | | | |   |   |        .'   -, |       _, `  ":.                 #
   #__/_/_/_/___/___|________|______;_\_ ,.-'         |:.              #
   #                                       | |       /   ":.           #
   #_______________________________________| |   _45°       ":.        #
   #___  ____  __ ||     ||     ||     |     |-~"              "       #
   #  / /   / / // |_____||_____||_____|     |                HI 45    #
   #_/ /___/ /_/ |                 /|\      \|/             Boresight  #
   #            /                            '                         #
   #            |                        ---------                     #
   #           /                        | -Z axis |                    #
   #===========                          ---------                     #
   #####################################################################


   The local IMAP-Hi frame[15]--identical for both sensors--is defined
   with the boresight aligned with the +Y' axis, the rectangular vent
   ports aligned with the +Z' axis, and X' = Y' x Z'.


   IMAP HI 45
   --------------

   The boresight look-direction is defined for the azimuth-elevation:

      HI 45 (azim, elev) = ( +255°, -45° )

   The boresight direction is the +Y' local axis of instrument, and the
   primary axis in the spacecraft frame of reference is:

      D = +Y' = [ -cos(-45) x sin(255), cos(-45) x cos(255), sin(-45) ]

   The secondary axis is the +Z' local axis, NOTIONALLY perpendicular to
   both the boresight direction D and the spacecraft Z axis:

      S = +Z' = D x Z = Y' x [ 0, 0, 1 ]

   The tertiary axis is NOTIONALLY:

      N = D x S = Y' x ( Y' x [ 0, 0, 1 ] )

   The rotation matrix formed using the column vectors is NOTIONALLY:

      RN = [ +N, +D, +S ]

   HOWEVER, the actual alignment is modified by a rotation about the
   local Y' axis by 3° as a consequence of the angular offset of the
   mounting inserts by the same amount. This rotation about local Y' is:

            [  cos(3)   0   sin(3) ]
      RY' = [   0       1     0    ]
            [ -sin(3)   0   cos(3) ]

   The final rotation that orients HI 45 on the spacecraft is the matrix
   multiplication:

      R = RN x RY'

   From the spacecraft MICD[6], the single-precision rotation matrices
   orienting IMAP-HI 45 on the spacecraft:

      [X]      [  -0.668531   0.683013  -0.294210  ] [X']
      [Y]    = [   0.233315  -0.183013  -0.955024  ] [Y']
      [Z]S/C   [  -0.706183  -0.707107  -0.037007  ] [Z']HI 45

   Using the method described in a Euler discussion section, the Euler
   angles rounded to 1/1000th of a degree are:

      HI 45:  (A, B, Y)  =  (  -17.122°,  92.121°, -135.037°  )

   Using the formulas described in the Euler angles section above, the
   rotation matrix generated from these Euler angles is consistent with
   the rotation matrix using the azimuth/elevation look direction;
   however, the full double-precision Euler angles are necessary to
   generate the proper precise rotation matrix.


   \begindata

   FRAME_IMAP_HI_45            = -43150
   FRAME_-43150_NAME           = 'IMAP_HI_45'
   FRAME_-43150_CLASS          = 4
   FRAME_-43150_CLASS_ID       = -43150
   FRAME_-43150_CENTER         = -43
   TKFRAME_-43150_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43150_SPEC         = 'MATRIX'
   TKFRAME_-43150_MATRIX       = ( -0.66853111450276550,
                                    0.23331454112339850,
                                   -0.70613771591812640,
                                    0.68301270189221940,
                                   -0.18301270189221924,
                                   -0.70710678118654750,
                                   -0.29421046547595930,
                                   -0.95502391375634550,
                                   -0.03700710955926802 )

   \begintext


   IMAP HI 90
   --------------

   The boresight look-direction is defined for the azimuth-elevation:

      HI 90 (azim, elev) = ( +285°, 0° )

   The boresight direction is the +Y' local axis of instrument, and the
   primary axis in the spacecraft frame of reference is:

      D = +Y' = [ -cos(0) x sin(285), cos(0) x cos(285), sin(0) ]

   The secondary axis is the +Z' local axis, NOTIONALLY perpendicular to
   both the boresight direction D and the spacecraft Z axis:

      S = -Z' = D x Z = -Y' x [ 0, 0, 1 ]

   The tertiary axis is NOTIONALLY:

      N = D x S = Y' x ( Y' x [ 0, 0, 1 ] )

   The rotation matrix formed using the column vectors is NOTIONALLY:

      RN = [ +N, +D, +S ]

   HOWEVER, the actual alignment is modified by a rotation about the
   local Y' axis by 15° as a consequence of the angular offset of the
   mounting inserts by the same amount. This rotation about local Y' is:

            [  cos(15)   0   sin(15) ]
      RY' = [    0       1      0    ]
            [ -sin(15)   0   cos(15) ]

   The final rotation that orients HI 45 on the spacecraft is the matrix
   multiplication:

      R = RN x RY'

   From the spacecraft MICD[6], the single-precision rotation matrices
   orienting IMAP-HI 45 on the spacecraft:

      [X]      [   0.066987   0.965926  -0.250000  ] [X']
      [Y]    = [  -0.250000   0.258819   0.933013  ] [Y']
      [Z]S/C   [   0.965926   0.000000   0.258819  ] [Z']HI 90

   Using the method described in a Euler discussion section, the Euler
   angles rounded to 1/1000th of a degree are:

      HI 90:  (A, B, Y)  =  ( -165.000°,  75.000°,   90.000°  )

   Using the formulas described in the Euler angles section above, the
   rotation matrix generated from these Euler angles is consistent with
   the rotation matrix using the azimuth/elevation look direction.


   \begindata

   FRAME_IMAP_HI_90            = -43151
   FRAME_-43151_NAME           = 'IMAP_HI_90'
   FRAME_-43151_CLASS          = 4
   FRAME_-43151_CLASS_ID       = -43151
   FRAME_-43151_CENTER         = -43
   TKFRAME_-43151_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43151_SPEC         = 'MATRIX'
   TKFRAME_-43151_MATRIX       = (  0.06698729810778055,
                                   -0.25000000000000000,
                                    0.96592582628906829,
                                    0.96592582628906829,
                                    0.25881904510252074,
                                    0.00000000000000000,
                                   -0.25000000000000000,
                                    0.93301270189221940,
                                    0.25881904510252074 )

   \begintext


IMAP-Ultra Frames
========================================================================

   The IMAP-Ultra instrument[7,14] consists of two identical sensors for
   imaging the emission of energetic neural atoms (ENAs) produced in the
   heliosheath and beyond. Ultra 90 is mounted perpendicular to the IMAP
   spin axis (+Z), while Ultra 45 is mounted at 45 degrees from the
   anti-sunward spin axis (-Z).


                       ---------
                      | +X axis |         --------------------
                       ---------         | +Z axis facing Sun |
                           .             |      into page     |
                          /|\             --------------------
                           |
                           |
                           |
                                              _
                  HI 45    /`~~__HI 90          `+    direction of
                    , = .^ - /_  ``-.              '.    positive
                  .+     +     `^~/ ./                ~    rotation
     ULTRA       ^ +     + . -- ' ``  \   _-~          \
      90    _   /   ',= '              \~'`   \ IMAP    \
   .      /' '-_ .~ '                   \,.=.. \ LO    \|/
    `;.  /    ~          _,.,_         +      + \       '
    /  `/  ,~'        +'       `'+     +      +  \
   30° / ~^        .'        , = .'.    '- =''  -``   ---------
   |   ^/         /  , = .  +     + \       \~'`     | +Y axis |----->
   ----  |        .  +     + +     +  .       \        ---------     ___
        |        |  +     +  ' = '   |        \--------------------|   |
   SWAPI|        |   ' = ', - .      |        /--------------------|___|
     _+_:        '       +     +     '       /        |      MAG boom
    \_ __\__      \      +     +    /      /^*~,      .
      +  |  SWE    '.     ' = '   .'    ULTRA  /     33°
          `~-'       '~..,___,..~'       45   /       ;
           _\                           /    /`.,    /
        * / CODICE                      ^*._/    `'./
         *\ _/`.                       /             `'.
          *   / /~ _ _  ,.-^-., _ _ _ /
             '='       +       +
            GLOWS      +       +
                        '-.,.-'
                          IDEX


   Each sensor comprises two separate assemblies of collector plates.
   Each assembly of collector plates is fanned out in a cylindrical
   pattern, and the cyclindrical axes of the fanned-out plates are
   parallel and offset in the direction perpendicular to the axes.

   The orientations of Ultra 45 and 90 are analogous to IMAP Hi 45 and
   90; see the diagram for IMAP Hi above. Take special note that the
   angle with the spacecraft Z axis and the boresights for IMAP Hi are
   the same as the angle with the spacecraft Z axis and the "outward"
   directions for Ultra.


   #####################################################################
   #                                                                   #
   #             One half of one IMAP Ultra sensor showing             #
   #              assembly of fanned-out collector plates              #
   # Outward                                                           #
   #    .             Assemblies are mirror-symmetric                  #
   #   /|\           about the leftmost edge of drawing                #
   #    |                                                              #
   #    |     ,--.        ,                                            #
   #          || |   |    ;        ,          63.42° FOR               #
   #    |      | |   |    ;   ;    ;          60.31° FOV               #
   #    |      | |   |    :   ;   ;    ;    ;                          #
   #   \|/     | |   |   :   ;    ;   ;    /                           #
   #    '      | ;_ _|_  ;   ;   ;    ;   /    /    .                  #
   #   S/C     | |      ``'''^-,/,   /   /    /    .'                  #
   #           | | ___            `''., /    /    .    .               #
   #_________;-|__|_  `'"^~-,._ /^~    `^., /   ,'    '     .          #
   #---------'- | |_|          `'":.,    _ `^, .    ,'    .'           #
   #--------. ,-| |    @             `'~/  \  `'. .`    .'    ,        #
   #     ,'`.'  | | @   @ @  @         '~,.;,    '.   .'    .'         #
   #   .','     | |  @    @  @  @  @        `;, /~_':'    .'     ,'    #
   #.'`,'      _| |_            @  @     @    `;, /  '. .'    ,'`      #
   # `,       |_|-|_|                @  @        '.    '.   .'    ,.   #
   #'-,'.       |-|                 @     @   @    ',    `.`    ;'     #
   #   '.'-.    | |                      @   @       ;,    \,;`'    .-`#
   #     `-.'.  | |                            @  @    ",   ',  ,.'` ,^#
   #_________:'-| | @   @                     @  @      :,  _,\' .-``  #
   #-----------||-|-,      @                          /~,".' ;'.'      #
   #===        ||---|@  @     @                 @ @  ,\ '  ,.^`        #
   #___________||_/-~_ _   @     @           _,  _,-' ,.-'`            #
   #   @| | |  ||       `- , @      @        \,\'  ,'`     +Z'         #
   #----' | |  ||           `~,@       @  ,~`' _,'`                    #
   #      | |  ||              ',@ .^\_,'` ,.'`             |          #
   #______'-'__||-@--~-~,        \ .;`  .'`                 |          #
   #___________||/ ~  #  ~        | {.'`                    |          #
   #        |* ||*|   +  <------------  Collector plate     +------ +Y'#
   #____     --||\ ~     ~       |      axis of symmetry   +X' out     #
   #_  *|      ||-@-^~-~^-------|                          of page     #
   #*| |       ||_______________|                                      #
   #___*|_______|_|_|__|__|_|_| |                                      #
   #####################################################################


   The local IMAP-Ultra frame[14]--identical for both sensors--is
   defined with the collector-plate-fan axes of symmetry aligned with
   the +X' axis, the cylindrical axes offset in the +Y' axis, and the
   Z' axis perpendicular to both and outward as in the diagram below.


   IMAP ULTRA 45
   --------------

   The outward look-direction is defined for the azimuth-elevation:

      ULTRA 45 (azim, elev) = ( +33°, -45° )

   The look-direction is the +Z' local axis of instrument, and the
   primary axis in the spacecraft frame of reference is:

      D = +Z' = [ -cos(-45) x sin(33), cos(-45) x cos(33), sin(-45) ]

   The secondary axis is the +X' local axis, lying in the plane spanned
   by the look-direction D and the spacecraft Z axis. An equivalent
   definition is selecting the secondary axis as the +Y' local axis,
   perpendicular to both the look-direction D and the spacecraft Z axis.

      S = +Y' = D x Z = Z' x [ 0, 0, 1 ]

   The tertiary axis is:

      N = D x S = Z' x Y' = Z' x ( Z' x [ 0, 0, 1 ] )

   The rotation matrix formed using the column vectors is:

      R = [ -N, +S, +D ]

   The rotation matrices orienting the IMAP-Ultra 45 sensor on the
   spacecraft is given by [6]:

      [X]      [  -0.385118   0.838671  -0.385118  ] [X']
      [Y]    = [   0.593030   0.544639   0.593030  ] [Y']
      [Z]S/C   [   0.707107   0.000000  -0.707107  ] [Z']ULTRA 45

   Using the method described in a Euler discussion section, the Euler
   angles rounded to 1/1000th of a degree are:

      ULTRA 45:  (A, B, Y)  =  ( -147.000°, 135.000°,   90.000°  )

   Using the formulas described in the Euler angles section above, the
   rotation matrix generated from these Euler angles is consistent with
   the rotation matrix using the azimuth/elevation look direction.


   \begindata

   FRAME_IMAP_ULTRA_45         = -43200
   FRAME_-43200_NAME           = 'IMAP_ULTRA_45'
   FRAME_-43200_CLASS          = 4
   FRAME_-43200_CLASS_ID       = -43200
   FRAME_-43200_CENTER         = -43
   TKFRAME_-43200_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43200_SPEC         = 'MATRIX'
   TKFRAME_-43200_MATRIX       = ( -0.38511795495802310,
                                    0.59302964577578240,
                                    0.70710678118654760,
                                    0.83867056794542390,
                                    0.54463903501502710,
                                    0.00000000000000000,
                                   -0.38511795495802320,
                                    0.59302964577578250,
                                   -0.70710678118654750 )

   \begintext


   IMAP ULTRA 90
   --------------

   The outward look-direction is defined for the azimuth-elevation:

      ULTRA 90 (azim, elev) = ( +210°, 0° )

   The look-direction is the +Z' local axis of instrument, and the
   primary axis in the spacecraft frame of reference is:

      D = +Z' = [ -cos(0) x sin(210), cos(0) x cos(210), sin(0) ]

   The secondary axis is the +X' local axis, lying along spacecraft
   -Z axis.

      S = +X' = +Z = [ 0, 0, 1 ]

   The tertiary axis is:

      N = D x S = Z' x X' = Z' x [ 0, 0, 1 ]

   The rotation matrix formed using the column vectors is:

      R = [ +S, +N, +D ]

  The rotation matrices orienting the IMAP-Ultra 90 sensor on the
   spacecraft is given by [6]:

      [X]      [   0.000000  -0.866025   0.500000  ] [X']
      [Y]    = [   0.000000  -0.500000  -0.866025  ] [Y']
      [Z]S/C   [   1.000000   0.000000   0.000000  ] [Z']ULTRA 90

   Using the method described in a Euler discussion section, the Euler
   angles rounded to 1/1000th of a degree are:

      ULTRA 90:  (A, B, Y)  =  (   30.000°,  90.000°,   90.000°  )

   Using the formulas described in the Euler angles section above, the
   rotation matrix generated from these Euler angles is consistent with
   the rotation matrix using the azimuth/elevation look direction.


   \begindata

   FRAME_IMAP_ULTRA_90         = -43201
   FRAME_-43201_NAME           = 'IMAP_ULTRA_90'
   FRAME_-43201_CLASS          = 4
   FRAME_-43201_CLASS_ID       = -43201
   FRAME_-43201_CENTER         = -43
   TKFRAME_-43201_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43201_SPEC         = 'MATRIX'
   TKFRAME_-43201_MATRIX       = (  0.00000000000000000,
                                    0.00000000000000000,
                                    1.00000000000000000,
                                   -0.86602540378443865,
                                   -0.50000000000000000,
                                    0.00000000000000006,
                                    0.50000000000000000,
                                   -0.86602540378443865,
                                    0.00000000000000000 )

   \begintext


IMAP Magnetometer (MAG) Frames
========================================================================

   The IMAP magnetometer (MAG)[7,16] consists of a pair of identical
   triaxial fluxgate magnetometers mounted on a ~2.5 meter boom. MAG-O
   is positioned at the end of the boom, while MAG-I is mounted ~0.75
   meters from MAG-O.


                       ---------
                      | +X axis |         --------------------
                       ---------         | +Z axis facing Sun |
                           .             |      into page     |
                          /|\             --------------------
                           |
                           |
                           |
                                              _
                  HI 45    /`~~__HI 90          `+    direction of
                    , = .^ - /_  ``-.              '.    positive
                  .+     +     `^~/ ./                ~    rotation
                 ^ +     + . -- ' ``  \   _-~          \
            _   /   ',= '              \~'`   \ IMAP    \
    ULTRA /' '-_ .~ '                   \,.=.. \ LO    \|/
     90  /    ~          _,.,_         +      + \       '
        /  ,~'        +'       `'+     +      +  \
       / ~^        .'        , = .'.    '- =''  -``
       ^/         /  , = .  +     + \       \~'`       +Y axis  ----->
        |        .  +     + +     +  .       \              ___     ___
        |        |  +     +  ' = '   |        \------------|   |---|   |
   SWAPI|        |   ' = ', - .      |        /------------|___|---|___|
     _+_:        '       +     +     '       /             MAG-I   MAG-O
    \_ __\__      \      +     +    /      /^*~,
      +  |  SWE    '.     ' = '   .'    ULTRA  /           MAGS and boom
          `~-'       '~..,___,..~'       45   /~,*          not to scale
           _\                           /    /~,*`
        * / CODICE                      ^*._/  *` HIT
         *\ _/`.                       /
          *   / /~ _ _  ,.-^-., _ _ _ /
             '='       +       +
            GLOWS      +       +
                        '-.,.-'
                          IDEX


                     ----------------------------
   S/C +Z axis      | Deployed Magnetometer Boom |         S/C +X axis
    .               |  (approximately to scale)  |         out of page
   /|\               ----------------------------
    |
    |                      S/C +Y axis -------->
    @=================================================================
    #\                                             | |             | |
      \                                            `'`             `'`
       Boom Deployment Hinge                      MAG-I           MAG-O

                                          +X' ------x   +Y' into
                                                    |     page
                                      MAG Local     |
                                    Coord System    |

                                                   +Z'


   Each MAG instrument is contained in a cylindrial casing with the
   local Z' axis along the cylindrical axis of symmetry. The local X'
   axis is along the boom, and the local Y' axis is perp to the boom.

   When deployed, the boom sticks out in the +Y axis of the spacecraft,
   with the MAG +X' axis in the -Y direction. The MAG +Z' axis is in the
   spacecraft -Z' direction, and +Y' is spacecraft -X.

      [X]      [   0  -1   0  ] [X']
      [Y]    = [  -1   0   0  ] [Y']
      [Z]S/C   [   0   0  -1  ] [Z']MAG deployed

   Prior to deployment, the boom is stowed pointing in the -Y direction
   of the spacecraft, with the MAG +X' axis in the +Y direction. The MAG
   +Z' axis is in the spacecraft +Z' direction, and +Y' is spacecraft -X

      [X]      [   0  +1   0  ] [X']
      [Y]    = [  -1   0   0  ] [Y']
      [Z]S/C   [   0   0  +1  ] [Z']MAG undeployed

   To facilitate possible operations prior to the boom deployment, a
   frame for the deployed boom is provided; the MAG-I and MAG-O frames
   are provided relative to this frame. If needed, the IMAP_MAG_BOOM
   can be modified to facilitate arbitrary operational reality.


   \begindata

   FRAME_IMAP_MAG_BOOM         = -43250
   FRAME_-43250_NAME           = 'IMAP_MAG_BOOM'
   FRAME_-43250_CLASS          = 4
   FRAME_-43250_CLASS_ID       = -43250
   FRAME_-43250_CENTER         = -43
   TKFRAME_-43250_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43250_SPEC         = 'MATRIX'
   TKFRAME_-43250_MATRIX       = (  1,
                                    0,
                                    0,
                                    0,
                                    1,
                                    0,
                                    0,
                                    0,
                                    1 )

   FRAME_IMAP_MAG_I            = -43251
   FRAME_-43251_NAME           = 'IMAP_MAG_I'
   FRAME_-43251_CLASS          = 4
   FRAME_-43251_CLASS_ID       = -43251
   FRAME_-43251_CENTER         = -43
   TKFRAME_-43251_RELATIVE     = 'IMAP_MAG_BOOM'
   TKFRAME_-43251_SPEC         = 'MATRIX'
   TKFRAME_-43251_MATRIX       = (  0,
                                   -1,
                                    0,
                                   -1,
                                    0,
                                    0,
                                    0,
                                    0,
                                   -1 )

   FRAME_IMAP_MAG_O            = -43252
   FRAME_-43252_NAME           = 'IMAP_MAG_O'
   FRAME_-43252_CLASS          = 4
   FRAME_-43252_CLASS_ID       = -43252
   FRAME_-43252_CENTER         = -43
   TKFRAME_-43252_RELATIVE     = 'IMAP_MAG_BOOM'
   TKFRAME_-43252_SPEC         = 'MATRIX'
   TKFRAME_-43252_MATRIX       = (  0,
                                   -1,
                                    0,
                                   -1,
                                    0,
                                    0,
                                    0,
                                    0,
                                   -1 )

   \begintext


IMAP Solar Wind Electron (SWE) Frames
========================================================================

   TODO: FIX ME...The orientation of the spacecraft body frame with
   respect to an inertial
   frame, for IMAP - ECLIPJ2000, is provided by a C-kernel (see [3]
   for details).

   This frame specifies the rotating X,Y and pointing Z coordinate body
   frame.

   \begindata

   FRAME_IMAP_SWE              = -43300
   FRAME_-43300_NAME           = 'IMAP_SWE'
   FRAME_-43300_CLASS          = 4
   FRAME_-43300_CLASS_ID       = -43300
   FRAME_-43300_CENTER         = -43
   TKFRAME_-43300_SPEC         = 'MATRIX'
   TKFRAME_-43300_MATRIX       = (  0.453990,
                                    0.891007,
                                    0.000000,
                                   -0.891007,
                                    0.453990,
                                    0.000000,
                                    0.000000,
                                    0.000000,
                                    1.000000 )
   TKFRAME_-43300_RELATIVE     = 'IMAP_SPACECRAFT'

\begintext


IMAP Solar Wind and Pickup Ion (SWAPI) Frames
========================================================================

   TODO: add diagrams

   SWAPI has the following nominal alignment to the spacecraft frame,
   reference Table 1 of [6]. The azimuth and elevation angles are
   illustrated in the 'IMAP I&T Component Placement' section near the top
   of this document.

        azimuth  | elevation
         (deg)   |  (deg)
        ---------+---------
          168    |   0

   The SWAPI base frame is defined in the instrument MICD [8] as follows:

      *  -Z axis is the axis of symmetry of the instrument, pointing
            away from the spacecraft body.
      *  +Y axis is along the aperture center, in the anti-sunward direction.

   The azimuth and elevation give the outward axis of symmetry, -Z in the
   instrument frame:

      -Z      = -[ -sin(az) * cos(el), cos(az) * cos(el), sin(el) ]
        instr

   The instrument +Y axis is in the sunward direction, towards the
   spacecraft +Z axis:

       Y      = [ 0 0 1 ]
        instr

   Taking the cross product and normalizing, we arrive at the instrumet +X
   axis:
                  Y x Z
       X      = ---------
        instr   | Y x Z |

   And adjusting Y:

                  Z x X
       Y      = ---------
        instr   | Z x X |

    This definition is captured in the keywords below.

   \begindata

   FRAME_IMAP_SWAPI              = -43350
   FRAME_-43350_NAME           = 'IMAP_SWAPI'
   FRAME_-43350_CLASS          = 4
   FRAME_-43350_CLASS_ID       = -43350
   FRAME_-43350_CENTER         = -43
   TKFRAME_-43350_SPEC         = 'MATRIX'
   TKFRAME_-43350_MATRIX       = ( -0.97814760073381,
                                    0.20791169081776,
                                    0.00000000000000,
                                    0.00000000000000,
                                    0.00000000000000,
                                    1.00000000000000,
                                    0.20791169081776,
                                    0.97814760073381,
                                    0.00000000000000 )
   TKFRAME_-43350_RELATIVE     = 'IMAP_SPACECRAFT'

\begintext


IMAP Compact Dual Ion Composition Experiment (CoDICE) Frames
========================================================================

   TODO: FIX ME...The orientation of the spacecraft body frame with
   respect to an inertial
   frame, for IMAP - ECLIPJ2000, is provided by a C-kernel (see [3]
   for details).

   This frame specifies the rotating X,Y and pointing Z coordinate body
   frame.

   \begindata

   FRAME_IMAP_CODICE              = -43400
   FRAME_-43400_NAME           = 'IMAP_CODICE'
   FRAME_-43400_CLASS          = 4
   FRAME_-43400_CLASS_ID       = -43400
   FRAME_-43400_CENTER         = -43
   TKFRAME_-43400_SPEC         = 'MATRIX'
   TKFRAME_-43400_MATRIX       = (  0.694626,
                                    0.719371,
                                    0.000000,
                                   -0.719371,
                                    0.694626,
                                    0.000000,
                                    0.000000,
                                    0.000000,
                                    1.000000 )
   TKFRAME_-43400_RELATIVE     = 'IMAP_SPACECRAFT'

\begintext


IMAP High-energy Ion Telescope (HIT) Frames
========================================================================

   TODO: FIX ME...The orientation of the spacecraft body frame with
   respect to an inertial
   frame, for IMAP - ECLIPJ2000, is provided by a C-kernel (see [3]
   for details).

   This frame specifies the rotating X,Y and pointing Z coordinate body
   frame.

   \begindata

   FRAME_IMAP_HIT              = -43500
   FRAME_-43500_NAME           = 'IMAP_HIT'
   FRAME_-43500_CLASS          = 4
   FRAME_-43500_CLASS_ID       = -43500
   FRAME_-43500_CENTER         = -43
   TKFRAME_-43500_SPEC         = 'MATRIX'
   TKFRAME_-43500_MATRIX       = (  0.866025,
                                    0.500000,
                                    0.000000,
                                   -0.500000,
                                    0.866025,
                                    0.000000,
                                    0.000000,
                                    0.000000,
                                    1.000000 )
   TKFRAME_-43500_RELATIVE     = 'IMAP_SPACECRAFT'

\begintext


IMAP Interstellar Dust Experiment (IDEX) Frames
========================================================================

   TODO: FIX ME...The orientation of the spacecraft body frame with
   respect to an inertial
   frame, for IMAP - ECLIPJ2000, is provided by a C-kernel (see [3]
   for details).

   This frame specifies the rotating X,Y and pointing Z coordinate body
   frame.

   \begindata

   FRAME_IMAP_IDEX              = -43700
   FRAME_-43700_NAME           = 'IMAP_IDEX'
   FRAME_-43700_CLASS          = 4
   FRAME_-43700_CLASS_ID       = -43700
   FRAME_-43700_CENTER         = -43
   TKFRAME_-43700_SPEC         = 'MATRIX'
   TKFRAME_-43700_MATRIX       = (  0.000000,
                                    1.000000,
                                    0.000000,
                                   -0.707107,
                                    0.000000,
                                   -0.707107,
                                   -0.707107,
                                    0.000000,
                                    0.707107 )
   TKFRAME_-43700_RELATIVE     = 'IMAP_SPACECRAFT'

\begintext


IMAP GLObal solar Wind Structure (GLOWS) Frames
========================================================================

   TODO: add diagrams

   GLOWS has the following nominal alignment to the spacecraft frame,
   reference Table 1 of [6]. The azimuth and elevation angles are
   illustrated in the 'IMAP I&T Component Placement' section near the top
   of this document.

        azimuth  | elevation
         (deg)   |  (deg)
        ---------+---------
          127    |   15

   The GLOWS base frame is defined by the instrument team as follows [10]:

      *  +Z axis points in the anti-boresight direction
      *  +Y axis points in the anti-sunward direction.

   The azimuth and elevation give the outward axis of symmetry, -Z in the
   instrument frame:

       Z      = -[ -sin(az) * cos(el), cos(az) * cos(el), sin(el) ]
        instr

   The instrument +Y axis is in the anti-sunward direction, towards the
   spacecraft -Z axis:

       Y      = [ 0 0 -1 ]
        instr

   Taking the cross product and normalizing, we arrive at the instrumet +X
   axis:
                  Y x Z
       X      = ---------
        instr   | Y x Z |

   And adjusting Y:

                  Z x X
       Y      = ---------
        instr   | Z x X |

    This definition is captured in the keywords below.

   \begindata

   FRAME_IMAP_GLOWS              = -43751
   FRAME_-43751_NAME           = 'IMAP_GLOWS'
   FRAME_-43751_CLASS          = 4
   FRAME_-43751_CLASS_ID       = -43751
   FRAME_-43751_CENTER         = -43
   TKFRAME_-43751_SPEC         = 'MATRIX'
   TKFRAME_-43751_MATRIX       = (  0.60181502315205,
                                   -0.79863551004729,
                                    0.00000000000000,
                                   -0.20670208009540,
                                   -0.15576118962056,
                                   -0.96592582628907,
                                    0.77142266494622,
                                    0.58130867351132,
                                   -0.25881904510252 )
   TKFRAME_-43751_RELATIVE     = 'IMAP_SPACECRAFT'

\begintext

   Generic axis

                                  +Z axis
                                     |
                                     |
                                     |
                                     |
                                     |
                                     |
                                     |
                                     |
                                     |
                                     |
                                     |
                                     |
                                   .~ ~
                             .~ '`      ^~
                       .~ '`               ^~
                 .~ '`                        ^~
           .~ '`                                 ^~
      +X axis                                       ^~
                                                       ^~
                                                          ^~
                                                         +Y axis

End of FK file.