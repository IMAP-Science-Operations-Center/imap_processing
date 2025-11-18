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

   TEXT_KERNEL_ID += 'IMAP_FRAMES V1.3.0 2025-NOV-13 FK'
   
   \begintext
   

   Version 1.3.0 -- Nov 13, 2025 -- Lillian Nguyen
   
      Inserted a nominal base frame for MAG.
      Corrected frame name to ID mapping for HI-90, ULTRA-90, MAG-O.
	  (Note: Release version number 1.1.0 was inadvertently skipped.)

   Version 1.2.0 -- Oct 21, 2025 -- Lillian Nguyen
   
      Updated SWAPI frame with launch site alignments.
      Added instrument coordinate system diagrams for SWAPI, CoDICE, and GLOWS.
      Removed unimplemented SWAPI and CODICE aperture frame IDs.

   Version 1.0.0 -- Sept 19, 2025 -- Douglas Rodgers
                                     Lillian Nguyen
                                     Nicholas Dutton
                                   
      This release includes:
         -Launch site alignment for LO, MAG, SWE, GLOWS, ULTRA, HI, HIT, CODICE.
         -Nominal (ideal) alignments for SWAPI.
      Frames not yet implemented:
         -SWAPI Apertures and Sunglasses
         -CODICE Apertures
      
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
   
   17. "Launch Site Alignments Report", Anthony Fanelli, Aug. 7, 2025
   
   18. "IMAP-SWE INSTRUMENT MICD", Drawing No. CN102M-i0000, Rev A, 
       Los Alamos National Laboratory
       
   19. https://imap.princeton.edu/spacecraft/instruments/solar-wind-electron-swe/
       swe-technical-overview
       
   20. "IMAP-HI SENSOR HEAD MICD", Drawing No. CN106M-i0000, Rev A, 
       Los Alamos National Laboratory
       
   21. “HIT TOP LEVEL ASSY MICD”, Drawing No. 2309580, Rev A, Goddard Space 
       Flight Center
   
   22. "IDEX, MECHANICAL INTERFACE CONTROL DOCUMENT (MICD)", Doc Num 165014, 
       Rev E, Laboratory for Lunar and Space Physics
       
   23. IMAP CODICE MICD, Drawing No. 268503001, Rev. C, Southwest Research
       Institute
       

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
      NAIF_BODY_CODE   += ( -43151       )

      NAIF_BODY_NAME   += ( 'IMAP_ULTRA_45' )
      NAIF_BODY_CODE   += ( -43200          )
      
      NAIF_BODY_NAME   += ( 'IMAP_ULTRA_90' )
      NAIF_BODY_CODE   += ( -43201          )

      NAIF_BODY_NAME   += ( 'IMAP_MAG_BOOM' )
      NAIF_BODY_CODE   += ( -43250          )
      
      NAIF_BODY_NAME   += ( 'IMAP_MAG_I' )
      NAIF_BODY_CODE   += ( -43251       )

      NAIF_BODY_NAME   += ( 'IMAP_MAG_O' )
      NAIF_BODY_CODE   += ( -43252       )

      NAIF_BODY_NAME   += ( 'IMAP_MAG_BASE' )
      NAIF_BODY_CODE   += ( -43253          )
      
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
      NAIF_BODY_CODE   += ( -43350      )
      
      NAIF_BODY_NAME   += ( 'IMAP_CODICE' )
      NAIF_BODY_CODE   += ( -43400        )
            
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
      NAIF_BODY_CODE   += ( -43750       )

   \begintext


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
      IMAP_MAG_BASE                 IMAP_SPACECRAFT    FIXED     -43253
      IMAP_MAG_I                    IMAP_MAG_BASE      FIXED     -43251
      IMAP_MAG_O                    IMAP_MAG_BASE      FIXED     -43252

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

      CODICE (400-499)
      --------------------------
      IMAP_CODICE                   IMAP_SPACECRAFT    FIXED     -43400

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
      IMAP_GLOWS                    IMAP_SPACECRAFT    FIXED     -43750


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
           |    
           IMAP_MAG_BASE
           |    |
           |    IMAP_MAG_I
           |    |
           |    IMAP_MAG_O
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
           |
           IMAP_CODICE
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
   

   IMAP -Z Bottom View (Figure 3-2 in [5], G-G in [6] rotated 180 deg)
   ---------------------------------------------------------------------
                       -------------
                      | S/C +X axis |        -----------------------
                       -------------        | S/C +Z axis into page |
                           .                |     (facing Sun)      |
                          /|\                -----------------------
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
       / ~^        .'        , = .'.    '- =''  -``   -------------     
       ^/         /  , = .  +     + \       \~'`     | S/C +Y axis |-----> 
        |        .  +     + +     +  .       \        ------------- ___
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
                      -------------
                     | S/C +Z axis |
                      -------------            -------------------------
                           .                  | S/C +X axis out of page |
                          /|\                  -------------------------
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
          |__|        -----------       /_\           -------------
          STAR       | S/C FRAME |      MGA          | S/C +Y axis |
        TRACKERS     |  ORIGIN   |                    -------------
                      -----------

                                
   IMAP -X Side View (C-C in [6])
   ---------------------------------------------------------------------
                                         -------------
                                        | S/C +Z axis |
        -----------------------          -------------
       | S/C +X axis into page |               .
        -----------------------               /|\
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
             -------------       /_\      -----------        |__|
            | S/C +Y axis |      MGA     | S/C FRAME |       STAR
             -------------               |  ORIGIN   |     TRACKERS
                                          -----------

   
   IMAP Component Location - Nominal Azimuth and Elevation
   ---------------------------------------------------------------------
   
   Payload and subsystem component locations are specified [5,6] in the
   Observatory Mechanical Design Reference Frame (described above).
   Locations are defined in azimuth and elevation (and resultant
   direction cosine matrices) of these angles [6] in the same reference
   frame. The azimuth and elevation angle diagram is provided below.
   
   In general, descriptions in this kernel treat the +Z direction as
   "up" and the -Z direction as "down." Locations referred to as "above"
   are generally closer to the Sun, and vice versa for "below." The
   "upper" side of the spacecraft is the plane of the solar panels,
   while the "lower" side may refer to the area near the adapter ring.
   If ambiguity could arise, more thorough descriptions will be used.
    
    
                                Toward Sun
   
                                 S/C +Z axis                                
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
                 .~ '` Boom     S/C   .  .` \  |
           .~ '`               FRAME |.`    : |
     / ~'`                    ORIGIN O      | |
    *---                     .~ '`    \  Elevation
                       .~ '`           \    | |
                 .~ '`                  \   ; |~
           .~ '\                         \ /  |  ^~
     S/C +Y axis \    \                    +   |     ^~
                 '.   '~,                  \  |        ^~
                   '~     Azimuth           \ |           ^~
                      '~.         `^~->      \|         S/C -X axis
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
   
   The instrument teams have defined the primary and secondary axes of
   the instrument-specific coordinate frames in [10]. Those definitions
   are described in the instrument-specific sections that follow. When
   a coordinate system has not been defined by the team, one is chosen
   in a convenient manner.

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
   
   
                             Location Azim         Direction Azim
                        -------------------    ------------------
                R1/R5              5.85 deg              30.0 deg
                R2/R6   180 deg +  5.85 deg    180 deg + 30.0 deg
                R3/R7             54.15 deg              30.0 deg
                R4/R8   180 deg + 54.15 deg    180 deg + 30.0 deg
                   
   
                              S/C +X axis       S/C +Z axis into page
                                   .                 (facing Sun) 
                                  /|\      
                                   |           
                                   |
                                   |             A1 (on +Z side)
                                                 A4 (on -Z side)
           R4/R8 Dir               /`~~__       / 
            '~._            , = .^ - /_  ``-.  /     
                /~._      .+     +     `^~/ .\/       
            30 |    '~.          + . -- ' `` @\   _-~  
           - - + - - - -# R4/R8                \~'`   \ 
                  /' '-_ .                      \,.=.. \
                 /    ~          _,.,_         +      + \
   R2/R6 Dir    /  ,~'        +'       `'+     +      +  \
    '~._       / ~^        .'        , = .'.    '- =''  -``        
        /~._   ^/         /  , = .  +     + \       \~'` 
    30 |    '~. |        .  +     + +     +  .       \    S/C +Y axis ----->
   - - + - - - -|# R2/R6 |  +     +  ' = '   |        \
                |        |   ' = ', - .      | R1/R5 #._- - - - - + - -
             _+_:        '       +     +     '       /  '~._      |                 
            \_ __\__      \      +     +    /      /^*~,    '~._ /  30 deg  
              +  |   \     '.     ' = '   .'      /    /        '~.  
                  `~-'       '~..,___,..~'       /    /~,*     R1/R5 Dir
                   _\                           /    /~,*`
                * /  \                          ^*._/  *`
                 *\ _/`.                R3/R7 #/._- - - - - + - -
                  *   / /\@_ _  ,.-^-., _ _ _ /   '~._      |
                     '='  |    +       +              '~._ /  30 deg
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
   
                                 S/C +Z' axis
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
    S/C  +X' axis                                   ^~     --*
                                                       ^~
                                                          ^~
                                                        S/C +Y' axis
   

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
                               
                                 S/C +Z' axis
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
  S/C +Y' axis '.,_._;-'                                ^~ 
                                                           ^~
                                                        S/C -X' axis


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
   mostly in the radial direction with a 30 deg tilt in the -Z direction.
   They are approximated aligned along the spacecraft Z axis, though the
   origins are offset from absolute alignment by a few centimeters (see
   table below). Azimuthally, the sun sensors are located near the SWAPI
   instrument approximately 18 deg off of the Y-Z plane.
       
   There are two star trackers mounted adjacent to each other on the
   underside of the spacecraft close to the -Z digital star sensor[6].
   Their boresights are generally downward (towards -Z), with an angular
   separation of 24 deg. One is angled toward the +X direction, the other
   angled towards the -X direction.

   Positional information may be captured in the IMAP structure SPK,
   while the orientation information is captured here.

                                 
   Digital Sun Sensor    ID      X (mm)    Y (mm)    Z (mm)    Loc. Azim
   --------------------  ------  --------  --------  --------  ---------
   IMAP_SUN_SENSOR_PZ    -43030   -364.22  -1121.90   1301.67   162.014 deg
   IMAP_SUN_SENSOR_MZ    -43031   -379.11  -1167.77     72.89   162.014 deg


   Digital Star Tracker  ID      X (mm)    Y (mm)    Z (mm)    Loc. Azim
   --------------------  ------  --------  --------  --------  ---------
   IMAP_STAR_TRACKER_PX  -43040    -45.75   -906.66    159.88   177.111 deg
   IMAP_STAR_TRACKER_MX  -43041   -188.05   -881.57    142.79   167.959 deg


   #####################################################################
   #                     /                            _-        __.----#
   #                   ,'                           ~`     _.~^'       #
   #                  /                           ~`    ,~^      S/C   #
   #                ,'  S/C +Z axis into page   .`    .^       +X axis #
   #               /      (facing Sun)         /    .^               . #
   #              |                           :    /_,-----,_       /|\#
   #              |                          ~    ~`         ^.      | #
   #              |                         ^    ^             ^_    | #
   #              |                        /    /                ,   | #
   #              |                       ,    ,                 ;   | #
   #              |                      ;    ;                   }  | #
   #    S/C       |           ___        :    :                   ~ ___#
   #  -Y axis  ___|         .`   `.      |    |                   }/  _#
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
  
      +Z DSS: (A, B, Y)  =  (  -18.000,   0.000,  0.000  )

      -Z DSS: (A, B, Y)  =  (  -18.000, 120.000, -0.008  )

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
   #                    \         ;      /               ^             #
   #                     ^,       :    ,^                |             #
   #                       ~_      ; _~                  |             #
   #                          '~--~"                     |             #
   #                               `                     x-------> +Y  #
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
  
      +X Star Tracker: (A, B, Y)  =  (   80.000, 168.000, -90.000  )

      -X Star Tracker: (A, B, Y)  =  ( -100.000, 168.000,  90.000  )

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


                                           -------------
                                          | S/C +Z axis |
        -----------------------            -------------
       | S/C +X axis into page |    #-----#    .
        -----------------------     | LGA |   /|\
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
             -------------       /_\      -----------        |__|
            | S/C +Y axis |    #-----#   | S/C FRAME |       STAR
             -------------     | MGA |   |  ORIGIN   |     TRACKERS
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
   #                             #-----#    { *  `"*,_____,*"`  * }    #
   #            +X               # LGA #    { *       | |       * }    #
   #             ^               #-----#     \ *      | |      * /     #
   #             |                            ~. *    | |    * .~      #
   #             |                              "|~####|####~|"        #
   #             |                                                     #
   #  +Y <-------o                                   IDEX              #
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
   #        /       S/C +Z into   __      ;    ;          -            #
   #       ,           page     .`##`.   :    :            `- . , _ ___#
   #     |/  S/C +Y <----x      ;#**#;   |    |                    /  _#
   #     |\             |       `.##.`   |    |        ,.----.,   / (` #
   #       '            |         |      |    |     _~`        `~_\ (,_#
   #        \           v      #-----#   |    |    ~              ~\___#
   #         '        S/C +X   # MGA #   :    :  ,`                `,  #
   #          \                #-----#    ;    ;,                    , #
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
   orientation of the boresight from a polar angle of 60 deg (slightly
   towards the Sun) to 180 deg (directed away from the Sun).
   

                       -------------
                      | S/C +Z axis |
                       -------------           -------------------------
                           .                  | S/C +X axis out of page |
                          /|\                  -------------------------
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
          |__|        -----------      /_\           -------------
          STAR       | S/C FRAME |     MGA          | S/C +Y axis |
        TRACKERS     |  ORIGIN   |                   -------------
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
   the +X' axis. When the pivot angle is 90 deg, the boresight is aligned
   with the local -Y' axis. The +Z' axis, from which the pivot angle is
   measured, aligns with the spacecraft +Z axis at pivot angle 0.
   
   The nominal boresight look-direction is defined in [6] for the 
   azimuth-elevation (deg):
   
      LO (azim, elev) = ( +330, -90 to +30 )
                           
   At 0 deg elevation (90 deg polar angle), the boresight direction and
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
   the spacecraft XYZ coordinate system are (deg):
  
      IMAP-Lo: (A, B, Y)  =  (  150.000,   0.000,  0.000  )

   Using the formulas described in the Euler angles section above, the
   rotation matrix generated from these Euler angles is consistent with
   the rotation matrix using the azimuth/elevation look direction.
   
   
   IMAP-Lo Orientation
   ---------------------------------------------------------------------
   
   The orientation of IMAP-Lo must be specified in a separate C-kernel.
   To facilitate this specification, a base frame representing the fixed
   transformation of the local X'Y'Z' frame to the spacecraft frame has
   been provided.
   
   The C-kernel will simply specify transformation within the
   local IMAP-Lo frame, and be generated using only the pivot angle.
   The implementation of this is outside the scope of this kernel.
     
   The IMAP-Lo base frame is defined such that
      -Y is the IMAP-Lo look direction at 0 degree pivot angle (nominally
         aligned with the S/C +Z axis)
      +X is the pivot angle, measured from 0 degrees.
 
   The rotation taking vectors from the IMAP-Lo base frame to the
   S/C frame is defined below.
     
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
                                    0.00000000000000000,
                                    0.00000000000000000,
                                   -1.00000000000000000,
                                   -0.50000000000000000,
                                   -0.86602540378443865,
                                    0.00000000000000000)
                                    
   \begintext

   The IMAP-Lo frame describes the articulation of the pivot and is
   captured in a SPICE C-Kernel (CK) file [1]. The IMAP-Lo CK frame
   rotates the base frame about its +X axis by the pivot angle shown
   in the diagram above.

   \begindata
   
   FRAME_IMAP_LO               = -43101
   FRAME_-43101_NAME           = 'IMAP_LO'
   FRAME_-43101_CLASS          = 3
   FRAME_-43101_CLASS_ID       = -43101
   FRAME_-43101_CENTER         = -43
   
   \begintext
   
   The IMAP-Lo star sensor frame is nominally aligned with the IMAP-Lo
   frame. The offset is determined from the measured alignments [17] and
   is captured in the definition below.

   \begindata
   
   FRAME_IMAP_LO_STAR_SENSOR   = -43102
   FRAME_-43102_NAME           = 'IMAP_LO_STAR_SENSOR'
   FRAME_-43102_CLASS          = 4
   FRAME_-43102_CLASS_ID       = -43102   
   FRAME_-43102_CENTER         = -43
   TKFRAME_-43102_RELATIVE     = 'IMAP_LO'
   TKFRAME_-43102_SPEC         = 'MATRIX'
   TKFRAME_-43102_MATRIX       = ( 0.999991181093041,
                                  -0.004199686195312,
                                  -0.000019287445755,
                                   0.004199730484764,
                                   0.999980635401645,
                                   0.004592503193045,
                                   0.000000000000000,
                                  -0.004592543694261,
                                   0.999989454215601 )

   \begintext


IMAP-Hi Frames
========================================================================

   IMAP-Hi consists of two identical, single-pixel high energy neutral 
   atom (ENA) imagers. Hi 90 is oriented with its boresight
   perpendicular to the spacecraft spin axis, while Hi 45 is radially
   outward but with the boresight angled 45 deg from the -Z axis.
   
                       -------------
                      | S/C +X axis |
                       -------------        
                                                      -----------------------
        Hi 45 BORESIGHT    .     Hi 90 BORESIGHT     | S/C +Z axis into page |
                \         /|\         /              |     (facing Sun)      |
                 \ 15 deg  | 15 deg  /                -----------------------
                  " .~'^'~.| .~'^'~." 
                   \       |       / 
                    ,      |      ,
                     ;    /`~~__ ,             `+    direction of
                    , = .^ - /_  ``-.              '.    positive
                  .+     +     `^~/ ./                ~    rotation
                 ^ +     + . -- ' ``  \   _-~          \
            _   /   ',= '              \~'`   \ IMAP    \ 
    ULTRA /' '-_ .~ '                   \,.=.. \ Lo    \|/
     90  /    ~          _,.,_         +      + \       '
        /  ,~'        +'       `'+     +      +  \
       / ~^        .'        , = .'.    '- =''  -``   -------------     
       ^/         /  , = .  +     + \       \~'`     | S/C +Y axis |-----> 
        |        .  +     + +     +  .       \        ------------- ___
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
   
   
                     -------------
                    | S/C +Z axis |
                     -------------             -------------------------
                           .                  | S/C +X axis out of page |
                          /|\                  -------------------------
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
          |__|        -----------      /_\           -------------
          STAR       | S/C FRAME |     MGA          | S/C +Y axis |
        TRACKERS     |  ORIGIN   |                   -------------
                      -----------


   #######################################################################
   #______________________________________________                       #
   #                   /  _       |        ||     |        IMAP Hi 90    #
   #----~.            /  |_| O o  |        ||     |====      hidden      #
   #     ULTRA 90    /\   x  x =  |        ||     |        behind S/C    #
   #                /  \__________|        ||     || <---- struct here   #
   #   ##### -- ####           |           ||     |]                     #
   #  ## % ## / \###\         /    ___     ||     |}  Hi 90 Boresight    #
   # /## % ##\--|####|       |____|*#*|    ||     |}________________\    #
   # |## % ##|--|####|       |    |*#*|    ||     |}                /    #
   # |## % ##|  |####|       |     ---     ||     |}                     #
   # |## % ##|--|####|       |             ||     |]                     #
   # |## % ##|--|####|        \            ||     ||     S/C +Z          #
   # \## % ##/  |####|         |           ||     .        ^             #
   #  ## % ## \ /###/          |           ||   .'.        |             #
   #   ##### -- ####           |           || .  /,        |             #
   #---------------            |           |.`   _~        x----> S/C +X #
   #                           |          ,`  ,~`  `~    S/C +Y          #
   #              ______      /         .`  ~`  _    \  into page        #
   #  .=.=.=.=.  |( ) ()|    /     ___ *  -' ~' `',  |                   #
   #  | | | | |  |( ) ()|    |____|*#*|| ~ .`_ _  / ~                    #
   #__#_#_#_#_#__|______|         |*#*||` / //// / ~                     #
   #----------------|            .-----  /  ` ` ' ~                      #
   #  |_ _ _|   |   |          .'_   / '`.,_ ,~' ~.                      #
   #  | | | |   |   |        .'   -, |       _, `  ":.                   #
   #__/_/_/_/___/___|________|______;_\_ ,.-'         |:.                #
   #                                       | |       /   ":.             #
   #_______________________________________| |   _45 deg    ":.          #
   #___  ____  __ ||     ||     ||     |     |-~"              "         #
   #  / /   / / // |_____||_____||_____|     |                Hi 45      #
   #_/ /___/ /_/ |                 /|\      \|/             Boresight    #
   #            /                            '                           #
   #            |                       -------------                    #
   #           /                       | S/C -Z axis |                   #
   #===========                         -------------                    #
   #######################################################################
   
   
   The local IMAP-Hi frame[15]--identical for both sensors--is defined
   with the boresight aligned with the +Y' axis, the rectangular vent
   ports aligned with the +Z' axis, and X' = Y' x Z'.
   
   The local coordinate system is shown below, looking into the sensor.
   The vent ports are aligned as shown with the Z' axis.
   
                                +Z'
                                 ^
                                 |
                                 |
                            _____|_____         
                        .-'`     |      '-.      
                      .'     ____|____     '.    
                     /     .'    |    '.     \   
                    /    .'    |'|'|    '.    \  
                   /    /      | | |      \    \ 
                  |    /       |_|_|       \    |
        +X' /_____|__ |__________|          |   |
            \     |   |                     |   |
                  |    \       |'''|       /    |
                   \    \      |   |      /    / 
                    \    '.    |___|    .'    /  
                     \     '.         .'     /   
                      '.     '-------'     .'    
                        '-.             .-'      
                           '-.........-'          
     

   IMAP HI 45
   --------------
   
   The nominal boresight look-direction is defined in [6] for the 
   azimuth-elevation (deg):
   
      HI 45 (azim, elev) = ( +255, -45 )
                           
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
   local Y' axis by 3 deg as a consequence of the angular offset of the
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
      [Z]S/C   [  -0.706138  -0.707107  -0.037007  ] [Z']HI 45

   Using the method described in a Euler discussion section, the Euler
   angles rounded to 1/1000th of a degree are:
  
      HI 45:  (A, B, Y)  =  ( -17.122,  92.121, -135.037 )

   Using the formulas described in the Euler angles section above, the
   rotation matrix generated from these Euler angles is consistent with
   the rotation matrix using the azimuth/elevation look direction;
   however, the full double-precision Euler angles are necessary to
   generate the proper precise rotation matrix. 

   Applying the method described above to the measured alignment vector
   in [17], 
   
   D = +Y' = [ 0.683178772, -0.185278978, -0.706355764 ],
   
   we arrive at the definition below.
   
   \begindata

   FRAME_IMAP_HI_45            = -43150
   FRAME_-43150_NAME           = 'IMAP_HI_45'
   FRAME_-43150_CLASS          = 4
   FRAME_-43150_CLASS_ID       = -43150   
   FRAME_-43150_CENTER         = -43
   TKFRAME_-43150_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43150_SPEC         = 'MATRIX'
   TKFRAME_-43150_MATRIX       = ( -0.667096841004280
                                    0.235144007037882
                                   -0.706886907981948
                                    0.683178772158437
                                   -0.185278978042968
                                   -0.706355764163813
                                   -0.297066608682633
                                   -0.954137828748825
                                   -0.037046373051589 )

   \begintext


   IMAP HI 90
   --------------
   
   The nominal boresight look-direction is defined in [6] for the 
   azimuth-elevation (deg):
   
      HI 90 (azim, elev) = ( +285, 0 )
                           
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
   local Y' axis by 15 deg as a consequence of the angular offset of the
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
  
      HI 90:  (A, B, Y)  =  ( -165.000,  75.000,   90.000  )

   Using the formulas described in the Euler angles section above, the
   rotation matrix generated from these Euler angles is consistent with
   the rotation matrix using the azimuth/elevation look direction. 

   Applying the method described above to the measured alignment vector
   in [17], 
   
   D = +Y' = [ 0.965176886, 0.261597765, 0.000434036 ],
   
   we arrive at the definition below.
 
   \begindata
     
   FRAME_IMAP_HI_90            = -43151
   FRAME_-43151_NAME           = 'IMAP_HI_90'
   FRAME_-43151_CLASS          = 4
   FRAME_-43151_CLASS_ID       = -43151   
   FRAME_-43151_CENTER         = -43
   TKFRAME_-43151_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43151_SPEC         = 'MATRIX'
   TKFRAME_-43151_MATRIX       = (  0.067301842956904
                                   -0.249915857460229
                                    0.965925735305009
                                    0.965176885850928
                                    0.261597764959596
                                    0.000434035999933
                                   -0.252792485951837
                                    0.932259981742251
                                    0.258819020723416 )

   \begintext


IMAP-Ultra Frames
========================================================================

   The IMAP-Ultra instrument[7,14] consists of two identical sensors for
   imaging the emission of energetic neural atoms (ENAs) produced in the
   heliosheath and beyond. Ultra 90 is mounted perpendicular to the IMAP
   spin axis (+Z), while Ultra 45 is mounted at 45 degrees from the
   anti-sunward spin axis (-Z).
   
   
                       -------------
                      | S/C +X axis |     -----------------------
                       -------------     | S/C +Z axis into page |
                           .             |     (facing Sun)      |
                          /|\             -----------------------
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
   30  / ~^        .'        , = .'.    '- =''  -``   -------------     
   |   ^/         /  , = .  +     + \       \~'`     | S/C +Y axis |-----> 
   ----  |        .  +     + +     +  .       \       ------------- ___
        |        |  +     +  ' = '   |        \--------------------|   |
   SWAPI|        |   ' = ', - .      |        /--------------------|___|
     _+_:        '       +     +     '       /        |      MAG boom        
    \_ __\__      \      +     +    /      /^*~,      .        
      +  |  SWE    '.     ' = '   .'    ULTRA  /     33 deg
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

   
   #########################################################################
   #                                                                       #
   #             One half of one IMAP Ultra sensor showing                 #
   #              assembly of fanned-out collector plates                  #
   # Outward                                                               #
   #    .             Assemblies are mirror-symmetric                      #
   #   /|\           about the leftmost edge of drawing                    #      
   #    |                                                                  #
   #    |     ,--.        ,                                                #
   #          || |   |    ;        ,          63.42 deg FOR                #
   #    |      | |   |    ;   ;    ;          60.31 deg FOV                #
   #    |      | |   |    :   ;   ;    ;    ;                              #                    
   #   \|/     | |   |   :   ;    ;   ;    /                               #
   #    '      | ;_ _|_  ;   ;   ;    ;   /    /    .                      #
   #   S/C      | |      ``'''^-,/,   /   /    /    .'                     #
   #           | | ___            `''., /    /    .    .                   #
   #_________;-|__|_  `'"^~-,._ /^~    `^., /   ,'    '     .              #
   #---------'- | |_|          `'":.,    _ `^, .    ,'    .'               #
   #--------. ,-| |    @             `'~/  \  `'. .`    .'    ,            #
   #     ,'`.'  | | @   @ @  @         '~,.;,    '.   .'    .'             #
   #   .','     | |  @    @  @  @  @        `;, /~_':'    .'     ,'        #
   #.'`,'      _| |_            @  @     @    `;, /  '. .'    ,'`          #
   # `,       |_|-|_|                @  @        '.    '.   .'    ,.       #
   #'-,'.       |-|                 @     @   @    ',    `.`    ;'         #
   #   '.'-.    | |                      @   @       ;,    \,;`'    .-`    #
   #     `-.'.  | |                            @  @    ",   ',  ,.'` ,^    #
   #_________:'-| | @   @                     @  @      :,  _,\' .-``      #
   #-----------||-|-,      @                          /~,".' ;'.'          #
   #===        ||---|@  @     @                 @ @  ,\ '  ,.^`            #
   #___________||_/-~_ _   @     @           _,  _,-' ,.-'`                #
   #   @| | |  ||       `- , @      @        \,\'  ,'`     +Z'             # 
   #----' | |  ||           `~,@       @  ,~`' _,'`         ^              #
   #      | |  ||              ',@ .^\_,'` ,.'`             |              #
   #______'-'__||-@--~-~,        \ .;`  .'`                 |              #
   #___________||/ ~  #  ~        | {.'`                    |              #
   #        |* ||*|   +  <------------ Collector plate      o------> +Y'   #
   #____     --||\ ~     ~       |     axis of symmetry     Instrument     #
   #_  *|      ||-@-^~-~^-------|                           Coordinates    #
   #*| |       ||_______________|                                          #
   #___*|_______|_|_|__|__|_|_| |                                          #
   #########################################################################
   
   
   The local IMAP-Ultra frame[14]--identical for both sensors--is
   defined with the collector-plate-fan axes of symmetry aligned with
   the +X' axis, the cylindrical axes offset in the +Y' axis, and the
   Z' axis perpendicular to both and outward as in the diagram below.
      
   
   IMAP ULTRA 45
   --------------
   
   The nominal outward look-direction is defined in [6] for the 
   azimuth-elevation (deg):
   
      ULTRA 45 (azim, elev) = ( +33, -45 )
                           
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
  
      ULTRA 45:  (A, B, Y)  =  ( -147.000, 135.000,   90.000  )

   Using the formulas described in the Euler angles section above, the
   rotation matrix generated from these Euler angles is consistent with
   the rotation matrix using the azimuth/elevation look direction.

   The measured alignment vectors [17] correspond to the instrument +Z
   and +X coordinates:
   
      D = +Z' = [0.499259504; -0.866451261; -0.001469775]; 
     -N = +X' = [0.0033786069; -0.0046688268; 0.9999833934]; 
   
   Using the calculation described above, we arrive at the rotation 
   below, taking vectors from the ULTRA 45 frame to the S/C frame.
     
   \begindata

   FRAME_IMAP_ULTRA_45         = -43200
   FRAME_-43200_NAME           = 'IMAP_ULTRA_45'
   FRAME_-43200_CLASS          = 4
   FRAME_-43200_CLASS_ID       = -43200   
   FRAME_-43200_CENTER         = -43
   TKFRAME_-43200_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43200_SPEC         = 'MATRIX'
   TKFRAME_-43200_MATRIX       = ( -0.381247548031845
                                    0.595862003625695
                                    0.706823018693420
                                    0.841149671034019
                                    0.540797922474097
                                   -0.002199537920242
                                   -0.383559041138477
                                    0.593705381214347
                                   -0.707387010255390 )

   \begintext


   IMAP ULTRA 90
   --------------
   
   The nominal outward look-direction is defined in [6] for the 
   azimuth-elevation (deg):
   
      ULTRA 90 (azim, elev) = ( +210, 0 )
                           
   The look-direction is the +Z' local axis of instrument, and the
   primary axis in the spacecraft frame of reference is:
       
      D = +Z' = [ -cos(0) x sin(210), cos(0) x cos(210), sin(0) ]
   
   The secondary axis is the +X' local axis, lying along spacecraft
   +Z axis.
   
      S = +X' = [ 0, 0, 1 ]
      
   The tertiary axis is:
      
      N = D x S = Z' x X' = Z' x [ 0, 0, 1 ]

   The rotation matrix formed using the column vectors is:
   
      R = [ +N, +S, +D ]

   The rotation matrices orienting the IMAP-Ultra 90 sensor on the
   spacecraft is given by [6]:
  
      [X]      [   0.000000  -0.866025   0.500000  ] [X']
      [Y]    = [   0.000000  -0.500000  -0.866025  ] [Y']
      [Z]S/C   [   1.000000   0.000000   0.000000  ] [Z']ULTRA 90
   
   Using the method described in a Euler discussion section, the Euler
   angles rounded to 1/1000th of a degree are:
  
      ULTRA 90:  (A, B, Y)  =  (   30.000,  90.000,   90.000  )

   Using the formulas described in the Euler angles section above, the
   rotation matrix generated from these Euler angles is consistent with
   the rotation matrix using the azimuth/elevation look direction.

   The measured alignment vectors [17] correspond to the instrument +Z
   and +X coordinates:
   
      D = +Z' = [0.499259504; -0.866451261; -0.001469775]; 
      N = +X' = [0.0033786069; -0.0046688268; 0.9999833934]; 
   
   Using the calculation described above, we arrive at the rotation 
   below, taking vectors from the ULTRA 90 frame to the S/C frame.
   
   \begindata

   FRAME_IMAP_ULTRA_90         = -43201
   FRAME_-43201_NAME           = 'IMAP_ULTRA_90'
   FRAME_-43201_CLASS          = 4
   FRAME_-43201_CLASS_ID       = -43201   
   FRAME_-43201_CENTER         = -43
   TKFRAME_-43201_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43201_SPEC         = 'MATRIX'
   TKFRAME_-43201_MATRIX       = (  0.001250593582091
                                   -0.000975706837283
                                    0.999998742005139
                                   -0.866451604965052
                                   -0.499260713960047
                                    0.000596447474492
                                    0.499259503934755
                                   -0.866451260886769
                                   -0.001469774999808 )

   \begintext


IMAP Magnetometer (MAG) Frames
========================================================================

   The IMAP magnetometer (MAG)[7,16] consists of a pair of identical 
   triaxial fluxgate magnetometers mounted on a ~2.5 meter boom. MAG-O
   is positioned at the end of the boom, while MAG-I is mounted ~0.75
   meters from MAG-O.


                       -------------
                      | S/C +X axis |      -----------------------
                       -------------      | S/C +Z axis into page |
                           .              |     (facing Sun)      |
                          /|\              -----------------------
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
       ^/         /  , = .  +     + \       \~'`       S/C +Y axis  -----> 
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

                                          +X' <-----x   +Y' into
                                                    |     page
                                      MAG Local     |
                                    Coord System    v
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
   
   The MAG local coordinate system is shown in the diagram above. The 
   matrix taking vectors from the MAG coordinate system to the spacecraft
   coordinate system is provided below in the IMAP_MAG_BASE frame definition.
   It represents a nominal, or idealized, orientation. The measured 
   alignments [17] are given in frames IMAP_MAG_I and IMAP_MAG_O.      

   \begindata

   FRAME_IMAP_MAG_BOOM         = -43250
   FRAME_-43250_NAME           = 'IMAP_MAG_BOOM'
   FRAME_-43250_CLASS          = 4
   FRAME_-43250_CLASS_ID       = -43250 
   FRAME_-43250_CENTER         = -43
   TKFRAME_-43250_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43250_SPEC         = 'MATRIX'
   TKFRAME_-43250_MATRIX       = (  0.999895975249542,
                                   -0.013794792994061,
                                   -0.004212168802143,
                                    0.013794915371822,
                                    0.999904845627765,
                                   -0.000000000000000,
                                    0.004211767995864,
                                   -0.000058106512157,
                                    0.999991128777642 )									

   FRAME_IMAP_MAG_BASE         = -43253
   FRAME_-43253_NAME           = 'IMAP_MAG_BASE'
   FRAME_-43253_CLASS          = 4
   FRAME_-43253_CLASS_ID       = -43253
   FRAME_-43253_CENTER         = -43
   TKFRAME_-43253_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43253_SPEC         = 'MATRIX'
   TKFRAME_-43253_MATRIX       = (  0.0,
                                   -1.0,
                                    0.0,
                                   -1.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                   -1.0 )

   FRAME_IMAP_MAG_I            = -43251
   FRAME_-43251_NAME           = 'IMAP_MAG_I'
   FRAME_-43251_CLASS          = 4
   FRAME_-43251_CLASS_ID       = -43251
   FRAME_-43251_CENTER         = -43
   TKFRAME_-43251_RELATIVE     = 'IMAP_MAG_BASE'
   TKFRAME_-43251_SPEC         = 'MATRIX'
   TKFRAME_-43251_MATRIX       = ( 0.999999871280306
                                   0.000507384835577
                                   0.0
                                  -0.000507348727136
                                   0.999928705504181
                                  -0.011930067309210
                                  -0.000006053135240
                                   0.011930065773575
                                   0.999928834214714 )

   FRAME_IMAP_MAG_O            = -43252
   FRAME_-43252_NAME           = 'IMAP_MAG_O'
   FRAME_-43252_CLASS          = 4
   FRAME_-43252_CLASS_ID       = -43252
   FRAME_-43252_CENTER         = -43
   TKFRAME_-43252_RELATIVE     = 'IMAP_MAG_BASE'
   TKFRAME_-43252_SPEC         = 'MATRIX'
   TKFRAME_-43252_MATRIX       = ( 0.999946560385648
                                   0.010338102964849
                                   0.0
                                  -0.010337900188807
                                   0.999926946999490
                                  -0.006263264641177
                                  -0.000064750274757
                                   0.006262929934730
                                   0.999980385565654 )

   \begintext


IMAP Solar Wind Electron (SWE) Frames
========================================================================
     
   The SWE instrument frame is defined in [18] as
   
      *  -X is the outward facing direction of the center of the field of view,
            pointing away from the S/C body 
      *  +Z is nominally aligned with S/C +Z
      *  +Y complements the right-handed frame
      
   A view of the instrument looking down the Y' axis is illustrated below. 
   
                                                
             .                                       ^   S/C +Z 
          P63 .      ^ +Z'                           | (spin axis)
          .    .     |                               |
      P43  `.   .    |_________ SWE Sensor           |
         .    `.  .  ||         |                    |
     P21   `.   `. . ||         |
        `   . `.  `..||         |
    000  -X' <--------x         |
        .   ' .'  .'.| +Y' (into page )
     M21   .'   .' . |          |
         '    .'  .  |__________|______________
       M43  .'   .     |                       |  Mounting Plate
           '    .     _|_______________________|_____
          M631 .     | / / / / / / / / / / / / / / /
              .      |/ / Spacecraft Deck / / / / / /                   
            
   
      
   [17] provides the measured value of the instrument -X axis. 
   Taking the cross product with the +Z axis and normalizing 
   results in the frame definition below.
     
   \begindata

   FRAME_IMAP_SWE              = -43300
   FRAME_-43300_NAME           = 'IMAP_SWE'
   FRAME_-43300_CLASS          = 4
   FRAME_-43300_CLASS_ID       = -43300   
   FRAME_-43300_CENTER         = -43
   TKFRAME_-43300_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43300_SPEC         = 'MATRIX'
   TKFRAME_-43300_MATRIX       = (  0.453749717807575,
                                    0.891129167735826,
                                    0.000000000000000,
                                   -0.891129167735826,
                                    0.453749717807575,
                                    0.000000000000000,
                                    0.000000000000000,
                                    0.000000000000000,
                                    1.000000000000000 )

\begintext

   The frames for the individual SWE detectors is defined as follows:
       +Y is aligned with the +Y axis in the SWE frame
       +Z is the outward facing direction of the center of the CEM field of 
          view, pointing away from the S/C body 
       +X complements the right-handed frame
      
   The orientation of each detector frame is a fixed offset from SWE frame,
   achieved by rotating the SWE frame about its +Y axis by the angle in the
   table below.
      
   The table below contains nominal rotation offsets, as specified in [19],
   for each of the CEMs:
   
                             Rotation about +Y
             Detector         from SWE frame
            ----------    ------------------------
                P63             +63 degrees
                P42             +42 degrees
                P21             +21 degrees
                000               0 degrees
                M21             -21 degrees
                M42             -42 degrees
                M63             -63 degrees
   
   \begindata
   
   FRAME_IMAP_SWE_DETECTOR_P63 = -43301
   FRAME_-43301_NAME           = 'IMAP_SWE_DETECTOR_P63'
   FRAME_-43301_CLASS          = 4
   FRAME_-43301_CLASS_ID       = -43301   
   FRAME_-43301_CENTER         = -43
   TKFRAME_-43301_RELATIVE     = 'IMAP_SWE'
   TKFRAME_-43301_SPEC         = 'ANGLES'
   TKFRAME_-43301_ANGLES       = ( 0,  63.0, 0 )
   TKFRAME_-43301_AXES         = ( 1,   2,   3 )
   TKFRAME_-43301_UNITS        = 'DEGREES'
      
   FRAME_IMAP_SWE_DETECTOR_P42 = -43302
   FRAME_-43302_NAME           = 'IMAP_SWE_DETECTOR_P42'
   FRAME_-43302_CLASS          = 4
   FRAME_-43302_CLASS_ID       = -43302
   FRAME_-43302_CENTER         = -43
   TKFRAME_-43302_RELATIVE     = 'IMAP_SWE'
   TKFRAME_-43302_SPEC         = 'ANGLES'
   TKFRAME_-43302_ANGLES       = ( 0,  42.0, 0 )
   TKFRAME_-43302_AXES         = ( 1,   2,   3 )
   TKFRAME_-43302_UNITS        = 'DEGREES'
      
   FRAME_IMAP_SWE_DETECTOR_P21 = -43303
   FRAME_-43303_NAME           = 'IMAP_SWE_DETECTOR_P21'
   FRAME_-43303_CLASS          = 4
   FRAME_-43303_CLASS_ID       = -43303
   FRAME_-43303_CENTER         = -43
   TKFRAME_-43303_RELATIVE     = 'IMAP_SWE'
   TKFRAME_-43303_SPEC         = 'ANGLES'
   TKFRAME_-43303_ANGLES       = ( 0,  21.0, 0 )
   TKFRAME_-43303_AXES         = ( 1,   2,   3 )
   TKFRAME_-43303_UNITS        = 'DEGREES'
      
   FRAME_IMAP_SWE_DETECTOR_000 = -43304
   FRAME_-43304_NAME           = 'IMAP_SWE_DETECTOR_000'
   FRAME_-43304_CLASS          = 4
   FRAME_-43304_CLASS_ID       = -43304
   FRAME_-43304_CENTER         = -43
   TKFRAME_-43304_RELATIVE     = 'IMAP_SWE'
   TKFRAME_-43304_SPEC         = 'ANGLES'
   TKFRAME_-43304_ANGLES       = ( 0,   0,   0 )
   TKFRAME_-43304_AXES         = ( 1,   2,   3 )
   TKFRAME_-43304_UNITS        = 'DEGREES'
      
   FRAME_IMAP_SWE_DETECTOR_M21 = -43305
   FRAME_-43305_NAME           = 'IMAP_SWE_DETECTOR_M21'
   FRAME_-43305_CLASS          = 4
   FRAME_-43305_CLASS_ID       = -43305
   FRAME_-43305_CENTER         = -43
   TKFRAME_-43305_RELATIVE     = 'IMAP_SWE'
   TKFRAME_-43305_SPEC         = 'ANGLES'
   TKFRAME_-43305_ANGLES       = ( 0  -21.0,   0 )
   TKFRAME_-43305_AXES         = ( 1,   2,   3 )
   TKFRAME_-43305_UNITS        = 'DEGREES'
      
   FRAME_IMAP_SWE_DETECTOR_M42 = -43306
   FRAME_-43306_NAME           = 'IMAP_SWE_DETECTOR_M42'
   FRAME_-43306_CLASS          = 4
   FRAME_-43306_CLASS_ID       = -43306
   FRAME_-43306_CENTER         = -43
   TKFRAME_-43306_RELATIVE     = 'IMAP_SWE'
   TKFRAME_-43306_SPEC         = 'ANGLES'
   TKFRAME_-43306_ANGLES       = ( 0, -42.0,   0 )
   TKFRAME_-43306_AXES         = ( 1,   2,   3 )
   TKFRAME_-43306_UNITS        = 'DEGREES'
      
   FRAME_IMAP_SWE_DETECTOR_M63 = -43307
   FRAME_-43307_NAME           = 'IMAP_SWE_DETECTOR_M63'
   FRAME_-43307_CLASS          = 4
   FRAME_-43307_CLASS_ID       = -43307
   FRAME_-43307_CENTER         = -43
   TKFRAME_-43307_RELATIVE     = 'IMAP_SWE'
   TKFRAME_-43307_SPEC         = 'ANGLES'
   TKFRAME_-43307_ANGLES       = ( 0, -63.0,   0 )
   TKFRAME_-43307_AXES         = ( 1,   2,   3 )
   TKFRAME_-43307_UNITS        = 'DEGREES'
   
   \begintext
   

IMAP Solar Wind and Pickup Ion (SWAPI) Frames
========================================================================
   
   SWAPI has the following nominal alignment to the spacecraft frame,
   reference Table 1 of [6]. The azimuth and elevation angles are 
   illustrated in the 'IMAP I&T Component Placement' section near the 
   top of this document.
   
        azimuth  | elevation 
         (deg)   |  (deg)
        ---------+---------
           168   |    0
   
   The SWAPI base frame is defined in the instrument MICD [8] as follows:
   
      *  -Z axis is the axis of symmetry of the instrument, pointing 
            away from the spacecraft body.
      *  +Y axis is along the aperture center, in the anti-sunward direction.
      
   Two views of the instrument are illustrated below. The diagram on the left 
   is looking down the top of the instrument towards the spacecraft body. The 
   diagram on the right is a side view of the instrument assembly. In both 
   diagrams the sunglasses aperture vanes point to the right (+Y direction).
   The labeled coordinate axes are in the instrument reference frame.   
   
                                                 -Z'
            +X'                                   ^
             ^                             _______|________
             |                            |       |        |---- 
        . ***|*** .                       |       o---------------> +Y'
       *     |     *    .-'               |________________|----  (towards
      *      |      *.-'                  . '            ' .        Sun)
     *       |       *                  '--------------------'
     *       o----------> +Y'                 /        \   
     *               *                        |        |
      *             *'-.                      |\      /|          |
       *.         .*    `-.                  _|_|____|_|_         |
          *******                           |            |        |
                                            |            |        v
      spacecraft body                       |            |   spacecraft
        behind page                         |____________|      body
        
       
   The nominal azimuth and elevation give the outward axis of symmetry, -Z in the
   instrument frame:
    
      -Z      = -[ -sin(az) * cos(el), cos(az) * cos(el), sin(el) ]
        instr    
      
   [17] provides measured values of the above nominal instrument alignment. The
   following measured vectors are parallel to the spacecraft axes listed:
   
      -Y = Tophat Topplate Rib                    = [ -0.00142, -0.01019, -0.99995 ]
      
      -Z = Top of Aperture Grid Frame             = [ -0.20761, -0.97821, 0.001128 ]
      -Z = Top of Lower Outer ESA Mounting Flange = [ -0.20775, -0.97818, 0.00003  ]
      
   Since two measurements were taken for instrument -Z, we use their average.
   The X axis completes the right-handed coordinate system.
      
   \begindata

   FRAME_IMAP_SWAPI            = -43350
   FRAME_-43350_NAME           = 'IMAP_SWAPI'
   FRAME_-43350_CLASS          = 4
   FRAME_-43350_CLASS_ID       = -43350   
   FRAME_-43350_CENTER         = -43
   TKFRAME_-43350_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43350_SPEC         = 'MATRIX'
   TKFRAME_-43350_MATRIX       = ( -0.978196569749791
   0.207679902805407
  -0.000727255443315
  -0.000591151927685
   0.000717413380944
   0.999999567928626
   0.207680334815652
   0.978196577017512
  -0.000579000933447 )

   \begintext


IMAP Compact Dual Ion Composition Experiment (CoDICE) Frames
========================================================================

   CoDICE has the following nominal alignment to the spacecraft frame,
   reference Table 1 of [6]. The azimuth and elevation angles are 
   illustrated in the 'IMAP I&T Component Placement' section near the 
   top of this document.
   
        azimuth  | elevation 
         (deg)   |  (deg)
        ---------+---------
           136   |    0
           
        
   The CoDICE local coordinate system is defined [23] as follows:
   
      *  -X is the axis of symmetry of the instrument, pointing 
            away from the spacecraft body.
      *  +Z is aligned with the spacecraft +Z axis
      
   A diagram of the CoDICE local coordinate system is shown below.
   
                         -X'
                          ^
                  ________|_______
                 |________|_______| 
                    \     |    /
                    /     |    \   
                   |      o----------> +Z'
                    \          /   (towards Sun) 
                    \ \ \  / / /          
                   |''''''''''''|         
                   |            |      |  
                   |            |      |  
                   |            |      |  
                   |            |      |   
                   |            |      v  
                   |            |  spacecraft     
                   |            |     body       
                   |____________|  
                   
   The alignment measurements in [17] give the three axes of the 
   instrument coordinate system and are captured below.
     
   \begindata

   FRAME_IMAP_CODICE           = -43400
   FRAME_-43400_NAME           = 'IMAP_CODICE'
   FRAME_-43400_CLASS          = 4
   FRAME_-43400_CLASS_ID       = -43400   
   FRAME_-43400_CENTER         = -43
   TKFRAME_-43400_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43400_SPEC         = 'MATRIX'
   TKFRAME_-43400_MATRIX       = (  0.695804588908984
                                    0.718231138906051
                                    0.000071825599991
                                   -0.717601202113529
                                    0.696453711110183
                                    0.000861978000136
                                    0.000267138000086
                                    0.000252966000082
                                    0.999999932322743 )

\begintext


IMAP High-energy Ion Telescope (HIT) Frames
========================================================================
   
   HIT has the following nominal alignment to the spacecraft frame,
   reference Table 1 of [6]. The azimuth and elevation angles are 
   illustrated in the 'IMAP I&T Component Placement' section near the top 
   of this document.
   
        azimuth  | elevation 
         (deg)   |  (deg)
        ---------+---------
           30    |    0
           
        
   The HIT local coordinate system is defined for convenience as follows:
   
      *  +Y is the axis of symmetry of the instrument, pointing 
            away from the spacecraft body.
      *  +Z is aligned with Boresight Vector #2 in [21]. It is the  
            outward pointing vector in the center of the five-detector
            grouping on the sunward side of the spacecraft.
             
   A diagram of the HIT local coordinate system is shown below. The ten
   apertures are numbered for convenience and may not be consistent with
   actual aperture names.
   
   There are two groups, or sectors, of apertures, A1-A5 and A5-A10. Each
   group of five detectors spans a 130 degree field of view in the HIT XY
   plane [21] and is located symmetrically on opposite sides of the sensor
   head. Thus, each individual detector has a 130/5 = 26 degree field of  
   view in the HIT XY plane. The space between sectors is 50 degrees.
   
   
                     HIT local coordinate system
                     ----------------------------
                                         
     S/C +Z axis                         +Z' 
    (facing Sun)                         ^
          ^                              `
          |                             .
          |                             `
          |                       |    . 
                         \        |  A3`   /
                          \  A2   |   .   /
                     A1    \      |   `  /   A4
               '-.          \_..--+--.._/          .-'
                  '-.     .'        .    '.     .-'
     50 deg space ||'-..'           `      ' .-'
   between detector||||/           .         \    A5
     groups A1-A5 ||||.            `          .       ___
      and A6-A10 |||||            .           |---''''
            ___....---|           o ....      ||||||| 50 deg space
                      '        +Y'out)   ```` '|||||
                A10    \                     /|||||``` --- 
                     .-''.                 .''-.||        ````--> -X'
                  .-'     '.             .'     '-.
               .-'          / ''--+--'' \          '-.
                     A9    /      |      \   A6
                          /   A8  |  A7   \
                         /        |        \
                                  |
 
      
   [17] gives a measured value for the instrument +Y axis expressed
   in the spacecraft coordinate system:
   
      +Y = [ -0.494627173, 0.869103932, -0.00152133 ]
      
   and the instrument -X axis:
   
      -X = [ -0.721687951, -0.411326605, -0.556755714 ]
            
   The +Z axis completes the right-handed coordinate system. 
     
   \begindata

   FRAME_IMAP_HIT              = -43500
   FRAME_-43500_NAME           = 'IMAP_HIT'
   FRAME_-43500_CLASS          = 4
   FRAME_-43500_CLASS_ID       = -43500   
   FRAME_-43500_CENTER         = -43
   TKFRAME_-43500_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43500_SPEC         = 'MATRIX'
   TKFRAME_-43500_MATRIX       = (  0.721525796505660
                                    0.411611615432541
                                    0.556755245164234
                                   -0.494627173165007
                                    0.869103932289932
                                   -0.001521330000507
                                   -0.484504369994396
                                   -0.274288594220012
                                    0.830675196774975 )

   \begintext
   
   Each L1 aperture coordinate frame is defined as a rotation about the HIT
   center axis (Y axis in the HIT local coordinate system):
   
      *  +X is the aperture boresight, or outward pointing center vector.
      *  +Y is aligned with the HIT frame +Y.
      
   Note that for aperture L1 03, the boresight (aperture +X axis) is coaligned 
   with the HIT frame +Z axis, as shown in the diagram above.
   
   The rotations required to take the HIT +X axis to the aperture +Z axis are
   described here and in the definitions below.
    
                                Rotation about +Y
            L1 Aperture            from HIT frame
            -----------    -------------------------------
                01         50/2 + 26 * 4.5 =  142 degrees
                02         50/2 + 26 * 3.5 =  116 degrees
                03         50/2 + 26 * 2.5 =   90 degrees
                04         50/2 + 26 * 1.5 =   64 degrees
                05         50/2 + 26 * 0.5 =   38 degrees
                06        -50/2 - 26 * 0.5 =  -38 degrees
                07        -50/2 - 26 * 1.5 =  -64 degrees
                08        -50/2 - 26 * 2.5 =  -90 degrees
                09        -50/2 - 26 * 3.5 = -116 degrees
                10        -50/2 - 26 * 4.5 = -142 degrees     
      
   \begindata
   
   FRAME_IMAP_HIT_L1_APERTURE_01 = -43501
   FRAME_-43501_NAME             = 'IMAP_HIT_L1_APERTURE_01'
   FRAME_-43501_CLASS            = 4
   FRAME_-43501_CLASS_ID         = -43501   
   FRAME_-43501_CENTER           = -43
   TKFRAME_-43501_RELATIVE       = 'IMAP_HIT'
   TKFRAME_-43501_SPEC           = 'ANGLES'
   TKFRAME_-43501_ANGLES         = ( 0,  142, 0 )
   TKFRAME_-43501_AXES           = ( 1,   2,   3 )
   TKFRAME_-43501_UNITS          = 'DEGREES'
   
   FRAME_IMAP_HIT_L1_APERTURE_02 = -43502
   FRAME_-43502_NAME             = 'IMAP_HIT_L1_APERTURE_02'
   FRAME_-43502_CLASS            = 4
   FRAME_-43502_CLASS_ID         = -43502   
   FRAME_-43502_CENTER           = -43
   TKFRAME_-43502_RELATIVE       = 'IMAP_HIT'
   TKFRAME_-43502_SPEC           = 'ANGLES'
   TKFRAME_-43502_ANGLES         = ( 0,  116, 0 )
   TKFRAME_-43502_AXES           = ( 1,   2,   3 )
   TKFRAME_-43502_UNITS          = 'DEGREES'
   
   FRAME_IMAP_HIT_L1_APERTURE_03 = -43503
   FRAME_-43503_NAME             = 'IMAP_HIT_L1_APERTURE_03'
   FRAME_-43503_CLASS            = 4
   FRAME_-43503_CLASS_ID         = -43503   
   FRAME_-43503_CENTER           = -43
   TKFRAME_-43503_RELATIVE       = 'IMAP_HIT'
   TKFRAME_-43503_SPEC           = 'ANGLES'
   TKFRAME_-43503_ANGLES         = ( 0,  90, 0 )
   TKFRAME_-43503_AXES           = ( 1,   2,   3 )
   TKFRAME_-43503_UNITS          = 'DEGREES'
   
   FRAME_IMAP_HIT_L1_APERTURE_04 = -43504
   FRAME_-43504_NAME             = 'IMAP_HIT_L1_APERTURE_04'
   FRAME_-43504_CLASS            = 4
   FRAME_-43504_CLASS_ID         = -43504   
   FRAME_-43504_CENTER           = -43
   TKFRAME_-43504_RELATIVE       = 'IMAP_HIT'
   TKFRAME_-43504_SPEC           = 'ANGLES'
   TKFRAME_-43504_ANGLES         = ( 0,  64, 0 )
   TKFRAME_-43504_AXES           = ( 1,   2,   3 )
   TKFRAME_-43504_UNITS          = 'DEGREES'
   
   FRAME_IMAP_HIT_L1_APERTURE_05 = -43505
   FRAME_-43505_NAME             = 'IMAP_HIT_L1_APERTURE_05'
   FRAME_-43505_CLASS            = 4
   FRAME_-43505_CLASS_ID         = -43505   
   FRAME_-43505_CENTER           = -43
   TKFRAME_-43505_RELATIVE       = 'IMAP_HIT'
   TKFRAME_-43505_SPEC           = 'ANGLES'
   TKFRAME_-43505_ANGLES         = ( 0,  38, 0 )
   TKFRAME_-43505_AXES           = ( 1,   2,   3 )
   TKFRAME_-43505_UNITS          = 'DEGREES'
   
   FRAME_IMAP_HIT_L1_APERTURE_06 = -43506
   FRAME_-43506_NAME             = 'IMAP_HIT_L1_APERTURE_06'
   FRAME_-43506_CLASS            = 4
   FRAME_-43506_CLASS_ID         = -43506   
   FRAME_-43506_CENTER           = -43
   TKFRAME_-43506_RELATIVE       = 'IMAP_HIT'
   TKFRAME_-43506_SPEC           = 'ANGLES'
   TKFRAME_-43506_ANGLES         = ( 0, -38, 0 )
   TKFRAME_-43506_AXES           = ( 1,   2,   3 )
   TKFRAME_-43506_UNITS          = 'DEGREES'
   
   FRAME_IMAP_HIT_L1_APERTURE_07 = -43507
   FRAME_-43507_NAME             = 'IMAP_HIT_L1_APERTURE_07'
   FRAME_-43507_CLASS            = 4
   FRAME_-43507_CLASS_ID         = -43507   
   FRAME_-43507_CENTER           = -43
   TKFRAME_-43507_RELATIVE       = 'IMAP_HIT'
   TKFRAME_-43507_SPEC           = 'ANGLES'
   TKFRAME_-43507_ANGLES         = ( 0, -64, 0 )
   TKFRAME_-43507_AXES           = ( 1,   2,   3 )
   TKFRAME_-43507_UNITS          = 'DEGREES'
   
   FRAME_IMAP_HIT_L1_APERTURE_08 = -43508
   FRAME_-43508_NAME             = 'IMAP_HIT_L1_APERTURE_08'
   FRAME_-43508_CLASS            = 4
   FRAME_-43508_CLASS_ID         = -43508   
   FRAME_-43508_CENTER           = -43
   TKFRAME_-43508_RELATIVE       = 'IMAP_HIT'
   TKFRAME_-43508_SPEC           = 'ANGLES'
   TKFRAME_-43508_ANGLES         = ( 0, -90, 0 )
   TKFRAME_-43508_AXES           = ( 1,   2,   3 )
   TKFRAME_-43508_UNITS          = 'DEGREES'
   
   FRAME_IMAP_HIT_L1_APERTURE_09 = -43509
   FRAME_-43509_NAME             = 'IMAP_HIT_L1_APERTURE_09'
   FRAME_-43509_CLASS            = 4
   FRAME_-43509_CLASS_ID         = -43509   
   FRAME_-43509_CENTER           = -43
   TKFRAME_-43509_RELATIVE       = 'IMAP_HIT'
   TKFRAME_-43509_SPEC           = 'ANGLES'
   TKFRAME_-43509_ANGLES         = ( 0, -116, 0 )
   TKFRAME_-43509_AXES           = ( 1,   2,   3 )
   TKFRAME_-43509_UNITS          = 'DEGREES'
   
   FRAME_IMAP_HIT_L1_APERTURE_10 = -43510
   FRAME_-43510_NAME             = 'FRAME_IMAP_HIT_L1_APERTURE_10'
   FRAME_-43510_CLASS            = 4
   FRAME_-43510_CLASS_ID         = -43510   
   FRAME_-43510_CENTER           = -43
   TKFRAME_-43510_RELATIVE       = 'IMAP_HIT'
   TKFRAME_-43510_SPEC           = 'ANGLES'
   TKFRAME_-43510_ANGLES         = ( 0, -142, 0 )
   TKFRAME_-43510_AXES           = ( 1,   2,   3 )
   TKFRAME_-43510_UNITS          = 'DEGREES'
   
   
   \begintext


IMAP Interstellar Dust Experiment (IDEX) Frames
========================================================================
   
   IDEX is located on the -X side of the spacecraft as shown below. 
   
                                           -------------
                                          | S/C +Z axis |
        -----------------------            -------------
       | S/C +X axis into page |    #-----#    .
        -----------------------     | LGA |   /|\
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
             -------------       /_\      -----------        |__|
            | S/C +Y axis |    #-----#   | S/C FRAME |       STAR
             -------------     | MGA |   |  ORIGIN   |     TRACKERS
                               #-----#    -----------
         
   IDEX has the following nominal alignment to the spacecraft frame,
   reference Table 1 of [6]. The azimuth and elevation angles are 
   illustrated in the 'IMAP I&T Component Placement' section near the  
   top of this document.
   
        azimuth  | elevation 
         (deg)   |  (deg)
        ---------+---------
          +90    |   -45
                    
   The local IDEX frame is defined in [22]:
      *  +Y axis is the boresight, pointing outward through the opening along 
         the instrument axis of symmetry
      *  +Z axis is in the direction of S/C +Z
         
   [17] gives the measured value of the +Y axis of the IDEX coordinate
   system:
    
       +Y = [ 0.683178772 -0.185278978 -0.706355764 ]
       
   Instrument +Z is aligned with spacecraft +Z:
    
       +Z = [ 0 0 1 ]
       
   The IDEX +X axis is determined from the cross product. By adjusting the +Z
   axis and normalizing, we arrive at the following definition.
           
   \begindata

   FRAME_IMAP_IDEX              = -43700
   FRAME_-43700_NAME           = 'IMAP_IDEX'
   FRAME_-43700_CLASS          = 4
   FRAME_-43700_CLASS_ID       = -43700   
   FRAME_-43700_CENTER         = -43
   TKFRAME_-43700_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43700_SPEC         = 'MATRIX'
   TKFRAME_-43700_MATRIX       = ( 0.001346490209331
                                   0.999999093481647
                                   0.000000000000000
                                  -0.707321249357286
                                   0.000952402000481
                                  -0.706891606357069
                                  -0.706890965546854
                                   0.000951822627018
                                   0.707321890557561 )

   \begintext
   
   The following frames are defined as identity offsets off the IDEX frame.
   
              
   \begindata

   FRAME_IMAP_IDEX_DETECTOR    = -43701
   FRAME_-43701_NAME           = 'IMAP_IDEX_DETECTOR'
   FRAME_-43701_CLASS          = 4
   FRAME_-43701_CLASS_ID       = -43701  
   FRAME_-43701_CENTER         = -43
   TKFRAME_-43701_RELATIVE     = 'IMAP_IDEX'
   TKFRAME_-43701_SPEC         = 'MATRIX'
   TKFRAME_-43701_MATRIX       = ( 1.0
                                   0.0
                                   0.0
                                   0.0
                                   1.0
                                   0.0
                                   0.0
                                   0.0
                                   1.0 )
                                   
   FRAME_IMAP_IDEX_FULL_SCIENCE= -43702
   FRAME_-43702_NAME           = 'IMAP_IDEX_FULL_SCIENCE'
   FRAME_-43702_CLASS          = 4
   FRAME_-43702_CLASS_ID       = -43702
   FRAME_-43702_CENTER         = -43
   TKFRAME_-43702_RELATIVE     = 'IMAP_IDEX'
   TKFRAME_-43702_SPEC         = 'MATRIX'
   TKFRAME_-43702_MATRIX       = ( 1.0
                                   0.0
                                   0.0
                                   0.0
                                   1.0
                                   0.0
                                   0.0
                                   0.0
                                   1.0 )

   \begintext
   

IMAP GLObal solar Wind Structure (GLOWS) Frames
========================================================================
    
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
      *  +Y axis points in the anti-sunward direction (towards S/C -Z)
         
   A diagram of the GLOWS local coordinate system is shown below.
                 ______________________                 
                |                      |              .-'|       __ -Z'
     S/C +Z     |                      |           .-'   |    _.-*/    
    (sunward)   |                      |        .-'      |.-*'
        ^       |                      |    _.-*'\  _.-*'|
        |       |                      '.-*'   _.-*'     |
        |       |                      \    o*'    \ _.-'
        |       |                       \    \_.-*' '
                |                        \.-*'\         
                |________________________|     \
                                                \
                                                 v  +Y'                                    
      
   The azimuth and elevation give the nominal outward axis of symmetry, 
   -Z in the instrument frame:
    
       Z      = -[ -sin(az) * cos(el), cos(az) * cos(el), sin(el) ]
        instr  
    
   The alignment report [17] gives the measured outward axis of symmetry, 
   -Z in the instrument frame:
    
      -Z      = [ -0.7699232700, -0.5831000067, 0.2592538148 ]
        instr
      
   The instrument +Y axis is in the anti-sunward direction, towards the
   spacecraft -Z axis:

       Y      = [ 0 0 -1 ]    
        instr    
           
   Taking the cross product and normalizing, we arrive at the instrument +X
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

   FRAME_IMAP_GLOWS            = -43750
   FRAME_-43750_NAME           = 'IMAP_GLOWS'
   FRAME_-43750_CLASS          = 4
   FRAME_-43750_CLASS_ID       = -43750   
   FRAME_-43750_CENTER         = -43
   TKFRAME_-43750_RELATIVE     = 'IMAP_SPACECRAFT'
   TKFRAME_-43750_SPEC         = 'MATRIX'
   TKFRAME_-43750_MATRIX       = ( 0.603742428089898
                                  -0.797179453149737
                                   0.000000000000000
                                  -0.206671814310344
                                  -0.156522527639751
                                  -0.965809225215277
                                   0.769923270000000
                                   0.583100006700000
                                  -0.259253814800000 )

\begintext

End of FK file.    