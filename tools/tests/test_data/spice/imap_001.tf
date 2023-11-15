KPL/FK

IMAP Frames Kernel
===========================================================================

   This frame kernel contains the current working set of frame definitions
   for the Interstellar Mapping and Acceleration Probe (IMAP) including 
   definitions for the IMAP structures and IMAP science instrument frames. 
   This kernel also contains NAIF ID/name mapping for the IMAP instruments.
   This kernel is currently a work in progress and will be updated as
   required when information from various subsystems and instrument teams
   becomes available.


Version and Date
------------------------------------------------------------------------                                

   Version 001 -- July 9, 2021 -- Ian Wick Murphy, JHU/APL
   
      Modyfing dart_008.tf to add basic IMAP frame components. This includes
      IMAP, IMAP_THRUSTER, and CK/SCLK IDs. Also adding a place holder for
      the IMAP-Lo instrument with the ID -43001 and IMAP_LO name.
      Future work includes adding more detailed instrument frames, and 
      reaching out to mechanical for an "official" IMAP_SPACECRAFT frame
      definition.

 < Please add versioning information and udpate logs as this kernel is  >
 < update, i.e:
 < Version ### -- Month, Day Year -- Your Name, Institution             >
 <    Some wonderful comments about your modifications/additions        >
   

References
------------------------------------------------------------------------

   1. ``Frames Required Reading''

   2. ``Kernel Pool Required Reading''

   3. ``C-Kernel Required Reading''


Contact Information
------------------------------------------------------------------------

   Ian Wick Murphy JHU/APL, ian.murphy@jhuapl.edu
   < Please add your name and contact here if you add information to this >
   < kernel.                                                              > 


Implementation Notes
------------------------------------------------------------------------

   This file is used by the SPICE system as follows: programs that make
   use of this kernel must ``load'' the kernel, normally during program
   initialization. The SPICE routine FURNSH loads a kernel file into
   the pool as shown below.

      CALL FURNSH ( 'kernel_name; )    -- FORTRAN
      furnsh_c ( "kernel_name" );      -- C
      cspice_furnsh, kernel_name       -- IDL
      cspice_furnsh( 'kernel_name' )   -- MATLAB

   In order for a program or routine to extract data from the pool, the
   SPICELIB routines GDPOOL, GIPOOL, and GCPOOL are used.  See [2] for
   more details.

   This file was created and may be updated with a text editor or word
   processor.


IMAP NAIF ID Codes -- Summary Section
------------------------------------------------------------------------

   The following names and NAIF ID codes are assigned to the IMAP s/c, 
   its structures and science instruments (the keywords implementing
   these definitions are located in the section "IMAP NAIF ID Codes --
   Definition Section" at the end of this file):

   IMAP and IMAP Structures names/IDs:
   -----------------------------------
            IMAP                     -43   
            
            IMAP_LO       	     -43001           

            IMAP_THRUSTER_A1         -43201
            IMAP_THRUSTER_A2         -43202
            IMAP_THRUSTER_A3         -43203
            IMAP_THRUSTER_A4         -43204
            IMAP_THRUSTER_R1         -43211
            IMAP_THRUSTER_R2         -43212
            IMAP_THRUSTER_R3         -43213
            IMAP_THRUSTER_R4         -43214
            IMAP_THRUSTER_R5         -43215
            IMAP_THRUSTER_R6         -43216
            IMAP_THRUSTER_R7         -43217
            IMAP_THRUSTER_R8         -43218


IMAP Frames
------------------------------------------------------------------------

   The following IMAP frames are defined in this kernel file:

           Name                    Relative to           Type        NAIF ID
      ======================    ===================  ============    =======

   IMAP Spacecraft and Spacecraft Structures frames:
   -------------------------------------------------
      IMAP_SPACECRAFT             J2000                  CK          -43000
      
      IMAP_LO 		          IMAP_SPACECRAFT        CK          -43001
      

IMAP Frames Hierarchy
------------------------------------------------------------------------

  The diagram below shows the IMAP spacecraft, its structures, and its
  science instrument frame hierarchy.


                               "J2000" INERTIAL
         +-------------------------------------------------------+
         |                          |
         |<-pck                     |<-ck
         V                          |
    "EARTH_FIXED"                   |
    -------------                   |
                                    |
                                    |
                                    |
                                    |
                                    |
                                    |
                                    V               
                             "IMAP_SPACECRAFT"      
              		     -----------------  

                                 

IMAP Spacecraft and Spacecraft Structures Frames
--------------------------------------------------------------------------------

   This section of the file contains the definitions of the spacecraft
   and spacecraft structures frames.


IMAP Spacecraft Bus Frame
-----------------------------------------------------------

   The spacecraft bus frame -- IMAP_SPACECRAFT -- is defined by the s/c
   design as follows:

      - < insert +X axis definition >

      - < insert +Y axis definition >

      - < insert +Z axis definition >

      - < insert coordinate frame origin definition >


   These diagrams illustrate the s/c frame:

   +X side view:
   -------------
	< insert some fun ASCII art of +X >

   +Z side view:
   ---------------------------
	< insert some fun ASCII art of +Z >
						 
   -X side view:
   -------------     
	< insert some fun ASCII art of -X >

   +Y side view:
   -------------
	< insert some fun ASCII art of +Y >
                                                

   -Z side view:
   -----------------------
	< insert some fun ASCII art of -Z >
						 

   Since the S/C bus attitude is provided by a C kernel (see [3] for
   more information), this frame is defined as a CK-based frame.

   \begindata

        FRAME_IMAP_SPACECRAFT       = -43000
        FRAME_-43000_NAME           = 'IMAP_SPACECRAFT'
        FRAME_-43000_CLASS          = 3
        FRAME_-43000_CLASS_ID       = -43000   
        FRAME_-43000_CENTER         = -43
        CK_-43000_SCLK              = -43
        CK_-43000_SPK               = -43

   \begintext

Thruster positions
------------------------------------------------------------------------

There are four axial (A) thrusters and 8 radial (R) thrusters on IMAP. The
positions of the nozzle exit in mm from the spacecraft center of mass
in the spacecraft frame are tabulated below. This information is
captured in the IMAP structure SPK.


      Component          ID       X (mm)    Y (mm)   Z (mm)
      ----------------   -------  ------    ------   ------

      IMAP_THRUSTER_A1   -43201  1007.28    516.50   667.23
      IMAP_THRUSTER_A2   -43202 -1007.28   -516.50   667.23
      IMAP_THRUSTER_A3   -43203 -1007.29   -516.51  -543.40
      IMAP_THRUSTER_A4   -43204  1007.29    516.51  -543.40

      IMAP_THRUSTER_R1   -43211  -126.90   1237.78   195.95
      IMAP_THRUSTER_R2   -43212   126.90  -1237.78   195.95
      IMAP_THRUSTER_R3   -43213 -1008.49    728.79   195.95
      IMAP_THRUSTER_R4   -43214  1008.49   -728.79   195.95
      IMAP_THRUSTER_R5   -43215  -126.90   1237.78  -197.75 
      IMAP_THRUSTER_R6   -43216   126.90  -1237.78  -197.75 
      IMAP_THRUSTER_R7   -43217 -1008.49    728.79  -197.75 
      IMAP_THRUSTER_R8   -43218  1008.49   -728.79  -197.75 



IMAP NAIF ID Codes -- Definitions
=====================================================================

   This section contains name to NAIF ID mappings for the IMAP mission.
   Once the contents of this file are loaded into the KERNEL POOL, these
   mappings become available within SPICE, making it possible to use
   names instead of ID code in high level SPICE routine calls.
   
   \begindata
   
      NAIF_BODY_NAME   += ( 'IMAP' )                        
      NAIF_BODY_CODE   += ( -43    )


      NAIF_BODY_NAME   += ( 'IMAP_SPACECRAFT' )       
      NAIF_BODY_CODE   += ( -43000            )
      
      NAIF_BODY_NAME   += ( 'IMAP_LO'         )       
      NAIF_BODY_CODE   += ( -43001            )

      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_A1' )
      NAIF_BODY_CODE   += ( -43201             )
      
      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_A2' )
      NAIF_BODY_CODE   += ( -43202             )
      
      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_A3' )
      NAIF_BODY_CODE   += ( -43203             )
      
      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_A4' )
      NAIF_BODY_CODE   += ( -43204             )
      
      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R1' )
      NAIF_BODY_CODE   += ( -43211             )
      
      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R2' )
      NAIF_BODY_CODE   += ( -43212             )
      
      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R3' )
      NAIF_BODY_CODE   += ( -43213             )
      
      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R4' )
      NAIF_BODY_CODE   += ( -43214             )
      
      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R5' )
      NAIF_BODY_CODE   += ( -43215             )
      
      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R6' )
      NAIF_BODY_CODE   += ( -43216             )
      
      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R7' )
      NAIF_BODY_CODE   += ( -43217             )
      
      NAIF_BODY_NAME   += ( 'IMAP_THRUSTER_R8' )
      NAIF_BODY_CODE   += ( -43218             )
      
      
   \begintext
   

End of FK file.    