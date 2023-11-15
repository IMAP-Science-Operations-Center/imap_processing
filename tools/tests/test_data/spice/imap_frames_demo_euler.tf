KPL/FK

Insterstellar Mapping and Acceleration Probe Frames Kernel
========================================================================

   This frames kernel contains the current set of coordinate frame
   definitions for the IMAP spacecraft, and Ultra Instruments.

Version and Date
========================================================================

   The TEXT_KERNEL_ID stores version information of loaded project text
   kernels. Each entry associated with the keyword is a string that consists
   of four parts: the kernel name, version, entry date, and type. For example,
   the frames kernel might have an entry as follows:

           TEXT_KERNEL_ID += 'SPP_FRAMES V1.0.0 25-JUNE-2018 FK'
                                  |         |        |        |
                                  |         |        |        |
              KERNEL NAME <-------+         |        |        |
                                            |        |        V
                             VERSION <------+        |   KERNEL TYPE
                                                     |
                                                     V
                                                ENTRY DATE

   Parker Solar Probe Frames Kernel Version:

   \begindata

   TEXT_KERNEL_ID += 'IMAP_FRAMES V1.0.0 23-DEC-2022 FK'

   NAIF_BODY_NAME += ( 'IMAP' )
   NAIF_BODY_CODE += ( -43 )

   \begintext


   Version 0.0.1 -- Nov. 30, 2022 -- Nick Dutton, JHU/APL


       Initial Version for Spacecraft, Ultra


References
========================================================================

   1. 'Frames Required Reading'

   2. 'Kernel Pool Required Reading'

   3. 'C-Kernel Required Reading'

   4. 'IMAP Mapping.pdf', received from Matina G., JHU/APL


Contact Information
========================================================================

   Lillian Nguyen, JHU/APL,
   (443)778-5477, Lillian.Nguyen@jhuapl.edu

   Scott Turner, JHU/APL
   (443)778-1693, Scott.Turner@jhuapl.edu

   M. Alexandra Matiella Novak, JHU/APL,
   (443)802-1417, Alexandra.Matiella.Novak@jhuapl.edu

   Wen-Jong Shyong, JHU/APL
   (443)778-8564, Wen-Jong.Shyong@jhuapl.edu


Implementation Notes
========================================================================

   This file is used by the SPICE system as follows: programs that make
   use of this frame kernel must `load' the kernel, normally during
   program initialization. Loading the kernel associates the data items with
   their names in a data structure called the `kernel pool'.  The SPICELIB
   routine FURNSH loads a kernel into the pool as shown below:

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


IMAP Frames
========================================================================

   The ID codes -43000 to -43999 have been reserved for the Interstellar
   Mapping and Acceleration Probe frames kernel.

   The following frames are defined in this kernel file:

      Frame Name                Relative To              Type      NAIF ID
      =======================   ===================      =======   =======

      Spacecraft Frames:
      ------------------
      IMAP_BODY                 ECLIPJ2000               CK        -43000

      ULTRA Inst Frames:
      ------------------
      IMAP_ULTRA90              IMAP_BODY                FIXED     -43002
      IMAP_ULTRA45              IMAP_BODY                FIXED     -43003

      Hi Inst Frames:
      ---------------
      IMAP_HI90                IMAP_BODY                 FIXED     -43051
      IMAP_HI45                IMAP_BODY                 FIXED     -43052

      Lo Inst Frames:
      ---------------
      IMAP_LO                  IMAP_BODY                 FIXED     -43030

      MAG Frames (TBD):
      -------------
      ID codes -43200  to -43299

      SWE Frames (TBD):
      --------------------
      ID codes -43300  to -43399

      SWAPI Frames (TBD):
      --------------------
      ID codes -43400  to -43499

      CODICE Frames (TBD):
      --------------------
      ID codes -43500  to -43599

      HIT Frames (TBD):
      --------------------
      ID codes -43600  to -43699

      IDEX Frames (TBD):
      --------------------
      ID codes -43700  to -43799

      GLOWS Frames (TBD):
      --------------------
      ID codes -43800  to -43899



IMAP Frame Tree
========================================================================

   The diagram below illustrates the IMAP frame hierarchy:

   TBD


Spacecraft Frames
========================================================================

   The orientation of the spacecraft body frame with respect to an inertial
   frame, for IMAP - ECLIPJ2000, is provided by a C-kernel (see [3]
   for details).

   This frame specifies the rotating X,Y and pointing Z coordinate body
   frame.

   \begindata

   FRAME_IMAP_BODY              = -43000
   FRAME_-43000_NAME           = 'IMAP_BODY'
   FRAME_-43000_CLASS          = 4
   FRAME_-43000_CLASS_ID       = -43000
   FRAME_-43000_CENTER         = -43
   CK_-43000_SCLK              = -43
   CK_-43000_SPK               = -43
   TKFRAME_-43000_SPEC         = 'MATRIX'
   TKFRAME_-43000_MATRIX       = ( 1,
                                   0,
                                   0,
                                   0,
                                   1,
                                   0,
                                   0,
                                   0,
                                   1 )
   TKFRAME_-43000_RELATIVE     = 'IMAP_SPIN'

   \begintext

Euler / Spinning Frame definition
==================================================================

   Euler definition assuming nominal 4 rpm rotation
   (15 seconds per rotation); no nutation.  We rotate 24 degrees
   per second about the +Z axis in the right-handed sense (CCW if
   looking in the -Z direction).

   Initial Freeze epoch is given in TDB.  The value is
   2025-APR-15/00:00:00.0 in UTC

   \begindata

   FRAME_IMAP_SPIN             = -43005
   FRAME_-43005_NAME           = 'IMAP_SPIN'
   FRAME_-43005_CLASS          = 5
   FRAME_-43005_CLASS_ID       = -43005
   FRAME_-43005_CENTER         = -43
   FRAME_-43005_RELATIVE       = 'IMAP_DPS'
   FRAME_-43005_DEF_STYLE      = 'PARAMETERIZED'
   FRAME_-43005_FAMILY         = 'EULER'
   FRAME_-43005_EPOCH          = @2025-APR-15/00:01:09.185
   FRAME_-43005_AXES           = (1,2,3)
   FRAME_-43005_UNITS          = 'DEGREES'
   FRAME_-43005_ANGLE_1_COEFFS = (0, 0)
   FRAME_-43005_ANGLE_2_COEFFS = (0, 0)
   FRAME_-43005_ANGLE_3_COEFFS = (0, -24)

   \begintext

Dynamic IMAP SUN Frame Definition
===============================================================
Note the freeze epoch must be overwritten as defined for each Pointing
Initial Freeze epoch is given in TDB.  The value is 2025-APR-15/00:00:00.0
in UTC

\begindata
         FRAME_IMAP_SUN                = -43006
         FRAME_-43006_NAME             = 'IMAP_SUN'
         FRAME_-43006_CLASS            = 5
         FRAME_-43006_CLASS_ID         = -43006
         FRAME_-43006_CENTER           = -43
         FRAME_-43006_RELATIVE         = 'ECLIPJ2000'
         FRAME_-43006_DEF_STYLE        = 'PARAMETERIZED'
         FRAME_-43006_FAMILY           = 'TWO-VECTOR'
         FRAME_-43006_PRI_AXIS         = 'Z'
         FRAME_-43006_PRI_VECTOR_DEF   = 'OBSERVER_TARGET_POSITION'
         FRAME_-43006_PRI_OBSERVER     = -43
         FRAME_-43006_PRI_TARGET       = 'SUN'
         FRAME_-43006_PRI_ABCORR       = 'NONE'
		 FRAME_-43006_PRI_FRAME 	   = 'ECLIPJ2000'
         FRAME_-43006_SEC_AXIS         = 'X'
         FRAME_-43006_SEC_VECTOR_DEF   = 'CONSTANT'
         FRAME_-43006_SEC_SPEC   	   = 'RECTANGULAR'
         FRAME_-43006_SEC_FRAME        = 'ECLIPJ2000'
         FRAME_-43006_SEC_VECTOR 	   = (0,0,1)
         FRAME_-43006_FREEZE_EPOCH     = @2025-APR-15/00:01:09.185


\begintext

IMAP DPS frame - a rotation of the IMAP SUN frame 4 degrees off the sun
=======================================================================

\begindata

         FRAME_IMAP_DPS                = -43007
         FRAME_-43007_NAME             = 'IMAP_DPS'
         FRAME_-43007_CLASS            = 4
         FRAME_-43007_CLASS_ID         = -43007
         FRAME_-43007_CENTER           = -43
         TKFRAME_-43007_SPEC           = 'ANGLES'
         TKFRAME_-43007_UNITS          = 'DEGREES'
         TKFRAME_-43007_AXES           = (1, 2, 3)
         TKFRAME_-43007_ANGLES         = (4.0, 0.0, 0.0)
         TKFRAME_-43007_RELATIVE       = 'IMAP_SUN'


\begintext

ULTRA 90
========================================================================

   IMAP Ultra 90 Nominal Alignment [ref: IMAP Mapping Doc]

   Ultra 90's boresight is in the IMAP Body X-Y plane and 168 degrees
   from the +Y body axis.  We define its frame orientation by a series of
   Euler Angle rotations.

   In the body frame, +Z points into the page, X and Y are 90 degrees
   apart in a plane perpendicular to the Z axis.

   The first rotation is 90 degrees about the body +X direction CW
   if looking in the -X direction.  This orients the +Z' angle with the
   body frame +Y and +Y' pointing out of the page.  The next rotation
   is about the +Y' axis by the offset angle 168 degrees and aligns
   +Z'' with the instrument boresight and offsets +X' = +X by 168 degrees
   as well.  The Final rotation is about the +Z'', again CW if looking in
   the -Z'' direction, about 90 degrees to put the +Y'' and +X'' in the
   proper instrument frame orientation. +Y''' points in the opposite
   direction of the spacecrafts rotation.


   \begindata

   FRAME_IMAP_ULTRA90          = -43002
   FRAME_-43002_NAME           = 'IMAP_ULTRA90'
   FRAME_-43002_CLASS          = 4
   FRAME_-43002_CLASS_ID       = -43002
   FRAME_-43002_CENTER         = -43
   TKFRAME_-43002_SPEC         = 'ANGLES'
   TKFRAME_-43002_ANGLES       = (90, 168.0, 90.0)
   TKFRAME_-43002_AXES         = (1, 2, 3)
   TKFRAME_-43002_UNITS        = 'DEGREES'
   TKFRAME_-43002_RELATIVE     = 'IMAP_BODY'

   \begintext

ULTRA 45
========================================================================


   IMAP Ultra 45 Nominal Alignment [ref: IMAP Mapping Doc]

   Ultra 45's boresight is 45 degrees off of the IMAP Body X-Y plane and
   -12 degrees from the +Y body axis.

   We define its frame orientation by a series of Euler Angle rotations.

   In the body frame, +Z points into the page, X and Y are 90 degrees
   apart in a plane perpendicular to the Z axis.

   The first rotation is 12 degrees about the body +Z direction CW
   if looking in the -Z direction.  This orients the +Y' angle 12 degrees
   off the with the body frame +Y and +X' 12 degrees off of the body frame
   +X.  The next rotation is about the +X' axis by the offset 135 degrees
   and aligns +Z'' with the instrument boresight. The Final rotation is
   about the +Z'', CW if looking in the -Z'' direction, 90 degrees to put
   the +Y'''' and +X'''' in the proper instrument frame orientation. +Y'''
   points in the opposite direction of the spacecrafts rotation.

   \begindata

   FRAME_IMAP_ULTRA45          = -43003
   FRAME_-43003_NAME           = 'IMAP_ULTRA45'
   FRAME_-43003_CLASS          = 4
   FRAME_-43003_CLASS_ID       = -43003
   FRAME_-43003_CENTER         = -43
   TKFRAME_-43003_SPEC         = 'ANGLES'
   TKFRAME_-43003_ANGLES       = (12, 135, 90.0)
   TKFRAME_-43003_AXES         = (3, 1, 3)
   TKFRAME_-43003_UNITS        = 'DEGREES'
   TKFRAME_-43003_RELATIVE     = 'IMAP_BODY'


   \begintext


IMAP-Hi90 Frames
===========================================================================

TBD

