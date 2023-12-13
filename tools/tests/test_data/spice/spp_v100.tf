KPL/FK

Parker Solar Probe Frames Kernel
========================================================================

   This frames kernel contains the current set of coordinate frame
   definitions for the Parker Solar Probe spacecraft, structures, and
   science instruments.

   To be consistent with conventions prior to the mission name change
   from Solar Probe Plus (SPP) to Parker Solar Probe (PSP) in 2017, frame
   names and other text IDs are prefixed with the legacy acronym, SPP.

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

   TEXT_KERNEL_ID += 'SPP_FRAMES V1.0.0 18-DEC-2018 FK'

   NAIF_BODY_NAME += ( 'SPP' )
   NAIF_BODY_CODE += ( -96 )

   \begintext


   Version 1.0.0 -- Dec. 18, 2018 -- Lillian Nguyen
                                     Scott Turner
                                     Alexandra Matiella Novak

       Added EPI-Lo frames.
       Added WISPR frames.
       Added NAIF body name to ID mapping for the spacecraft.
       Added comments (references, frame table, frame tree, etc.)

   Version 0.0.1 -- May 12, 2017 -- Wen-Jong Shyong

       Spacecraft, high gain antenna, and solar panel frames.


References
========================================================================

   1. 'Frames Required Reading'

   2. 'Kernel Pool Required Reading'

   3. 'C-Kernel Required Reading'

   4. 'SPP_ISIS_EPI_lo_mechanical_ICD_7464-0008.pdf', received from Matt Hill,
      JHU/APL

   5. 'EPI_Lo_collimators_FOV_mechanical_SteveLayman_drawings_pathlength_SSD_
      foil_distances_angles_for_GEANT_7464-matt-hill_b.pdf', received from
      Matt Hill, JHU/APL

   6. 'EPI-Lo_wedge_aperture_naming_orientation_2016_11_18.pptx', received
      from Matt Hill, JHU/APL

   7. '7434-0011.pdf', from JHU/APL engineering database PLM Windchill

   8. '16105 1001 EM ISIS MICD 14-08-13 Rev-.pdf', received from Alexandra
       Dupont, JHU/APL

   9. 'EPI LO FOV.pptx', received from Steve Layman, JHU/APL

   10. 'epiLO.xlsx', received from Chris Choi, JHU/APL

   11. 'EPI_LO_FOV_2014_07_29_2014_09_09.xlsx', received from Matt Hill,
       JHU/APL

   12. Email from Martha Kusterer regarding frame IDs, received 2/7/2018.

   13. '7434-9056_Rev_D.pdf', received from Martha Kusterer on 5/29/2018.

   14. Email from Roberto Livi confirming SWEAP FOVs as described in Appendix
       A of [13], received 7/13/2018.

   15. Frames kernel spp_v004.tf, having SPICE TEXT_KERNEL_ID
       'SPP V0.0.4 26-JUL-2016 FK', received from Angelos Vourlidas on
       5/29/2018

   16. 'PSPalignment020718-v29b.pptx', received from Chris Choi 5/29/2018.

   17. 'spp_001.tf', received 9/12/18 from Wen-Jong Shyong

   18. 'spp_v004', kernel ID 'SPP V0.0.4 26-JUL-2016 FK', received 5/29/18
       from Angelos Vourlidas

   19. Email from Angelos Vourlidas containing updates boresights,
       received 9/11/18.

   20. "PSP Kernel Review EPI-Lo WISPR.pptx", Lillian Nguyen 12/12/2018

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


SPP Frames
========================================================================

   The ID codes -96900 to -96999 have been reserved for the Parker Solar
   Probe dynamics frames kernel [12] and are not utilized in this file.

   The following frames are defined in this kernel file:

      Frame Name                Relative To              Type      NAIF ID
      =======================   ===================      =======   =======

      Spacecraft Frames:
      ------------------
      SPP_SPACECRAFT            J2000                    CK        -96000

      Solar Array Frames:
      -------------------
      SPP_SOLARPANEL_PLUS       SPP_SPACECRAFT           CK        -96001
      SPP_SP_PLUS_BASE          SPP_SPACECRAFT           FIXED     -96011
      SPP_SOLARPANEL_MINUS      SPP_SPACECRAFT           CK        -96002
      SPP_SP_MINUS_BASE         SPP_SPACECRAFT           FIXED     -96012

      Antenna Frames:
      ---------------
      SPP_HIGH_GAIN_ANTENNA     SPP_SPACECRAFT           CK        -96003
      SPP_HGA_BASE              SPP_SPACECRAFT           FIXED     -96013

      Spacecraft Deck Frames:
      -----------------------
      SPP_DECK_1                SPP_SPACECRAFT           FIXED     -96081
      SPP_DECK_2                SPP_SPACECRAFT           FIXED     -96082
      SPP_DECK_3                SPP_SPACECRAFT           FIXED     -96083
      SPP_DECK_4                SPP_SPACECRAFT           FIXED     -96084
      SPP_DECK_5                SPP_SPACECRAFT           FIXED     -96085
      SPP_DECK_6                SPP_SPACECRAFT           FIXED     -96086

      WISPR Frames:
      -------------
      SPP_WISPR_INNER           SPP_SPACECRAFT           FIXED     -96100
      SPP_WISPR_OUTER           SPP_SPACECRAFT           FIXED     -96120

      SWEAP Frames (TBD):
      -------------
      ID codes -96200  to -96299

      FIELDS Frames (TBD):
      --------------------
      ID codes -96300  to -96399

      ISOIS EPI-Lo Frames:
      --------------------
      SPP_EPILO_BASE            SPP_DECK_1               FIXED     -96401
      SPP_EPILO_W0              SPP_EPILO_BASE           FIXED     -96411
      SPP_EPILO_L00             SPP_EPILO_W0             FIXED     -96412
      SPP_EPILO_L01             SPP_EPILO_W0             FIXED     -96413
      SPP_EPILO_L02             SPP_EPILO_W0             FIXED     -96414
      SPP_EPILO_L03             SPP_EPILO_W0             FIXED     -96415
      SPP_EPILO_L04             SPP_EPILO_W0             FIXED     -96416
      SPP_EPILO_L05             SPP_EPILO_W0             FIXED     -96417
      SPP_EPILO_L06             SPP_EPILO_W0             FIXED     -96418
      SPP_EPILO_L07             SPP_EPILO_W0             FIXED     -96419
      SPP_EPILO_L08             SPP_EPILO_W0             FIXED     -96420
      SPP_EPILO_L09             SPP_EPILO_W0             FIXED     -96421
      SPP_EPILO_W1              SPP_EPILO_BASE           FIXED     -96422
      SPP_EPILO_L10             SPP_EPILO_W1             FIXED     -96423
      SPP_EPILO_L11             SPP_EPILO_W1             FIXED     -96424
      SPP_EPILO_L12             SPP_EPILO_W1             FIXED     -96425
      SPP_EPILO_L13             SPP_EPILO_W1             FIXED     -96426
      SPP_EPILO_L14             SPP_EPILO_W1             FIXED     -96427
      SPP_EPILO_L15             SPP_EPILO_W1             FIXED     -96428
      SPP_EPILO_L16             SPP_EPILO_W1             FIXED     -96429
      SPP_EPILO_L17             SPP_EPILO_W1             FIXED     -96430
      SPP_EPILO_L18             SPP_EPILO_W1             FIXED     -96431
      SPP_EPILO_L19             SPP_EPILO_W1             FIXED     -96432
      ...
      SPP_EPILO_W7              SPP_EPILO_BASE           FIXED     -96488
      SPP_EPILO_L70             SPP_EPILO_W7             FIXED     -96489
      SPP_EPILO_L71             SPP_EPILO_W7             FIXED     -96490
      SPP_EPILO_L72             SPP_EPILO_W7             FIXED     -96491
      SPP_EPILO_L73             SPP_EPILO_W7             FIXED     -96492
      SPP_EPILO_L74             SPP_EPILO_W7             FIXED     -96493
      SPP_EPILO_L75             SPP_EPILO_W7             FIXED     -96494
      SPP_EPILO_L76             SPP_EPILO_W7             FIXED     -96495
      SPP_EPILO_L77             SPP_EPILO_W7             FIXED     -96496
      SPP_EPILO_L78             SPP_EPILO_W7             FIXED     -96497
      SPP_EPILO_L79             SPP_EPILO_W7             FIXED     -96498

      NOTE: ID codes -96400 through -96699 are reserved for EPI-Lo frames
      and fields of view.

      ISOIS EPI-Hi Frames (TBD):
      --------------------------
      -96700  to -96799


SPP Frame Tree
========================================================================

   The diagram below illustrates the SPP frame hierarchy:

   J2000
      |
      |<---ck
      |
      SPP_SPACECRAFT
          |
          SPP_SP_PLUS_BASE
          |
          |<---ck
          |
          SPP_SOLARPANEL_PLUS
          |
          SPP_SP_MINUS_BASE
          |
          |<---ck
          |
          SPP_SOLARPANEL_MINUS
          |
          SPP_HGA_BASE
          |
          |<---ck
          |
          SPP_HIGH_GAIN_ANTENNA
          |
          SPP_DECK_1
          |    |
          |    SPP_EPILO_BASE
          |         |
          |         SPP_EPILO_W0
          |         |    |
          |         |    SPP_EPILO_L00
          |         |    |
          |         |    SPP_EPILO_L01
          |         |    |
          |         |    ...
          |         |    SPP_EPILO_L09
          |         |
          |         SPP_EPILO_W1
          |         |    |
          |         |    SPP_EPILO_L10
          |         |    |
          |         |    SPP_EPILO_L11
          |         |    |
          |         |    ...
          |         |    SPP_EPILO_L19
          |         |
          |         ...
          |         SPP_EPILO_W7
          |              |
          |              SPP_EPILO_L70
          |              |
          |              SPP_EPILO_L71
          |              |
          |              ...
          |              SPP_EPILO_L79
          |
          SPP_DECK_2
          |
          SPP_DECK_3
          |
          SPP_DECK_4
          |
          SPP_DECK_5
          |
          SPP_DECK_6
          |
          SPP_WISPR_INNER
          |
          SPP_WISPR_OUTER


Spacecraft Frames
========================================================================

   The orientation of the spacecraft body frame with respect to an inertial
   frame, typically J2000, is provided by a C-kernel (see [3] for details).

   \begindata

   FRAME_SPP_SPACECRAFT        = -96000
   FRAME_-96000_NAME           = 'SPP_SPACECRAFT'
   FRAME_-96000_CLASS          = 3
   FRAME_-96000_CLASS_ID       = -96000
   FRAME_-96000_CENTER         = -96
   CK_-96000_SCLK              = -96
   CK_-96000_SPK               = -96

   \begintext


Solar Array Frames
========================================================================

   Solar Array Reference Frames and Nominal Alignments [17]

   The default transformation matrix for the solar array frames to the
   S/C body frame for flap angle of 0 degree and feather angle of 0 degree
   is:

         [X]      [ -1   0   0 ] [X]
         [Y]    = [  0  -1   0 ] [Y]
         [Z]S/C   [  0   0   1 ] [Z]+Y SA

         [X]      [  1   0   0 ] [X]
         [Y]    = [  0   1   0 ] [Y]
         [Z]S/C   [  0   0   1 ] [Z]-Y SA

   The orientation of the solar array frames with respect to an inertial
   frame, typically J2000, is provided by a C-kernel (see [3] for details).

   \begindata

   FRAME_SPP_SP_PLUS_BASE      = -96011
   FRAME_-96011_NAME           = 'SPP_SP_PLUS_BASE'
   FRAME_-96011_CLASS          = 4
   FRAME_-96011_CLASS_ID       = -96011
   FRAME_-96011_CENTER         = -96
   TKFRAME_-96011_SPEC         = 'MATRIX'
   TKFRAME_-96011_RELATIVE     = 'SPP_SPACECRAFT'
   TKFRAME_-96011_MATRIX       = ( 1,
                                   0,
                                   0,
                                   0,
                                   1,
                                   0,
                                   0,
                                   0,
                                   1 )

   FRAME_SPP_SOLARPANEL_PLUS   = -96001
   FRAME_-96001_NAME           = 'SPP_SOLARPANEL_PLUS'
   FRAME_-96001_CLASS          = 3
   FRAME_-96001_CLASS_ID       = -96001
   FRAME_-96001_CENTER         = -96
   CK_-96001_SCLK              = -96
   CK_-96001_SPK               = -96

   FRAME_SPP_SP_MINUS_BASE     = -96012
   FRAME_-96012_NAME           = 'SPP_SP_MINUS_BASE'
   FRAME_-96012_CLASS          = 4
   FRAME_-96012_CLASS_ID       = -96012
   FRAME_-96012_CENTER         = -96
   TKFRAME_-96012_SPEC         = 'MATRIX'
   TKFRAME_-96012_RELATIVE     = 'SPP_SPACECRAFT'
   TKFRAME_-96012_MATRIX       = ( 1,
                                   0,
                                   0,
                                   0,
                                   1,
                                   0,
                                   0,
                                   0,
                                   1 )

   FRAME_SPP_SOLARPANEL_MINUS  = -96002
   FRAME_-96002_NAME           = 'SPP_SOLARPANEL_MINUS'
   FRAME_-96002_CLASS          = 3
   FRAME_-96002_CLASS_ID       = -96002
   FRAME_-96002_CENTER         = -96
   CK_-96002_SCLK              = -96
   CK_-96002_SPK               = -96


   \begintext


Antenna Frames
========================================================================

   High-Gain Antenna Nominal Alignment [17]

   The HGA reference axis is nominally the spacecraft -Y axis.
   The reference angle is nominally 90 degree and the reference axis is
   nominally the sapcecraft -X axis.

         [X]      [  0   0  -1 ] [X]
         [Y]    = [  0  -1   0 ] [Y]
         [Z]S/C   [ -1   0   0 ] [Z]HGA

   A more general representation of the transormation for a specified
   rotation angle about the gimbal axis is:

         [X]      [ -cos(s)   0  -sin(s) ] [X]
         [Y]    = [    0     -1     0    ] [Y]
         [Z]S/C   [ -sin(s)   0   cos(s) ] [Z]HGA

   where s varires between 45 degree and 135 degree as the antenna rotates
   about the -Y axis.

   The orientation of the high gain antenna frame with respect to an inertial
   frame, typically J2000, is provided by a C-kernel (see [3] for details).

   \begindata

   FRAME_SPP_HGA_BASE          = -96013
   FRAME_-96013_NAME           = 'SPP_HGA_BASE'
   FRAME_-96013_CLASS          = 4
   FRAME_-96013_CLASS_ID       = -96013
   FRAME_-96013_CENTER         = -96
   TKFRAME_-96013_SPEC         = 'MATRIX'
   TKFRAME_-96013_RELATIVE     = 'SPP_SPACECRAFT'
   TKFRAME_-96013_MATRIX       = ( 1,
                                   0,
                                   0,
                                   0,
                                   1,
                                   0,
                                   0,
                                   0,
                                   1 )

   FRAME_SPP_HIGH_GAIN_ANTENNA = -96003
   FRAME_-96003_NAME           = 'SPP_HIGH_GAIN_ANTENNA'
   FRAME_-96003_CLASS          = 3
   FRAME_-96003_CLASS_ID      = -96003
   FRAME_-96003_CENTER        = -96
   CK_-96003_SCLK             = -96
   CK_-96003_SPK              = -96

   \begintext


Deck Frames
===========================================================================

   The shape of the spacecraft bus is a hexagonal prism [7]. Defined here are
   frames for each of the six panels, or spacecraft decks. Including frames
   for the spacecraft decks simplify the nominal frame definitions for
   some instruments.

   The decks are numbered arbitrarily from 1 to 6, beginning with the deck
   in the first quadrant of the spacecraft XY plane and increasing clockwise
   as viewed looking up the spacecraft +Z axis. The deck numbering is
   illustrated below.

                        X
                         sc
                       ^
                       |
            deck 6  . '|' .  deck 1
                . '    |    ' .
              '        |        '
             ' -.      |      .- '
      deck 5 |    `- . | .- '    | deck 2
    <------------------x------------------->  Y
             |    . -' | `- .    |             sc
              .-'      |      `- .
               '       |        '
                 ' .   |    . `
            deck 4   '.|. '  deck 3
                       |
                       v


   Defined below for each deck is a coordinate system that has its +Z axis
   aligned with the spacecraft +Z axis, and the outward pointing normal to
   the deck as its +X axis.

   The rotation matrices taking vectors from the deck frames to the
   spacecraft frame are defined below.

   \begindata

   FRAME_SPP_DECK_1        = -96081
   FRAME_-96081_NAME       = 'SPP_DECK_1'
   FRAME_-96081_CLASS      = 4
   FRAME_-96081_CLASS_ID   = -96081
   FRAME_-96081_CENTER     = -96
   TKFRAME_-96081_SPEC     = 'MATRIX'
   TKFRAME_-96081_RELATIVE = 'SPP_SPACECRAFT'
   TKFRAME_-96081_MATRIX   = (
                  0.866025403784
                  0.5
                  0.0
                  -0.5
                  0.866025403784
                  0.0
                  0.0
                  0.0
                  1.0 )

   FRAME_SPP_DECK_2        = -96082
   FRAME_-96082_NAME       = 'SPP_DECK_2'
   FRAME_-96082_CLASS      = 4
   FRAME_-96082_CLASS_ID   = -96082
   FRAME_-96082_CENTER     = -96
   TKFRAME_-96082_SPEC     = 'MATRIX'
   TKFRAME_-96082_RELATIVE = 'SPP_SPACECRAFT'
   TKFRAME_-96082_MATRIX   = (
                  0.0
                  1.0
                  0.0
                  -1.0
                  0.0
                  0.0
                  0.0
                  0.0
                  1.0 )

   FRAME_SPP_DECK_3        = -96083
   FRAME_-96083_NAME       = 'SPP_DECK_3'
   FRAME_-96083_CLASS      = 4
   FRAME_-96083_CLASS_ID   = -96083
   FRAME_-96083_CENTER     = -96
   TKFRAME_-96083_SPEC     = 'MATRIX'
   TKFRAME_-96083_RELATIVE = 'SPP_SPACECRAFT'
   TKFRAME_-96083_MATRIX   = (
                  -0.866025403784
                  0.5
                  0.0
                  -0.5
                  -0.866025403784
                  0.0
                  0.0
                  0.0
                  1.0 )

   FRAME_SPP_DECK_4        = -96084
   FRAME_-96084_NAME       = 'SPP_DECK_4'
   FRAME_-96084_CLASS      = 4
   FRAME_-96084_CLASS_ID   = -96084
   FRAME_-96084_CENTER     = -96
   TKFRAME_-96084_SPEC     = 'MATRIX'
   TKFRAME_-96084_RELATIVE = 'SPP_SPACECRAFT'
   TKFRAME_-96084_MATRIX   = (
                  -0.866025403784
                  -0.5
                  0.0
                  0.5
                  -0.866025403784
                  0.0
                  0.0
                  0.0
                  1.0 )

   FRAME_SPP_DECK_5        = -96085
   FRAME_-96085_NAME       = 'SPP_DECK_5'
   FRAME_-96085_CLASS      = 4
   FRAME_-96085_CLASS_ID   = -96085
   FRAME_-96085_CENTER     = -96
   TKFRAME_-96085_SPEC     = 'MATRIX'
   TKFRAME_-96085_RELATIVE = 'SPP_SPACECRAFT'
   TKFRAME_-96085_MATRIX   = (
                  0.0
                  -1.0
                  0.0
                  1.0
                  0.0
                  0.0
                  0.0
                  0.0
                  1.0 )

   FRAME_SPP_DECK_6        = -96086
   FRAME_-96086_NAME       = 'SPP_DECK_6'
   FRAME_-96086_CLASS      = 4
   FRAME_-96086_CLASS_ID   = -96086
   FRAME_-96086_CENTER     = -96
   TKFRAME_-96086_SPEC     = 'MATRIX'
   TKFRAME_-96086_RELATIVE = 'SPP_SPACECRAFT'
   TKFRAME_-96086_MATRIX   = (
                  0.866025403784
                  -0.5
                  0.0
                  0.5
                  0.866025403784
                  0.0
                  0.0
                  0.0
                  1.0 )

   \begintext


WISPR Frames
===========================================================================

   From [18]:

   WISPR consists of two different telescopes that have slightly
   overlapping fields of view.  The current implementation of these
   define these two frames as independently connected to the
   SPP_SPACECRAFT frame, but this can be altered as needed if in
   practice one is calibrated relative to the other.

   The axes of each of the two WISPR frames are as follows:

      Z-axis is the boresight of the telescope (vector pointing out
      of the telescope along its line of sight)

      Y-axis is the normal to the plane containing both camera boresights.

      X-axis completes the right handed system.

   WISPR Inner Telescope Boresight (instrument Z axis):

      The nominal boresight [18] of this detector is 32.2 degrees off the
      SPP_SPACECRAFT +Z-axis towards the spacecraft +X-axis (ram direction
      in encounter orientation). It is inclined 10 degrees out of the
      ecliptic "southward" in the encounter orientation. "Southward" is
      along the Y-axis nominally, so this translates into a boresight
      vector in the spacecraft frame of:

                   [ cos(10.0)*sin(32.2) ] = [ 0.52478068807072864 ]
         Z_Inner = [ sin(10.0)           ] = [ 0.17364817766693033 ]
                   [ cos(10.0)*cos(32.2) ] = [ 0.83333759054837242 ]

      [19] provides an update to this nominal boresight, normalized here:

                   [ 0.53806848033103727 ]
         Z_Inner = [ 0.08578895005278017 ]
                   [ 0.83852404051588747 ]

   WISPR Outer Telescope Boresight (instrument Z axis):

      The nominal boresight [18] of this detector is 77.0 degrees off the
      SPP_SPACECRAFT +Z-axis towards the spacecraft +X-axis (ram direction
      in encounter orientation). It is inclined 10 degrees out of the
      ecliptic "southward" in the encounter orientation. "Southward" is
      along the Y-axis nominally, so this translates into a boresight
      vector in the spacecraft frame of:

                   [ cos(10.0)*sin(77.0) ] = [ 0.95956719410350710 ]
         Z_Outer = [ sin(10.0)           ] = [ 0.17364817766693033 ]
                   [ cos(10.0)*cos(77.0) ] = [ 0.22153354236610870 ]

      [19] provides an update to this nominal boresight, normalized here:

                   [ 0.96022820172582790 ]
         Z_Outer = [ 0.16541537029730274 ]
                   [ 0.22494345040429314 ]

   WISPR Inner and Outer Telescope X and Y axes

      The Y-axis for both cameras is the the normal to the updated Inner and
      Outer Telescope boresights. This is accomplished with the cross product:

                                 [ -0.17192917141267056 ]
         Y = Z_Inner x Z_Outer = [  0.98506308873062820 ]
                                 [  0.00954312516039215 ]

      The X-axis completes the right handed frame:

         X_Inner = Y x Z_Inner
         X_Outer = Y x Z_Outer

   WISPR Inner Telescope Frame (SPP_WISPR_INNER):

      The resultant rotation from the inner telescope frame to the spacecraft
      is then:

        [ X_Inner  Y  Z_Inner ] =

        [  0.82518038663773430 -0.17192917141267056  0.53806848033103727 ]
        [  0.14930159834816223  0.98506308873062820  0.08578895005278017 ]
        [ -0.54478102228242442  0.00954312516039215  0.83852404051588747 ]

      And the corresponding frame definition:

         \begindata
            FRAME_SPP_WISPR_INNER        = -96100
            FRAME_-96100_NAME            = 'SPP_WISPR_INNER'
            FRAME_-96100_CLASS           = 4
            FRAME_-96100_CLASS_ID        = -96100
            FRAME_-96100_CENTER          = 'SPP'
            TKFRAME_-96100_RELATIVE      = 'SPP_SPACECRAFT'
            TKFRAME_-96100_SPEC          = 'MATRIX'
            TKFRAME_-96100_MATRIX        = ( 0.82518038663773430
                                             0.14930159834816223
                                            -0.54478102228242442
                                            -0.17192917141267056
                                             0.98506308873062820
                                             0.00954312516039215
                                             0.53806848033103727
                                             0.08578895005278017
                                             0.83852404051588747 )
         \begintext

   WISPR Outer Telescope Frame (SPP_WISPR_OUTER):

      The resultant rotation from the outer telescope frame to the spacecraft
      is then:

        [ X_Outer  Y  Z_Outer ] =

        [  0.22000491046277809 -0.17192917141267056  0.96022820172582790 ]
        [  0.04783791895432513  0.98506308873062820  0.16541537029730274 ]
        [ -0.97432508583243616  0.00954312516039215  0.22494345040429314 ]

      And the corresponding frame definition:

         \begindata
            FRAME_SPP_WISPR_OUTER        = -96120
            FRAME_-96120_NAME            = 'SPP_WISPR_OUTER'
            FRAME_-96120_CLASS           = 4
            FRAME_-96120_CLASS_ID        = -96120
            FRAME_-96120_CENTER          = 'SPP'
            TKFRAME_-96120_RELATIVE      = 'SPP_SPACECRAFT'
            TKFRAME_-96120_SPEC          = 'MATRIX'
            TKFRAME_-96120_MATRIX        = ( 0.22000491046277809
                                             0.04783791895432513
                                            -0.97432508583243616
                                            -0.17192917141267056
                                             0.98506308873062820
                                             0.00954312516039215
                                             0.96022820172582790
                                             0.16541537029730274
                                             0.22494345040429314 )
         \begintext


SWEAP Frames
===========================================================================

TBD


FIELDS Frames
===========================================================================

TBD


EPI-Lo Frames
===========================================================================

   [20] describes in detail the EPI-Lo frame determination described in the
   following sections.

   EPI-Lo Frame Naming and Numbering Convention
   ------------------------------------------------------------------------

   EPI-Lo has 8 octants, or wedges, each having 10 apertures, for a total of
   80 apertures. Each wedge and each aperture has its own frame defined below.

   Frame Naming:

   [6] shows the wedge and aperture locations in the instrument. The eight
   wedges are named W0 - W7. The ten apertures in a wedge are named
   L#0 - L#9, where # is the wedge number, to give eighty unique names
   L00 - L79 for the apertures.

   Frame Numbering:

   Each wedge frame is assigned the NAIF ID -964##, where ## is
   the corresponding double digit number listed below.

   Wedge 0 has number 11
   Wedge 1 has number 22
   Wedge 2 has number 33
   Wedge 3 has number 44
   Wedge 4 has number 55
   Wedge 5 has number 66
   Wedge 6 has number 77
   Wedge 7 has number 88

   The aperture frames are also assigned NAIF IDs -964##, where ## is
   incremented from the NAIF ID of the containing wedge. Thus,

   Wedge 0 apertures L00-L09 have numbers 12-21
   Wedge 1 apertures L10-L19 have numbers 23-32
   Wedge 2 apertures L20-L29 have numbers 34-43
   Wedge 3 apertures L30-L39 have numbers 45-54
   Wedge 4 apertures L40-L49 have numbers 56-65
   Wedge 5 apertures L50-L59 have numbers 67-76
   Wedge 6 apertures L60-L69 have numbers 78-87
   Wedge 7 apertures L70-L79 have numbers 89-98


   EPI-Lo Mounting Alignment Frame
   ------------------------------------------------------------------------

   From page 17 of [7] we see that EPI-Lo is mounted onto spacecraft deck 1
   by a bracket. The deck 1 orientation is described in the Spacecraft Frames
   section of this document, and the bracket orientation is described on sheet
   2 of [8]. The rotation taking vectors from the instrument base frame to the
   deck 1 frame is captured below.

   The instrument base coordinate system is defined in [4]. There, the
   instrument +Z axis points outward from the center of symmetry of the
   instrument and is normal to the instrument base, or mounting face. By
   comparing the location of the connectors in that drawing with those on
   page 2 of [6], the instrument +X axis lies in the plane of the instrument
   base and points outward from the center of the instrument between wedges
   W6 and W7. The instrument +Y axis lies in the plane of the instrument
   base and points outward from the center of the instrument between wedges
   W4 and W5. The following diagram illustrates the base frame.

                       Y
                        base
                                            View looking down on the
                       ^                    instrument (apertures point
                       |                    out of the plane of the
                ,   W4 | W5   .             page toward the viewer)
                  ,    |    .
                    ,  |  .
                W3    ,|.    W6
            -----------o----------->   X
                W2    '|`    W7         base
                    '  |  `
                  '    |    `
                '   W1 | W0   `
                       |

   From the mounting bracket drawing in [8], the instrument mounting face is
   a 13.35 degree rotation about the +Y axis of deck 1, and a -9.70 degree
   rotation about the +Z axis of deck 1. We determine the normal to the mounting
   face by taking the cross product of two orthogonal vectors lying in the
   plane of the mounting face. This normal vector is the instrument base
   frame's +Z axis. The diagram further shows a twist about the base frame's
   Z axis. This nominal twist was adjusted to align the L12, L30, L31, L51,
   and L52 apertures to their pre-environmental measured positions given in
   [10]. The

   The rotation described above taking vectors from the SPP_EPILO_BASE frame
   to the SPP_DECK_1 frame is given below.

   \begindata

   FRAME_SPP_EPILO_BASE    = -96401
   FRAME_-96401_NAME       = 'SPP_EPILO_BASE'
   FRAME_-96401_CLASS      = 4
   FRAME_-96401_CLASS_ID   = -96401
   FRAME_-96401_CENTER     = -96
   TKFRAME_-96401_SPEC     = 'MATRIX'
   TKFRAME_-96401_RELATIVE = 'SPP_DECK_1'
   TKFRAME_-96401_MATRIX   = (
                -0.12716527279986342
                 0.4692559064526792
                 -0.873858047770352
                 0.2502482446581486
                 0.8676883125456071
                 0.42952626033488644
                 0.9597941495129318
                 -0.16406061846473613
                 -0.22777028785536657)

   \begintext


   EPI-Lo Wedge Frames
   ------------------------------------------------------------------------

   Each wedge frame is defined such that +Z axis is in the mounting plane of
   the instrument and points outward from the center of the instrument along
   the azimuthal center line of the wedge. The wedge +Y axis is normal to the
   mounting plane and points outward from the center of the instrument to the
   side containing the instrument apertures (i.e. each wedge has its +Y axis
   aligned with SPP_EPILO_BASE's +Z axis). The wedge frame is illustrated
   below.

   Looking crosswise through the wedge:

                    Y
                     wedge

                   ^
                   |
                   |
              \`` '|- - .
               \   |      ` -
                \  |           ` -
                 \ |               `.
                  \|_________________\----> Z
                                             wedge

   Looking down on the wedge (apertures point out of the plane
   of the page toward the viewer):

              X
                wedge

              ^
              |
              |                    .'|
              |                .'    |
              |            .'        |
              |        .'`. 45/2 deg |
              |    .'     `          |
              |.'_________'__________|_____\  Z
               `.                    |     /    wedge
                   `.                |
                       `.            |
                           `.        |
                               `.    |
                                   `.|

   As there are 8 wedges, each wedge's +Z axis is 45 degrees apart from its
   neighbor's. The matrices taking vectors from the wedge frames to the
   SPP_EPILO_BASE frame are described below.

   \begindata

   FRAME_SPP_EPILO_W0     = -96411
   FRAME_-96411_NAME       = 'SPP_EPILO_W0'
   FRAME_-96411_CLASS      = 4
   FRAME_-96411_CLASS_ID   = -96411
   FRAME_-96411_CENTER     = -96
   TKFRAME_-96411_SPEC     = 'MATRIX'
   TKFRAME_-96411_RELATIVE = 'SPP_EPILO_BASE'
   TKFRAME_-96411_MATRIX   = (0.9238795325112866
                              0.38268343236509
                              0
                              0
                              0
                              1
                              0.38268343236509
                              -0.9238795325112866
                              0)


   FRAME_SPP_EPILO_W1     = -96422
   FRAME_-96422_NAME       = 'SPP_EPILO_W1'
   FRAME_-96422_CLASS      = 4
   FRAME_-96422_CLASS_ID   = -96422
   FRAME_-96422_CENTER     = -96
   TKFRAME_-96422_SPEC     = 'MATRIX'
   TKFRAME_-96422_RELATIVE = 'SPP_EPILO_BASE'
   TKFRAME_-96422_MATRIX   = (0.9238795325112865
                              -0.38268343236509034
                              0
                              0
                              0
                              1
                              -0.38268343236509034
                              -0.9238795325112865
                              0)


   FRAME_SPP_EPILO_W2     = -96433
   FRAME_-96433_NAME       = 'SPP_EPILO_W2'
   FRAME_-96433_CLASS      = 4
   FRAME_-96433_CLASS_ID   = -96433
   FRAME_-96433_CENTER     = -96
   TKFRAME_-96433_SPEC     = 'MATRIX'
   TKFRAME_-96433_RELATIVE = 'SPP_EPILO_BASE'
   TKFRAME_-96433_MATRIX   = (0.38268343236508967
                              -0.9238795325112868
                              0
                              0
                              0
                              1
                              -0.9238795325112868
                              -0.38268343236508967
                              0)


   FRAME_SPP_EPILO_W3     = -96444
   FRAME_-96444_NAME       = 'SPP_EPILO_W3'
   FRAME_-96444_CLASS      = 4
   FRAME_-96444_CLASS_ID   = -96444
   FRAME_-96444_CENTER     = -96
   TKFRAME_-96444_SPEC     = 'MATRIX'
   TKFRAME_-96444_RELATIVE = 'SPP_EPILO_BASE'
   TKFRAME_-96444_MATRIX   = (-0.3826834323650899
                              -0.9238795325112867
                              0
                              0
                              0
                              1
                              -0.9238795325112867
                              0.3826834323650899
                              0)


   FRAME_SPP_EPILO_W4     = -96455
   FRAME_-96455_NAME       = 'SPP_EPILO_W4'
   FRAME_-96455_CLASS      = 4
   FRAME_-96455_CLASS_ID   = -96455
   FRAME_-96455_CENTER     = -96
   TKFRAME_-96455_SPEC     = 'MATRIX'
   TKFRAME_-96455_RELATIVE = 'SPP_EPILO_BASE'
   TKFRAME_-96455_MATRIX   = (-0.9238795325112867
                              -0.3826834323650897
                              0
                              0
                              0
                              1
                              -0.3826834323650897
                              0.9238795325112867
                              0)


   FRAME_SPP_EPILO_W5     = -96466
   FRAME_-96466_NAME       = 'SPP_EPILO_W5'
   FRAME_-96466_CLASS      = 4
   FRAME_-96466_CLASS_ID   = -96466
   FRAME_-96466_CENTER     = -96
   TKFRAME_-96466_SPEC     = 'MATRIX'
   TKFRAME_-96466_RELATIVE = 'SPP_EPILO_BASE'
   TKFRAME_-96466_MATRIX   = (-0.9238795325112867
                              0.38268343236508984
                              0
                              0
                              0
                              1
                              0.38268343236508984
                              0.9238795325112867
                              0)


   FRAME_SPP_EPILO_W6     = -96477
   FRAME_-96477_NAME       = 'SPP_EPILO_W6'
   FRAME_-96477_CLASS      = 4
   FRAME_-96477_CLASS_ID   = -96477
   FRAME_-96477_CENTER     = -96
   TKFRAME_-96477_SPEC     = 'MATRIX'
   TKFRAME_-96477_RELATIVE = 'SPP_EPILO_BASE'
   TKFRAME_-96477_MATRIX   = (-0.3826834323650898
                              0.9238795325112867
                              0
                              0
                              0
                              1
                              0.9238795325112867
                              0.3826834323650898
                              0)


   FRAME_SPP_EPILO_W7     = -96488
   FRAME_-96488_NAME       = 'SPP_EPILO_W7'
   FRAME_-96488_CLASS      = 4
   FRAME_-96488_CLASS_ID   = -96488
   FRAME_-96488_CENTER     = -96
   TKFRAME_-96488_SPEC     = 'MATRIX'
   TKFRAME_-96488_RELATIVE = 'SPP_EPILO_BASE'
   TKFRAME_-96488_MATRIX   = (0.3826834323650898
                              0.9238795325112867
                              0
                              0
                              0
                              1
                              0.9238795325112867
                              -0.3826834323650898
                              0)


   \begintext

   EPI-Lo Aperture Frames
   ------------------------------------------------------------------------

   [5] describes the aperture orientations and angles. Each aperture frame is
   defined such that its +Z axis points out of the aperture from the Ion pixel
   center. The aperture +Y axis is the projection of the wedge +Y axis onto the
   plane perpendicular to the aperture +Z axis. The aperture +X axis completes
   the right-handed frame. The aperture +X axis is aligned with the wedge +X
   axis for the apertures that lie along the wedge line of symmetry, namely
   L#8 and L#9.

   The following cross-section illustrates the aperture frame. Note that the
   aperture Y axis has a component into or out of the page for apertures L#0
   through L#7.


                    Y
                     wedge
                   ^
                   |
                   |       Y              __ Z
                   |    __  aperture     .'/  aperture
                   |   /.             .'
                   |     `.        .'
              \`` '|- - .  `.   .'
               \   |      ` -.'
                \  |      .'    ` -
                 \ |   .'           `.
                  \|.'                \----> Z
                   *-------------------       wedge

              Ion pixel
               center


   The following view looking down on a wedge roughly illustrates the location
   of each aperture within the wedge. The circles represent the apertures;
   positions are not to scale.


              X
                wedge

              ^
              |
              |                    .'|
              |                .' L#0|
              |            .'      O |
              |        .'L#6  L#4    |
              |    .'     O    O  L#1|
              |.L#9 L#8            O |
               . O   O            L#2|-----> Z
                `.       L#7  L#5  O |         wedge
                   `.     O    O     |
                       `.         L#3|
                           `.      O |
                               `.    |
                                   `.|


   Wedge 0 aperture frames:

   \begindata

   FRAME_SPP_EPILO_L00     = -96412
   FRAME_-96412_NAME       = 'SPP_EPILO_L00'
   FRAME_-96412_CLASS      = 4
   FRAME_-96412_CLASS_ID   = -96412
   FRAME_-96412_CENTER     = -96
   TKFRAME_-96412_SPEC     = 'MATRIX'
   TKFRAME_-96412_RELATIVE = 'SPP_EPILO_W0'
   TKFRAME_-96412_MATRIX   = (0.9569185564499445
                              -0.0027262750100587,
                              -0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L01     = -96413
   FRAME_-96413_NAME       = 'SPP_EPILO_L01'
   FRAME_-96413_CLASS      = 4
   FRAME_-96413_CLASS_ID   = -96413
   FRAME_-96413_CENTER     = -96
   TKFRAME_-96413_SPEC     = 'MATRIX'
   TKFRAME_-96413_RELATIVE = 'SPP_EPILO_W0'
   TKFRAME_-96413_MATRIX   = (0.9951765914819487
                              -0.0009211052643909,
                              -0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L02     = -96414
   FRAME_-96414_NAME       = 'SPP_EPILO_L02'
   FRAME_-96414_CLASS      = 4
   FRAME_-96414_CLASS_ID   = -96414
   FRAME_-96414_CENTER     = -96
   TKFRAME_-96414_SPEC     = 'MATRIX'
   TKFRAME_-96414_RELATIVE = 'SPP_EPILO_W0'
   TKFRAME_-96414_MATRIX   = (0.9951765914819487
                              0.0009211052643909,
                              0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              -0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L03     = -96415
   FRAME_-96415_NAME       = 'SPP_EPILO_L03'
   FRAME_-96415_CLASS      = 4
   FRAME_-96415_CLASS_ID   = -96415
   FRAME_-96415_CENTER     = -96
   TKFRAME_-96415_SPEC     = 'MATRIX'
   TKFRAME_-96415_RELATIVE = 'SPP_EPILO_W0'
   TKFRAME_-96415_MATRIX   = (0.9569185564499445
                              0.0027262750100587,
                              0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              -0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L04     = -96416
   FRAME_-96416_NAME       = 'SPP_EPILO_L04'
   FRAME_-96416_CLASS      = 4
   FRAME_-96416_CLASS_ID   = -96416
   FRAME_-96416_CENTER     = -96
   TKFRAME_-96416_SPEC     = 'MATRIX'
   TKFRAME_-96416_RELATIVE = 'SPP_EPILO_W0'
   TKFRAME_-96416_MATRIX   = (0.9836441852061112
                              -0.0703416078843675,
                              -0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L05     = -96417
   FRAME_-96417_NAME       = 'SPP_EPILO_L05'
   FRAME_-96417_CLASS      = 4
   FRAME_-96417_CLASS_ID   = -96417
   FRAME_-96417_CENTER     = -96
   TKFRAME_-96417_SPEC     = 'MATRIX'
   TKFRAME_-96417_RELATIVE = 'SPP_EPILO_W0'
   TKFRAME_-96417_MATRIX   = (0.9836441852061112
                              0.0703416078843675,
                              0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              -0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L06     = -96418
   FRAME_-96418_NAME       = 'SPP_EPILO_L06'
   FRAME_-96418_CLASS      = 4
   FRAME_-96418_CLASS_ID   = -96418
   FRAME_-96418_CENTER     = -96
   TKFRAME_-96418_SPEC     = 'MATRIX'
   TKFRAME_-96418_RELATIVE = 'SPP_EPILO_W0'
   TKFRAME_-96418_MATRIX   = (0.9903244000533897
                              -0.0984911534882631,
                              -0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L07     = -96419
   FRAME_-96419_NAME       = 'SPP_EPILO_L07'
   FRAME_-96419_CLASS      = 4
   FRAME_-96419_CLASS_ID   = -96419
   FRAME_-96419_CENTER     = -96
   TKFRAME_-96419_SPEC     = 'MATRIX'
   TKFRAME_-96419_RELATIVE = 'SPP_EPILO_W0'
   TKFRAME_-96419_MATRIX   = (0.9903244000533897
                              0.0984911534882631,
                              0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              -0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L08     = -96420
   FRAME_-96420_NAME       = 'SPP_EPILO_L08'
   FRAME_-96420_CLASS      = 4
   FRAME_-96420_CLASS_ID   = -96420
   FRAME_-96420_CENTER     = -96
   TKFRAME_-96420_SPEC     = 'MATRIX'
   TKFRAME_-96420_RELATIVE = 'SPP_EPILO_W0'
   TKFRAME_-96420_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.3902632340374340,
                              -0.9207033225521906,
                              0.0000000000000000,
                              0.9207033225521906,
                              0.3902632340374340)

   FRAME_SPP_EPILO_L09     = -96421
   FRAME_-96421_NAME       = 'SPP_EPILO_L09'
   FRAME_-96421_CLASS      = 4
   FRAME_-96421_CLASS_ID   = -96421
   FRAME_-96421_CENTER     = -96
   TKFRAME_-96421_SPEC     = 'MATRIX'
   TKFRAME_-96421_RELATIVE = 'SPP_EPILO_W0'
   TKFRAME_-96421_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000001,
                              -1.0000000000000000,
                              0.0000000000000000,
                              1.0000000000000000,
                              0.0000000000000001)

   \begintext


   Wedge 1 aperture frames:

   \begindata

   FRAME_SPP_EPILO_L10     = -96423
   FRAME_-96423_NAME       = 'SPP_EPILO_L10'
   FRAME_-96423_CLASS      = 4
   FRAME_-96423_CLASS_ID   = -96423
   FRAME_-96423_CENTER     = -96
   TKFRAME_-96423_SPEC     = 'MATRIX'
   TKFRAME_-96423_RELATIVE = 'SPP_EPILO_W1'
   TKFRAME_-96423_MATRIX   = (0.9569185564499445
                              -0.0027262750100587,
                              -0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L11     = -96424
   FRAME_-96424_NAME       = 'SPP_EPILO_L11'
   FRAME_-96424_CLASS      = 4
   FRAME_-96424_CLASS_ID   = -96424
   FRAME_-96424_CENTER     = -96
   TKFRAME_-96424_SPEC     = 'MATRIX'
   TKFRAME_-96424_RELATIVE = 'SPP_EPILO_W1'
   TKFRAME_-96424_MATRIX   = (0.9951765914819487
                              -0.0009211052643909,
                              -0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L12     = -96425
   FRAME_-96425_NAME       = 'SPP_EPILO_L12'
   FRAME_-96425_CLASS      = 4
   FRAME_-96425_CLASS_ID   = -96425
   FRAME_-96425_CENTER     = -96
   TKFRAME_-96425_SPEC     = 'MATRIX'
   TKFRAME_-96425_RELATIVE = 'SPP_EPILO_W1'
   TKFRAME_-96425_MATRIX   = (0.9951765914819487
                              0.0009211052643909,
                              0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              -0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L13     = -96426
   FRAME_-96426_NAME       = 'SPP_EPILO_L13'
   FRAME_-96426_CLASS      = 4
   FRAME_-96426_CLASS_ID   = -96426
   FRAME_-96426_CENTER     = -96
   TKFRAME_-96426_SPEC     = 'MATRIX'
   TKFRAME_-96426_RELATIVE = 'SPP_EPILO_W1'
   TKFRAME_-96426_MATRIX   = (0.9569185564499445
                              0.0027262750100587,
                              0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              -0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L14     = -96427
   FRAME_-96427_NAME       = 'SPP_EPILO_L14'
   FRAME_-96427_CLASS      = 4
   FRAME_-96427_CLASS_ID   = -96427
   FRAME_-96427_CENTER     = -96
   TKFRAME_-96427_SPEC     = 'MATRIX'
   TKFRAME_-96427_RELATIVE = 'SPP_EPILO_W1'
   TKFRAME_-96427_MATRIX   = (0.9836441852061112
                              -0.0703416078843675,
                              -0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L15     = -96428
   FRAME_-96428_NAME       = 'SPP_EPILO_L15'
   FRAME_-96428_CLASS      = 4
   FRAME_-96428_CLASS_ID   = -96428
   FRAME_-96428_CENTER     = -96
   TKFRAME_-96428_SPEC     = 'MATRIX'
   TKFRAME_-96428_RELATIVE = 'SPP_EPILO_W1'
   TKFRAME_-96428_MATRIX   = (0.9836441852061112
                              0.0703416078843675,
                              0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              -0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L16     = -96429
   FRAME_-96429_NAME       = 'SPP_EPILO_L16'
   FRAME_-96429_CLASS      = 4
   FRAME_-96429_CLASS_ID   = -96429
   FRAME_-96429_CENTER     = -96
   TKFRAME_-96429_SPEC     = 'MATRIX'
   TKFRAME_-96429_RELATIVE = 'SPP_EPILO_W1'
   TKFRAME_-96429_MATRIX   = (0.9903244000533897
                              -0.0984911534882631,
                              -0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L17     = -96430
   FRAME_-96430_NAME       = 'SPP_EPILO_L17'
   FRAME_-96430_CLASS      = 4
   FRAME_-96430_CLASS_ID   = -96430
   FRAME_-96430_CENTER     = -96
   TKFRAME_-96430_SPEC     = 'MATRIX'
   TKFRAME_-96430_RELATIVE = 'SPP_EPILO_W1'
   TKFRAME_-96430_MATRIX   = (0.9903244000533897
                              0.0984911534882631,
                              0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              -0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L18     = -96431
   FRAME_-96431_NAME       = 'SPP_EPILO_L18'
   FRAME_-96431_CLASS      = 4
   FRAME_-96431_CLASS_ID   = -96431
   FRAME_-96431_CENTER     = -96
   TKFRAME_-96431_SPEC     = 'MATRIX'
   TKFRAME_-96431_RELATIVE = 'SPP_EPILO_W1'
   TKFRAME_-96431_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.3902632340374340,
                              -0.9207033225521906,
                              0.0000000000000000,
                              0.9207033225521906,
                              0.3902632340374340)

   FRAME_SPP_EPILO_L19     = -96432
   FRAME_-96432_NAME       = 'SPP_EPILO_L19'
   FRAME_-96432_CLASS      = 4
   FRAME_-96432_CLASS_ID   = -96432
   FRAME_-96432_CENTER     = -96
   TKFRAME_-96432_SPEC     = 'MATRIX'
   TKFRAME_-96432_RELATIVE = 'SPP_EPILO_W1'
   TKFRAME_-96432_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000001,
                              -1.0000000000000000,
                              0.0000000000000000,
                              1.0000000000000000,
                              0.0000000000000001)

   \begintext


   Wedge 2 aperture frames:

   \begindata

   FRAME_SPP_EPILO_L20     = -96434
   FRAME_-96434_NAME       = 'SPP_EPILO_L20'
   FRAME_-96434_CLASS      = 4
   FRAME_-96434_CLASS_ID   = -96434
   FRAME_-96434_CENTER     = -96
   TKFRAME_-96434_SPEC     = 'MATRIX'
   TKFRAME_-96434_RELATIVE = 'SPP_EPILO_W2'
   TKFRAME_-96434_MATRIX   = (0.9569185564499445
                              -0.0027262750100587,
                              -0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L21     = -96435
   FRAME_-96435_NAME       = 'SPP_EPILO_L21'
   FRAME_-96435_CLASS      = 4
   FRAME_-96435_CLASS_ID   = -96435
   FRAME_-96435_CENTER     = -96
   TKFRAME_-96435_SPEC     = 'MATRIX'
   TKFRAME_-96435_RELATIVE = 'SPP_EPILO_W2'
   TKFRAME_-96435_MATRIX   = (0.9951765914819487
                              -0.0009211052643909,
                              -0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L22     = -96436
   FRAME_-96436_NAME       = 'SPP_EPILO_L22'
   FRAME_-96436_CLASS      = 4
   FRAME_-96436_CLASS_ID   = -96436
   FRAME_-96436_CENTER     = -96
   TKFRAME_-96436_SPEC     = 'MATRIX'
   TKFRAME_-96436_RELATIVE = 'SPP_EPILO_W2'
   TKFRAME_-96436_MATRIX   = (0.9951765914819487
                              0.0009211052643909,
                              0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              -0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L23     = -96437
   FRAME_-96437_NAME       = 'SPP_EPILO_L23'
   FRAME_-96437_CLASS      = 4
   FRAME_-96437_CLASS_ID   = -96437
   FRAME_-96437_CENTER     = -96
   TKFRAME_-96437_SPEC     = 'MATRIX'
   TKFRAME_-96437_RELATIVE = 'SPP_EPILO_W2'
   TKFRAME_-96437_MATRIX   = (0.9569185564499445
                              0.0027262750100587,
                              0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              -0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L24     = -96438
   FRAME_-96438_NAME       = 'SPP_EPILO_L24'
   FRAME_-96438_CLASS      = 4
   FRAME_-96438_CLASS_ID   = -96438
   FRAME_-96438_CENTER     = -96
   TKFRAME_-96438_SPEC     = 'MATRIX'
   TKFRAME_-96438_RELATIVE = 'SPP_EPILO_W2'
   TKFRAME_-96438_MATRIX   = (0.9836441852061112
                              -0.0703416078843675,
                              -0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L25     = -96439
   FRAME_-96439_NAME       = 'SPP_EPILO_L25'
   FRAME_-96439_CLASS      = 4
   FRAME_-96439_CLASS_ID   = -96439
   FRAME_-96439_CENTER     = -96
   TKFRAME_-96439_SPEC     = 'MATRIX'
   TKFRAME_-96439_RELATIVE = 'SPP_EPILO_W2'
   TKFRAME_-96439_MATRIX   = (0.9836441852061112
                              0.0703416078843675,
                              0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              -0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L26     = -96440
   FRAME_-96440_NAME       = 'SPP_EPILO_L26'
   FRAME_-96440_CLASS      = 4
   FRAME_-96440_CLASS_ID   = -96440
   FRAME_-96440_CENTER     = -96
   TKFRAME_-96440_SPEC     = 'MATRIX'
   TKFRAME_-96440_RELATIVE = 'SPP_EPILO_W2'
   TKFRAME_-96440_MATRIX   = (0.9903244000533897
                              -0.0984911534882631,
                              -0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L27     = -96441
   FRAME_-96441_NAME       = 'SPP_EPILO_L27'
   FRAME_-96441_CLASS      = 4
   FRAME_-96441_CLASS_ID   = -96441
   FRAME_-96441_CENTER     = -96
   TKFRAME_-96441_SPEC     = 'MATRIX'
   TKFRAME_-96441_RELATIVE = 'SPP_EPILO_W2'
   TKFRAME_-96441_MATRIX   = (0.9903244000533897
                              0.0984911534882631,
                              0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              -0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L28     = -96442
   FRAME_-96442_NAME       = 'SPP_EPILO_L28'
   FRAME_-96442_CLASS      = 4
   FRAME_-96442_CLASS_ID   = -96442
   FRAME_-96442_CENTER     = -96
   TKFRAME_-96442_SPEC     = 'MATRIX'
   TKFRAME_-96442_RELATIVE = 'SPP_EPILO_W2'
   TKFRAME_-96442_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.3902632340374340,
                              -0.9207033225521906,
                              0.0000000000000000,
                              0.9207033225521906,
                              0.3902632340374340)

   FRAME_SPP_EPILO_L29     = -96443
   FRAME_-96443_NAME       = 'SPP_EPILO_L29'
   FRAME_-96443_CLASS      = 4
   FRAME_-96443_CLASS_ID   = -96443
   FRAME_-96443_CENTER     = -96
   TKFRAME_-96443_SPEC     = 'MATRIX'
   TKFRAME_-96443_RELATIVE = 'SPP_EPILO_W2'
   TKFRAME_-96443_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000001,
                              -1.0000000000000000,
                              0.0000000000000000,
                              1.0000000000000000,
                              0.0000000000000001)

   \begintext


   Wedge 3 aperture frames:

   \begindata

   FRAME_SPP_EPILO_L30     = -96445
   FRAME_-96445_NAME       = 'SPP_EPILO_L30'
   FRAME_-96445_CLASS      = 4
   FRAME_-96445_CLASS_ID   = -96445
   FRAME_-96445_CENTER     = -96
   TKFRAME_-96445_SPEC     = 'MATRIX'
   TKFRAME_-96445_RELATIVE = 'SPP_EPILO_W3'
   TKFRAME_-96445_MATRIX   = (0.9569185564499445
                              -0.0027262750100587,
                              -0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L31     = -96446
   FRAME_-96446_NAME       = 'SPP_EPILO_L31'
   FRAME_-96446_CLASS      = 4
   FRAME_-96446_CLASS_ID   = -96446
   FRAME_-96446_CENTER     = -96
   TKFRAME_-96446_SPEC     = 'MATRIX'
   TKFRAME_-96446_RELATIVE = 'SPP_EPILO_W3'
   TKFRAME_-96446_MATRIX   = (0.9951765914819487
                              -0.0009211052643909,
                              -0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L32     = -96447
   FRAME_-96447_NAME       = 'SPP_EPILO_L32'
   FRAME_-96447_CLASS      = 4
   FRAME_-96447_CLASS_ID   = -96447
   FRAME_-96447_CENTER     = -96
   TKFRAME_-96447_SPEC     = 'MATRIX'
   TKFRAME_-96447_RELATIVE = 'SPP_EPILO_W3'
   TKFRAME_-96447_MATRIX   = (0.9951765914819487
                              0.0009211052643909,
                              0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              -0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L33     = -96448
   FRAME_-96448_NAME       = 'SPP_EPILO_L33'
   FRAME_-96448_CLASS      = 4
   FRAME_-96448_CLASS_ID   = -96448
   FRAME_-96448_CENTER     = -96
   TKFRAME_-96448_SPEC     = 'MATRIX'
   TKFRAME_-96448_RELATIVE = 'SPP_EPILO_W3'
   TKFRAME_-96448_MATRIX   = (0.9569185564499445
                              0.0027262750100587,
                              0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              -0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L34     = -96449
   FRAME_-96449_NAME       = 'SPP_EPILO_L34'
   FRAME_-96449_CLASS      = 4
   FRAME_-96449_CLASS_ID   = -96449
   FRAME_-96449_CENTER     = -96
   TKFRAME_-96449_SPEC     = 'MATRIX'
   TKFRAME_-96449_RELATIVE = 'SPP_EPILO_W3'
   TKFRAME_-96449_MATRIX   = (0.9836441852061112
                              -0.0703416078843675,
                              -0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L35     = -96450
   FRAME_-96450_NAME       = 'SPP_EPILO_L35'
   FRAME_-96450_CLASS      = 4
   FRAME_-96450_CLASS_ID   = -96450
   FRAME_-96450_CENTER     = -96
   TKFRAME_-96450_SPEC     = 'MATRIX'
   TKFRAME_-96450_RELATIVE = 'SPP_EPILO_W3'
   TKFRAME_-96450_MATRIX   = (0.9836441852061112
                              0.0703416078843675,
                              0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              -0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L36     = -96451
   FRAME_-96451_NAME       = 'SPP_EPILO_L36'
   FRAME_-96451_CLASS      = 4
   FRAME_-96451_CLASS_ID   = -96451
   FRAME_-96451_CENTER     = -96
   TKFRAME_-96451_SPEC     = 'MATRIX'
   TKFRAME_-96451_RELATIVE = 'SPP_EPILO_W3'
   TKFRAME_-96451_MATRIX   = (0.9903244000533897
                              -0.0984911534882631,
                              -0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L37     = -96452
   FRAME_-96452_NAME       = 'SPP_EPILO_L37'
   FRAME_-96452_CLASS      = 4
   FRAME_-96452_CLASS_ID   = -96452
   FRAME_-96452_CENTER     = -96
   TKFRAME_-96452_SPEC     = 'MATRIX'
   TKFRAME_-96452_RELATIVE = 'SPP_EPILO_W3'
   TKFRAME_-96452_MATRIX   = (0.9903244000533897
                              0.0984911534882631,
                              0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              -0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L38     = -96453
   FRAME_-96453_NAME       = 'SPP_EPILO_L38'
   FRAME_-96453_CLASS      = 4
   FRAME_-96453_CLASS_ID   = -96453
   FRAME_-96453_CENTER     = -96
   TKFRAME_-96453_SPEC     = 'MATRIX'
   TKFRAME_-96453_RELATIVE = 'SPP_EPILO_W3'
   TKFRAME_-96453_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.3902632340374340,
                              -0.9207033225521906,
                              0.0000000000000000,
                              0.9207033225521906,
                              0.3902632340374340)

   FRAME_SPP_EPILO_L39     = -96454
   FRAME_-96454_NAME       = 'SPP_EPILO_L39'
   FRAME_-96454_CLASS      = 4
   FRAME_-96454_CLASS_ID   = -96454
   FRAME_-96454_CENTER     = -96
   TKFRAME_-96454_SPEC     = 'MATRIX'
   TKFRAME_-96454_RELATIVE = 'SPP_EPILO_W3'
   TKFRAME_-96454_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000001,
                              -1.0000000000000000,
                              0.0000000000000000,
                              1.0000000000000000,
                              0.0000000000000001)

   \begintext


   Wedge 4 aperture frames:

   \begindata

   FRAME_SPP_EPILO_L40     = -96456
   FRAME_-96456_NAME       = 'SPP_EPILO_L40'
   FRAME_-96456_CLASS      = 4
   FRAME_-96456_CLASS_ID   = -96456
   FRAME_-96456_CENTER     = -96
   TKFRAME_-96456_SPEC     = 'MATRIX'
   TKFRAME_-96456_RELATIVE = 'SPP_EPILO_W4'
   TKFRAME_-96456_MATRIX   = (0.9569185564499445
                              -0.0027262750100587,
                              -0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L41     = -96457
   FRAME_-96457_NAME       = 'SPP_EPILO_L41'
   FRAME_-96457_CLASS      = 4
   FRAME_-96457_CLASS_ID   = -96457
   FRAME_-96457_CENTER     = -96
   TKFRAME_-96457_SPEC     = 'MATRIX'
   TKFRAME_-96457_RELATIVE = 'SPP_EPILO_W4'
   TKFRAME_-96457_MATRIX   = (0.9951765914819487
                              -0.0009211052643909,
                              -0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L42     = -96458
   FRAME_-96458_NAME       = 'SPP_EPILO_L42'
   FRAME_-96458_CLASS      = 4
   FRAME_-96458_CLASS_ID   = -96458
   FRAME_-96458_CENTER     = -96
   TKFRAME_-96458_SPEC     = 'MATRIX'
   TKFRAME_-96458_RELATIVE = 'SPP_EPILO_W4'
   TKFRAME_-96458_MATRIX   = (0.9951765914819487
                              0.0009211052643909,
                              0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              -0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L43     = -96459
   FRAME_-96459_NAME       = 'SPP_EPILO_L43'
   FRAME_-96459_CLASS      = 4
   FRAME_-96459_CLASS_ID   = -96459
   FRAME_-96459_CENTER     = -96
   TKFRAME_-96459_SPEC     = 'MATRIX'
   TKFRAME_-96459_RELATIVE = 'SPP_EPILO_W4'
   TKFRAME_-96459_MATRIX   = (0.9569185564499445
                              0.0027262750100587,
                              0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              -0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L44     = -96460
   FRAME_-96460_NAME       = 'SPP_EPILO_L44'
   FRAME_-96460_CLASS      = 4
   FRAME_-96460_CLASS_ID   = -96460
   FRAME_-96460_CENTER     = -96
   TKFRAME_-96460_SPEC     = 'MATRIX'
   TKFRAME_-96460_RELATIVE = 'SPP_EPILO_W4'
   TKFRAME_-96460_MATRIX   = (0.9836441852061112
                              -0.0703416078843675,
                              -0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L45     = -96461
   FRAME_-96461_NAME       = 'SPP_EPILO_L45'
   FRAME_-96461_CLASS      = 4
   FRAME_-96461_CLASS_ID   = -96461
   FRAME_-96461_CENTER     = -96
   TKFRAME_-96461_SPEC     = 'MATRIX'
   TKFRAME_-96461_RELATIVE = 'SPP_EPILO_W4'
   TKFRAME_-96461_MATRIX   = (0.9836441852061112
                              0.0703416078843675,
                              0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              -0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L46     = -96462
   FRAME_-96462_NAME       = 'SPP_EPILO_L46'
   FRAME_-96462_CLASS      = 4
   FRAME_-96462_CLASS_ID   = -96462
   FRAME_-96462_CENTER     = -96
   TKFRAME_-96462_SPEC     = 'MATRIX'
   TKFRAME_-96462_RELATIVE = 'SPP_EPILO_W4'
   TKFRAME_-96462_MATRIX   = (0.9903244000533897
                              -0.0984911534882631,
                              -0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L47     = -96463
   FRAME_-96463_NAME       = 'SPP_EPILO_L47'
   FRAME_-96463_CLASS      = 4
   FRAME_-96463_CLASS_ID   = -96463
   FRAME_-96463_CENTER     = -96
   TKFRAME_-96463_SPEC     = 'MATRIX'
   TKFRAME_-96463_RELATIVE = 'SPP_EPILO_W4'
   TKFRAME_-96463_MATRIX   = (0.9903244000533897
                              0.0984911534882631,
                              0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              -0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L48     = -96464
   FRAME_-96464_NAME       = 'SPP_EPILO_L48'
   FRAME_-96464_CLASS      = 4
   FRAME_-96464_CLASS_ID   = -96464
   FRAME_-96464_CENTER     = -96
   TKFRAME_-96464_SPEC     = 'MATRIX'
   TKFRAME_-96464_RELATIVE = 'SPP_EPILO_W4'
   TKFRAME_-96464_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.3902632340374340,
                              -0.9207033225521906,
                              0.0000000000000000,
                              0.9207033225521906,
                              0.3902632340374340)

   FRAME_SPP_EPILO_L49     = -96465
   FRAME_-96465_NAME       = 'SPP_EPILO_L49'
   FRAME_-96465_CLASS      = 4
   FRAME_-96465_CLASS_ID   = -96465
   FRAME_-96465_CENTER     = -96
   TKFRAME_-96465_SPEC     = 'MATRIX'
   TKFRAME_-96465_RELATIVE = 'SPP_EPILO_W4'
   TKFRAME_-96465_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000001,
                              -1.0000000000000000,
                              0.0000000000000000,
                              1.0000000000000000,
                              0.0000000000000001)

   \begintext


   Wedge 5 aperture frames:

   \begindata

   FRAME_SPP_EPILO_L50     = -96467
   FRAME_-96467_NAME       = 'SPP_EPILO_L50'
   FRAME_-96467_CLASS      = 4
   FRAME_-96467_CLASS_ID   = -96467
   FRAME_-96467_CENTER     = -96
   TKFRAME_-96467_SPEC     = 'MATRIX'
   TKFRAME_-96467_RELATIVE = 'SPP_EPILO_W5'
   TKFRAME_-96467_MATRIX   = (0.9569185564499445
                              -0.0027262750100587,
                              -0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L51     = -96468
   FRAME_-96468_NAME       = 'SPP_EPILO_L51'
   FRAME_-96468_CLASS      = 4
   FRAME_-96468_CLASS_ID   = -96468
   FRAME_-96468_CENTER     = -96
   TKFRAME_-96468_SPEC     = 'MATRIX'
   TKFRAME_-96468_RELATIVE = 'SPP_EPILO_W5'
   TKFRAME_-96468_MATRIX   = (0.9951765914819487
                              -0.0009211052643909,
                              -0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L52     = -96469
   FRAME_-96469_NAME       = 'SPP_EPILO_L52'
   FRAME_-96469_CLASS      = 4
   FRAME_-96469_CLASS_ID   = -96469
   FRAME_-96469_CENTER     = -96
   TKFRAME_-96469_SPEC     = 'MATRIX'
   TKFRAME_-96469_RELATIVE = 'SPP_EPILO_W5'
   TKFRAME_-96469_MATRIX   = (0.9951765914819487
                              0.0009211052643909,
                              0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              -0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L53     = -96470
   FRAME_-96470_NAME       = 'SPP_EPILO_L53'
   FRAME_-96470_CLASS      = 4
   FRAME_-96470_CLASS_ID   = -96470
   FRAME_-96470_CENTER     = -96
   TKFRAME_-96470_SPEC     = 'MATRIX'
   TKFRAME_-96470_RELATIVE = 'SPP_EPILO_W5'
   TKFRAME_-96470_MATRIX   = (0.9569185564499445
                              0.0027262750100587,
                              0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              -0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L54     = -96471
   FRAME_-96471_NAME       = 'SPP_EPILO_L54'
   FRAME_-96471_CLASS      = 4
   FRAME_-96471_CLASS_ID   = -96471
   FRAME_-96471_CENTER     = -96
   TKFRAME_-96471_SPEC     = 'MATRIX'
   TKFRAME_-96471_RELATIVE = 'SPP_EPILO_W5'
   TKFRAME_-96471_MATRIX   = (0.9836441852061112
                              -0.0703416078843675,
                              -0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L55     = -96472
   FRAME_-96472_NAME       = 'SPP_EPILO_L55'
   FRAME_-96472_CLASS      = 4
   FRAME_-96472_CLASS_ID   = -96472
   FRAME_-96472_CENTER     = -96
   TKFRAME_-96472_SPEC     = 'MATRIX'
   TKFRAME_-96472_RELATIVE = 'SPP_EPILO_W5'
   TKFRAME_-96472_MATRIX   = (0.9836441852061112
                              0.0703416078843675,
                              0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              -0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L56     = -96473
   FRAME_-96473_NAME       = 'SPP_EPILO_L56'
   FRAME_-96473_CLASS      = 4
   FRAME_-96473_CLASS_ID   = -96473
   FRAME_-96473_CENTER     = -96
   TKFRAME_-96473_SPEC     = 'MATRIX'
   TKFRAME_-96473_RELATIVE = 'SPP_EPILO_W5'
   TKFRAME_-96473_MATRIX   = (0.9903244000533897
                              -0.0984911534882631,
                              -0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L57     = -96474
   FRAME_-96474_NAME       = 'SPP_EPILO_L57'
   FRAME_-96474_CLASS      = 4
   FRAME_-96474_CLASS_ID   = -96474
   FRAME_-96474_CENTER     = -96
   TKFRAME_-96474_SPEC     = 'MATRIX'
   TKFRAME_-96474_RELATIVE = 'SPP_EPILO_W5'
   TKFRAME_-96474_MATRIX   = (0.9903244000533897
                              0.0984911534882631,
                              0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              -0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L58     = -96475
   FRAME_-96475_NAME       = 'SPP_EPILO_L58'
   FRAME_-96475_CLASS      = 4
   FRAME_-96475_CLASS_ID   = -96475
   FRAME_-96475_CENTER     = -96
   TKFRAME_-96475_SPEC     = 'MATRIX'
   TKFRAME_-96475_RELATIVE = 'SPP_EPILO_W5'
   TKFRAME_-96475_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.3902632340374340,
                              -0.9207033225521906,
                              0.0000000000000000,
                              0.9207033225521906,
                              0.3902632340374340)

   FRAME_SPP_EPILO_L59     = -96476
   FRAME_-96476_NAME       = 'SPP_EPILO_L59'
   FRAME_-96476_CLASS      = 4
   FRAME_-96476_CLASS_ID   = -96476
   FRAME_-96476_CENTER     = -96
   TKFRAME_-96476_SPEC     = 'MATRIX'
   TKFRAME_-96476_RELATIVE = 'SPP_EPILO_W5'
   TKFRAME_-96476_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000001,
                              -1.0000000000000000,
                              0.0000000000000000,
                              1.0000000000000000,
                              0.0000000000000001)

   \begintext


   Wedge 6 aperture frames:

   \begindata

   FRAME_SPP_EPILO_L60     = -96478
   FRAME_-96478_NAME       = 'SPP_EPILO_L60'
   FRAME_-96478_CLASS      = 4
   FRAME_-96478_CLASS_ID   = -96478
   FRAME_-96478_CENTER     = -96
   TKFRAME_-96478_SPEC     = 'MATRIX'
   TKFRAME_-96478_RELATIVE = 'SPP_EPILO_W6'
   TKFRAME_-96478_MATRIX   = (0.9569185564499445
                              -0.0027262750100587,
                              -0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L61     = -96479
   FRAME_-96479_NAME       = 'SPP_EPILO_L61'
   FRAME_-96479_CLASS      = 4
   FRAME_-96479_CLASS_ID   = -96479
   FRAME_-96479_CENTER     = -96
   TKFRAME_-96479_SPEC     = 'MATRIX'
   TKFRAME_-96479_RELATIVE = 'SPP_EPILO_W6'
   TKFRAME_-96479_MATRIX   = (0.9951765914819487
                              -0.0009211052643909,
                              -0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L62     = -96480
   FRAME_-96480_NAME       = 'SPP_EPILO_L62'
   FRAME_-96480_CLASS      = 4
   FRAME_-96480_CLASS_ID   = -96480
   FRAME_-96480_CENTER     = -96
   TKFRAME_-96480_SPEC     = 'MATRIX'
   TKFRAME_-96480_RELATIVE = 'SPP_EPILO_W6'
   TKFRAME_-96480_MATRIX   = (0.9951765914819487
                              0.0009211052643909,
                              0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              -0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L63     = -96481
   FRAME_-96481_NAME       = 'SPP_EPILO_L63'
   FRAME_-96481_CLASS      = 4
   FRAME_-96481_CLASS_ID   = -96481
   FRAME_-96481_CENTER     = -96
   TKFRAME_-96481_SPEC     = 'MATRIX'
   TKFRAME_-96481_RELATIVE = 'SPP_EPILO_W6'
   TKFRAME_-96481_MATRIX   = (0.9569185564499445
                              0.0027262750100587,
                              0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              -0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L64     = -96482
   FRAME_-96482_NAME       = 'SPP_EPILO_L64'
   FRAME_-96482_CLASS      = 4
   FRAME_-96482_CLASS_ID   = -96482
   FRAME_-96482_CENTER     = -96
   TKFRAME_-96482_SPEC     = 'MATRIX'
   TKFRAME_-96482_RELATIVE = 'SPP_EPILO_W6'
   TKFRAME_-96482_MATRIX   = (0.9836441852061112
                              -0.0703416078843675,
                              -0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L65     = -96483
   FRAME_-96483_NAME       = 'SPP_EPILO_L65'
   FRAME_-96483_CLASS      = 4
   FRAME_-96483_CLASS_ID   = -96483
   FRAME_-96483_CENTER     = -96
   TKFRAME_-96483_SPEC     = 'MATRIX'
   TKFRAME_-96483_RELATIVE = 'SPP_EPILO_W6'
   TKFRAME_-96483_MATRIX   = (0.9836441852061112
                              0.0703416078843675,
                              0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              -0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L66     = -96484
   FRAME_-96484_NAME       = 'SPP_EPILO_L66'
   FRAME_-96484_CLASS      = 4
   FRAME_-96484_CLASS_ID   = -96484
   FRAME_-96484_CENTER     = -96
   TKFRAME_-96484_SPEC     = 'MATRIX'
   TKFRAME_-96484_RELATIVE = 'SPP_EPILO_W6'
   TKFRAME_-96484_MATRIX   = (0.9903244000533897
                              -0.0984911534882631,
                              -0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L67     = -96485
   FRAME_-96485_NAME       = 'SPP_EPILO_L67'
   FRAME_-96485_CLASS      = 4
   FRAME_-96485_CLASS_ID   = -96485
   FRAME_-96485_CENTER     = -96
   TKFRAME_-96485_SPEC     = 'MATRIX'
   TKFRAME_-96485_RELATIVE = 'SPP_EPILO_W6'
   TKFRAME_-96485_MATRIX   = (0.9903244000533897
                              0.0984911534882631,
                              0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              -0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L68     = -96486
   FRAME_-96486_NAME       = 'SPP_EPILO_L68'
   FRAME_-96486_CLASS      = 4
   FRAME_-96486_CLASS_ID   = -96486
   FRAME_-96486_CENTER     = -96
   TKFRAME_-96486_SPEC     = 'MATRIX'
   TKFRAME_-96486_RELATIVE = 'SPP_EPILO_W6'
   TKFRAME_-96486_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.3902632340374340,
                              -0.9207033225521906,
                              0.0000000000000000,
                              0.9207033225521906,
                              0.3902632340374340)

   FRAME_SPP_EPILO_L69     = -96487
   FRAME_-96487_NAME       = 'SPP_EPILO_L69'
   FRAME_-96487_CLASS      = 4
   FRAME_-96487_CLASS_ID   = -96487
   FRAME_-96487_CENTER     = -96
   TKFRAME_-96487_SPEC     = 'MATRIX'
   TKFRAME_-96487_RELATIVE = 'SPP_EPILO_W6'
   TKFRAME_-96487_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000001,
                              -1.0000000000000000,
                              0.0000000000000000,
                              1.0000000000000000,
                              0.0000000000000001)

   \begintext


   Wedge 7 aperture frames:

   \begindata

   FRAME_SPP_EPILO_L70     = -96489
   FRAME_-96489_NAME       = 'SPP_EPILO_L70'
   FRAME_-96489_CLASS      = 4
   FRAME_-96489_CLASS_ID   = -96489
   FRAME_-96489_CENTER     = -96
   TKFRAME_-96489_SPEC     = 'MATRIX'
   TKFRAME_-96489_RELATIVE = 'SPP_EPILO_W7'
   TKFRAME_-96489_MATRIX   = (0.9569185564499445
                              -0.0027262750100587,
                              -0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L71     = -96490
   FRAME_-96490_NAME       = 'SPP_EPILO_L71'
   FRAME_-96490_CLASS      = 4
   FRAME_-96490_CLASS_ID   = -96490
   FRAME_-96490_CENTER     = -96
   TKFRAME_-96490_SPEC     = 'MATRIX'
   TKFRAME_-96490_RELATIVE = 'SPP_EPILO_W7'
   TKFRAME_-96490_MATRIX   = (0.9951765914819487
                              -0.0009211052643909,
                              -0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L72     = -96491
   FRAME_-96491_NAME       = 'SPP_EPILO_L72'
   FRAME_-96491_CLASS      = 4
   FRAME_-96491_CLASS_ID   = -96491
   FRAME_-96491_CENTER     = -96
   TKFRAME_-96491_SPEC     = 'MATRIX'
   TKFRAME_-96491_RELATIVE = 'SPP_EPILO_W7'
   TKFRAME_-96491_MATRIX   = (0.9951765914819487
                              0.0009211052643909,
                              0.0980953787467206,
                              0.0000000000000000,
                              0.9999559178549919,
                              -0.0093894806448862,
                              -0.0980997031920620,
                              0.0093441913439635,
                              0.9951327219631343)

   FRAME_SPP_EPILO_L73     = -96492
   FRAME_-96492_NAME       = 'SPP_EPILO_L73'
   FRAME_-96492_CLASS      = 4
   FRAME_-96492_CLASS_ID   = -96492
   FRAME_-96492_CENTER     = -96
   TKFRAME_-96492_SPEC     = 'MATRIX'
   TKFRAME_-96492_RELATIVE = 'SPP_EPILO_W7'
   TKFRAME_-96492_MATRIX   = (0.9569185564499445
                              0.0027262750100587,
                              0.2903436648978653,
                              0.0000000000000000,
                              0.9999559185467560,
                              -0.0093894069734682,
                              -0.2903564642327675,
                              0.0089848977669722,
                              0.9568763740893401)

   FRAME_SPP_EPILO_L74     = -96493
   FRAME_-96493_NAME       = 'SPP_EPILO_L74'
   FRAME_-96493_CLASS      = 4
   FRAME_-96493_CLASS_ID   = -96493
   FRAME_-96493_CENTER     = -96
   TKFRAME_-96493_SPEC     = 'MATRIX'
   TKFRAME_-96493_RELATIVE = 'SPP_EPILO_W7'
   TKFRAME_-96493_MATRIX   = (0.9836441852061112
                              -0.0703416078843675,
                              -0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L75     = -96494
   FRAME_-96494_NAME       = 'SPP_EPILO_L75'
   FRAME_-96494_CLASS      = 4
   FRAME_-96494_CLASS_ID   = -96494
   FRAME_-96494_CENTER     = -96
   TKFRAME_-96494_SPEC     = 'MATRIX'
   TKFRAME_-96494_RELATIVE = 'SPP_EPILO_W7'
   TKFRAME_-96494_MATRIX   = (0.9836441852061112
                              0.0703416078843675,
                              0.1658197066408207,
                              0.0000000000000000,
                              0.9205940500233174,
                              -0.3905209278152269,
                              -0.1801225052851696,
                              0.3841336398467433,
                              0.9055369842407799)

   FRAME_SPP_EPILO_L76     = -96495
   FRAME_-96495_NAME       = 'SPP_EPILO_L76'
   FRAME_-96495_CLASS      = 4
   FRAME_-96495_CLASS_ID   = -96495
   FRAME_-96495_CENTER     = -96
   TKFRAME_-96495_SPEC     = 'MATRIX'
   TKFRAME_-96495_RELATIVE = 'SPP_EPILO_W7'
   TKFRAME_-96495_MATRIX   = (0.9903244000533897
                              -0.0984911534882631,
                              -0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L77     = -96496
   FRAME_-96496_NAME       = 'SPP_EPILO_L77'
   FRAME_-96496_CLASS      = 4
   FRAME_-96496_CLASS_ID   = -96496
   FRAME_-96496_CENTER     = -96
   TKFRAME_-96496_SPEC     = 'MATRIX'
   TKFRAME_-96496_RELATIVE = 'SPP_EPILO_W7'
   TKFRAME_-96496_MATRIX   = (0.9903244000533897
                              0.0984911534882631,
                              0.0977602953322317,
                              0.0000000000000000,
                              0.7044685664732034,
                              -0.7097351892439814,
                              -0.1387716925705447,
                              0.7028680754848248,
                              0.6976524104490466)

   FRAME_SPP_EPILO_L78     = -96497
   FRAME_-96497_NAME       = 'SPP_EPILO_L78'
   FRAME_-96497_CLASS      = 4
   FRAME_-96497_CLASS_ID   = -96497
   FRAME_-96497_CENTER     = -96
   TKFRAME_-96497_SPEC     = 'MATRIX'
   TKFRAME_-96497_RELATIVE = 'SPP_EPILO_W7'
   TKFRAME_-96497_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.3902632340374340,
                              -0.9207033225521906,
                              0.0000000000000000,
                              0.9207033225521906,
                              0.3902632340374340)

   FRAME_SPP_EPILO_L79     = -96498
   FRAME_-96498_NAME       = 'SPP_EPILO_L79'
   FRAME_-96498_CLASS      = 4
   FRAME_-96498_CLASS_ID   = -96498
   FRAME_-96498_CENTER     = -96
   TKFRAME_-96498_SPEC     = 'MATRIX'
   TKFRAME_-96498_RELATIVE = 'SPP_EPILO_W7'
   TKFRAME_-96498_MATRIX   = (1.0000000000000000
                              -0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000000,
                              0.0000000000000001,
                              -1.0000000000000000,
                              0.0000000000000000,
                              1.0000000000000000,
                              0.0000000000000001)

   \begintext


EPI-Hi Frames
===========================================================================

TBD

