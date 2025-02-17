KPL/FK

Interstellar Mapping and Acceleration Probe Dynamic Frames Kernel
========================================================================

   This kernel contains SPICE frame definitions to support the IMAP mission.

   To be consistent with conventions prior to the mission name change
   from Solar Probe Plus (SPP) to Parker Solar Probe (PSP) in 2017, frame
   names and other text IDs are prefixed with the legacy acronym, SPP.

Version and Date
---------------------------------------------------------------

   The TEXT_KERNEL_ID stores version information of loaded project text
   kernels.  Each entry associated with the keyword is a string that
   consists of four parts: the kernel name, version, entry date, and type.

   SPP Dynamic Frame Kernel Version:

      \begindata

         TEXT_KERNEL_ID = 'IMAP_DYNAMIC_FRAMES V1.0.0 2024-XXXX-NN FK'

      \begintext


   Version 1.0.0 -- April 10, 2024 -- Nick Dutton (JHU/APL)


References
---------------------------------------------------------------

      1.   NAIF SPICE `Kernel Pool Required Reading'

      2.   NAIF SPICE `Frames Required Reading'

      3.   Email from Scott Turner, received May 2, 2018, containing
           attachment InstrumentFrames.pptx, by Martha Kusterer dated
           Sept. 19, 2017

      4.   msgr_dyn_v600.tf, in Planetary Data System (PDS)
           data set MESS-E/V/H-SPICE-6-V1.0

      5.   stereo_rtn.tf, at
           ftp://sohoftp.nascom.nasa.gov/solarsoft/stereo/gen/data/spice

      6.   heliospheric.tf, at
           ftp://sohoftp.nascom.nasa.gov/solarsoft/stereo/gen/data/spice/gen

      7.   Email from Scott Turner received May 11, 2018 containing notes
           taken from the science team meeting on the same date.

      8.   Snodgrass, H.B., Ulrich, R.K., 1990, Rotation of Doppler features
           in the solar photosphere. Astrophys. J. 351, 309. doi:10.1086/168467


Contact Information
---------------------------------------------------------------

   Direct questions, comments, or concerns about the contents of this
   kernel to:

      Scott Turner, JHUAPL, (443)778-1693, Scott.Turner@jhuapl.edu

   or

      Lillian Nguyen, JHUAPL (443)778-5477, Lillian.Nguyen@jhuapl.edu

   or

      Douglas Rodgers, JHUAPL (443)778-4228, Douglas.Rodgers@jhuapl.edu


Implementation Notes
---------------------------------------------------------------

   This file is used by the SPICE system as follows: programs that make use
   of this frame kernel must `load' the kernel normally during program
   initialization.  Loading the kernel associates the data items with
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


IMAP Dynamic Frames
---------------------------------------------------------------

   This frame kernel defines a series of dynamic frames listed in [3]
   that support Parker Solar Probe data reduction and analysis.  All
   of the frame names defined by this kernel are prefixed with 'SPP_'
   (the legacy mission name acronym as described in the leading paragraph
   of this file) to avoid conflict with alternative definitions not
   specific to the project. Further, the project-specific ID codes
   -96900 to -96999 have been set aside to support these dynamic frames.

   The following dynamic frames are defined in this kernel file:

      Frame Name               Relative To              Type     NAIF ID
      ======================   ===================      =======  =======

      Earth Based Frames:
      ------------------
      EARTH_FIXED              IAU_EARTH                FIXED
      IMAP_RTN                 J2000                    DYNAMIC  -96900
      SPP_GSE                  J2000                    DYNAMIC  -96901

      Mercury Based Frames:
      ------------------
      SPP_MSO                  J2000                    DYNAMIC  -96903

      Venus Based Frames:
      ------------------
      SPP_VSO                  J2000                    DYNAMIC  -96904

      Sun Based Frames:
      ------------------
      SPP_HG                   J2000                    DYNAMIC  -96910
      SPP_HCI                  J2000                    DYNAMIC  -96911
      SPP_HEE                  J2000                    DYNAMIC  -96912
      SPP_HEEQ                 J2000                    DYNAMIC  -96913
      SPP_RTN                  J2000                    DYNAMIC  -96914
      SPP_HERTN                J2000                    DYNAMIC  -96915
      SPP_HGI                  J2000                    DYNAMIC  -96916
      SPP_HGDOPP               J2000                    DYNAMIC  -96917
      SPP_HGMAG                J2000                    DYNAMIC  -96918
      SPP_HGSPEC               J2000                    DYNAMIC  -96919



   \begindata

      NAIF_BODY_NAME   += ( 'IMAP_DPS' )
      NAIF_BODY_CODE   += ( -43901     )



   \begintext

DPS Frame

\begindata

        FRAME_IMAP_DPS              = -43901
        FRAME_-43901_NAME           = 'IMAP_DPS'
        FRAME_-43901_CLASS          = 3
        FRAME_-43901_CLASS_ID       = -43901
        FRAME_-43901_CENTER         = -43
        CK_-43901_SCLK              = -43

   \begintext

END OF FILE
