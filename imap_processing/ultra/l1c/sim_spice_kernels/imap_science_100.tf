KPL/FK

Interstellar Mapping and Acceleration Probe (IMAP) Dynamic Frames Kernel
========================================================================

   This kernel contains SPICE frame definitions to support the
   IMAP mission.

   This kernel is composed of primarily dynamic frames, but in general
   it  holds frame definitions for all instrument-agnostic frames, CK
   frames used in science data processing and mapping.


Version and Date
------------------------------------------------------------------------

   The TEXT_KERNEL_ID stores version information of loaded project
   text kernels. Each entry associated with the keyword is a string
   that consists of four parts: the kernel name, version, entry date,
   and type.

   IMAP Dynamic Frame Kernel Version:

   \begindata

      TEXT_KERNEL_ID = 'IMAP_DYNAMIC_FRAMES V0.0.1 2025-JUNE-26 FK'

   \begintext

   Version 0.0.0 -- April 10, 2024 -- Nick Dutton (JHU/APL)
   Version 0.0.1 -- June  26, 2025 -- Nick Dutton (JHU/APL)
   Version 1.0.0 -- July   8, 2025 -- Nick and Doug (JHU/APL)


References
------------------------------------------------------------------------

      1.   NAIF SPICE `Kernel Pool Required Reading'

      2.   NAIF SPICE `Frames Required Reading'

      3.   "IMAP Coordinate Frame Science.pdf"

      4.   stereo_rtn.tf, at
           https://soho.nascom.nasa.gov/solarsoft/stereo/...
           ...gen/data/spice/gen/stereo_rtn.tf

      5.   heliospheric.tf, at
           https://soho.nascom.nasa.gov/solarsoft/stereo/...
           ...gen/data/spice/gen/heliospheric.tf

      6.   "Geophysical Coordinate Transformations", C. T. Russell

      7.   "Heliospheric Coordinate Systems", M. Franz and D. Harper

      8.   "Global observations of the interstellar interaction from the
           Interstellar Boundary Explorer (IBEX)", D. J. McComas, et al.

      9.   "Very Local Interstellar Medium Revealed by a Complete Solar
           Cycle of Interstellar Neutral Helium Observations with IBEX",
           P. Swaczyna, et al.

      10.  Lagrange L1 definition and SPK, Min-Kun Chung,
           https://naif.jpl.nasa.gov/pub/naif/...
           ...generic_kernels/spk/lagrange_point/


Contact Information
------------------------------------------------------------------------

   Direct questions, comments, or concerns about the contents of this
   kernel to:

      Nick Dutton, JHUAPL, Nicholas.Dutton@jhuapl.edu

   or

      Doug Rodgers, JHUAPL, Douglas.Rodgers@jhuapl.edu

   or

      Lillian Nguyen, JHUAPL, Lillian.Nguyen@jhuapl.edu


Implementation Notes
------------------------------------------------------------------------

   This file is used by the SPICE system as follows: programs that make
   use of this frame kernel must `load' the kernel normally during
   program initialization. Loading the kernel associates the data items
   with their names in a data structure called the `kernel pool'. The
   SPICELIB routine FURNSH loads a kernel into the pool as shown below:

      Python: (SpiceyPy)

         spiceypy.furnsh( frame_kernel_name )

      IDL: (ICY)

         cspice_furnsh, frame_kernel_name

      MATLAB: (MICE)

         cspice_furnsh ( frame_kernel_name )

      C: (CSPICE)

         furnsh_c ( frame_kernel_name );

      FORTRAN: (SPICELIB)

         CALL FURNSH ( frame_kernel_name )

   This file was created, and may be updated with, a text editor or word
   processor.


IMAP Science Frames
========================================================================

   This frame kernel defines a series of frames listed in [3] that
   support IMAP data reduction and analysis. All of the frame names
   assigned an IMAP NAIF ID (beginning with -43) defined by this kernel
   are prefixed with 'IMAP_' to avoid conflict with alternative
   definitions not specific to the project.

   The project-specific ID codes -43900 to -43999 have been set aside to
   support these dynamic frames.


      Frame Name                Relative To        Type        NAIF ID
      ======================    ===============    ========    =======

      IMAP Based Frames:
      ----------------------
      IMAP_OMD                  IMAP_SPACECRAFT    FIXED       -43900
      IMAP_DPS                  [n/a]              CK          -43901

      Earth Based Frames:
      ----------------------
      IMAP_EARTHFIXED           IAU_EARTH          FIXED       -43910
      IMAP_ECLIPDATE            J2000              DYNAMIC     -43911
      IMAP_MDI                  ECLIPJ2000         FIXED       -43912
      IMAP_MDR                  J2000              DYNAMIC     -43913
      IMAP_GMC                  IAU_EARTH          DYNAMIC     -43914
      IMAP_GEI                  J2000              FIXED       -43915
      IMAP_GSE                  J2000              DYNAMIC     -43916
      IMAP_GSM                  J2000              DYNAMIC     -43917
      IMAP_SMD                  J2000              DYNAMIC     -43918

      Sun Based Frames:
      ----------------------
      IMAP_RTN                  J2000              DYNAMIC     -43920
      IMAP_HCI (ie, HGI_J2K)    J2000              DYNAMIC     -43921
      IMAP_HCD (ie, HGI_D)      J2000              DYNAMIC     -43922
      IMAP_HGC (ie, HGS_D)      IAU_SUN            FIXED       -43923
      IMAP_HAE                  ECLIPJ2000         FIXED       -43924
      IMAP_HAED                 IMAP_ECLIPDATE     FIXED       -43925
      IMAP_HEE                  J2000              DYNAMIC     -43926
      IMAP_HRE                  J2000              DYNAMIC     -43927
      IMAP_HNU                  J2000              DYNAMIC     -43928
      IMAP_GCS                  GALACTIC           FIXED       -43929


IMAP Based Frames
========================================================================

   These dynamic frames are used for analyzing data in a reference
   frame tied to the dynamics of IMAP.


   Observatory Mechanical Design (OMD) Frame ([3])
   ---------------------------------------------------------------------

      Alias for IMAP_SPACECRAFT frame defined in the primary
      'imap_vNNN.tf' frame kernel. From that file:

      Origin:   Center of the launch vehicle adapter ring at the
                observatory/launch vehicle interface plane

      +Z axis:  Perpendicular to the launch vehicle interface plane
                pointed in the direction of the top deck (runs through
                the center of the central cylinder structure element)

      +Y axis:  Direction of the vector orthogonal to the +Z axis and
                parallel to the deployed MAG boom

      +X axis:  The third orthogonal axis defined using an X, Y, Z
                ordered right hand rule

   \begindata

      FRAME_IMAP_EARTHFIXED   = -43900
      FRAME_-43900_NAME       = 'IMAP_OMD'
      FRAME_-43900_CLASS      = 4
      FRAME_-43900_CLASS_ID   = -43900
      FRAME_-43900_CENTER     = -43
      TKFRAME_-43900_RELATIVE = 'IMAP_SPACECRAFT'
      TKFRAME_-43900_SPEC     = 'MATRIX'
      TKFRAME_-43900_MATRIX   = ( 1  0  0
                                  0  1  0
                                  0  0  1 )

   \begintext


   Despun Pointing Sets (DPS) Frame ([3])
   ---------------------------------------------------------------------

      Coordinate frame used for ENA imager data processing and
      intentionally designed for use in producing all-sky map products.

      This is provided by a CK file external to this file. Notionally,
      the frame is defined:

      +Z axis is parallel to the nominal spin axis of the spacecraft.
      The axis is notionally a time-average of the spin axis of the
      exact orientation (IMAP_SPACECRAFT or IMAP_OMD).

      Y = Z cross Necliptic where Necliptic is the the unit normal
      (North) to the ecliptic plane.

      This is a quasi-inertial reference frame and will have a unique
      transformation matrix, valid between repointings of the spacecraft

   \begindata

      FRAME_IMAP_DPS              = -43901
      FRAME_-43901_NAME           = 'IMAP_DPS'
      FRAME_-43901_CLASS          =  3
      FRAME_-43901_CLASS_ID       =  -43901
      FRAME_-43901_CENTER         = -43
      CK_-43901_SCLK              = -43
      CK_-43901_SPK               = -43

   \begintext


Earth Based Frames
========================================================================

   These dynamic frames are used for analyzing data in a reference
   frame tied to the dynamics of Earth.


   Earth-Fixed Frame (IMAP_EARTHFIXED)
   ---------------------------------------------------------------------

      Some of these Earth based dynamic frames reference vectors in an
      Earth-fixed frame. To support loading of either rotation model
      (IAU_EARTH or ITRF93), the following keywords control which model
      is used. The model is enabled by surrounding its keyword-value
      block  with the \begindata and \begintext markers (currently
      IAU_EARTH).

      IAU_EARTH based model is currently employed:

   \begindata

      FRAME_IMAP_EARTHFIXED   = -43910
      FRAME_-43910_NAME       = 'IMAP_EARTHFIXED'
      FRAME_-43910_CLASS      = 4
      FRAME_-43910_CLASS_ID   = -43910
      FRAME_-43910_CENTER     = 399
      TKFRAME_-43910_RELATIVE = 'IAU_EARTH'
      TKFRAME_-43910_SPEC     = 'MATRIX'
      TKFRAME_-43910_MATRIX   = ( 1  0  0
                                  0  1  0
                                  0  0  1 )

   \begintext

      Alternately, the more precise ITRF93-based model could be used:

      FRAME_IMAP_EARTHFIXED   = -43910
      FRAME_-43910_NAME       = 'IMAP_EARTHFIXED'
      FRAME_-43910_CLASS      = 4
      FRAME_-43910_CLASS_ID   = -43910
      FRAME_-43910_CENTER     = 399
      TKFRAME_-43910_RELATIVE = 'ITRF93'
      TKFRAME_-43910_SPEC     = 'MATRIX'
      TKFRAME_-43910_MATRIX   = ( 1  0  0
                                  0  1  0
                                  0  0  1 )

      However, using the ITRF93 frame requires supplying SPICE with
      sufficient binary PCK data to cover the period of interest.
      The IAU_EARTH frame just requires a text PCK with Earth data
      to be loaded.


   Mean Ecliptic of Date (IMAP_ECLIPDATE) ([2],[5])
   ---------------------------------------------------------------------

      Mean Ecliptic of Date is the more precise, rotating counterpart
      to the inertial Mean Ecliptic and Equinox of J2000 (ECLIPJ2000).

      If computations involving this frame (or frames relative to this)
      are too expensive, the user may instruct SPICE to ignore
      rotational effects by changing 'ROTATING' to 'INERTIAL'.

      The X axis is the first point in Aries for the mean ecliptic of
      date and the Z axis points along the ecliptic north pole.

      The Y axis is Z cross X, completing the right-handed reference
      frame.

   \begindata

      FRAME_IMAP_ECLIPDATE        = -43911
      FRAME_-43911_NAME           = 'IMAP_ECLIPDATE'
      FRAME_-43911_CLASS          = 5
      FRAME_-43911_CLASS_ID       = -43911
      FRAME_-43911_CENTER         = 399
      FRAME_-43911_RELATIVE       = 'J2000'
      FRAME_-43911_DEF_STYLE      = 'PARAMETERIZED'
      FRAME_-43911_FAMILY         = 'MEAN_ECLIPTIC_AND_EQUINOX_OF_DATE'
      FRAME_-43911_PREC_MODEL     = 'EARTH_IAU_1976'
      FRAME_-43911_OBLIQ_MODEL    = 'EARTH_IAU_1980'
      FRAME_-43911_ROTATION_STATE = 'ROTATING'

   \begintext


   Mission Design Inertial (MDI) Frame ([3])
   ---------------------------------------------------------------------

      Alias for SPICE ECLIPJ2000.

      Primary coordinate frame used to define IMAP's trajectory and
      orbit, as well as for some science data products.

      The X axis is the first point in Aries for the mean ecliptic of
      J2000 and the Z axis points along the ecliptic north pole.

      The Y axis is Z cross X, completing the right-handed reference
      frame.

   \begindata

      FRAME_IMAP_MDI          = -43912
      FRAME_-43912_NAME       = 'IMAP_MDI'
      FRAME_-43912_CLASS      = 4
      FRAME_-43912_CLASS_ID   = -43912
      FRAME_-43912_CENTER     = 399
      TKFRAME_-43912_RELATIVE = 'IMAP_EARTHFIXED'
      TKFRAME_-43912_SPEC     = 'MATRIX'
      TKFRAME_-43912_MATRIX   = ( 1  0  0
                                  0  1  0
                                  0  0  1 )

   \begintext


   Mission Design Rotating (MDR) Frame ([3],[10])
   ---------------------------------------------------------------------

      IMAP observatory body coordinate frame.

      The origin of the frame is the L1 point of the Sun and the Earth-
      Moon barycenter defined in SPK 'L1_de431.bsp' by reference [10];
      this author assigned the NAIF body code 391 to this L1 point.

      The position of the Earth-Moon barycenter relative to the Sun is
      the primary vector: the X axis points from the Sun to the
      Earth-Moon barycenter.

      The northern surface normal to the mean ecliptic of date is the
      secondary vector: the Z axis is the component of this vector
      orthogonal to the X axis. Combined with the definition of the
      X axis, this yields a unit vector along the angular momentum
      vector of the Earth-Moon barycenter orbiting the Sun.

      The Y axis is Z cross X, completing the right-handed reference
      frame.

      All vectors are geometric: no aberration corrections are used.

   \begindata

      FRAME_IMAP_MDR              = -43913
      FRAME_-43913_NAME           = 'IMAP_MDR'
      FRAME_-43913_CLASS          =  5
      FRAME_-43913_CLASS_ID       =  -43913
      FRAME_-43913_CENTER         =  391
      FRAME_-43913_RELATIVE       = 'J2000'
      FRAME_-43913_DEF_STYLE      = 'PARAMETERIZED'
      FRAME_-43913_FAMILY         = 'TWO-VECTOR'
      FRAME_-43913_PRI_AXIS       = 'X'
      FRAME_-43913_PRI_VECTOR_DEF = 'OBSERVER_TARGET_POSITION'
      FRAME_-43913_PRI_OBSERVER   = 'SUN'
      FRAME_-43913_PRI_TARGET     = 'EARTH MOON BARYCENTER'
      FRAME_-43913_PRI_ABCORR     = 'NONE'
      FRAME_-43913_SEC_AXIS       = 'Z'
      FRAME_-43913_SEC_VECTOR_DEF = 'CONSTANT'
      FRAME_-43913_SEC_FRAME      = 'IMAP_ECLIPDATE'
      FRAME_-43913_SEC_SPEC       = 'RECTANGULAR'
      FRAME_-43913_SEC_VECTOR     = ( 0, 0, 1 )

   \begintext


   Geomagnetic Coordinate (GMC) Frame (IGRF-14 Modeled Pole) ([6])
   ---------------------------------------------------------------------

      The geomagnetic coordinate (GMC) system is defined so that its
      Z-axis is parallel to the magnetic dipole. The geographic
      coordinates, D, of the dipole axis are found from the
      International Geomagnetic Reference Field.

      The Y-axis of this system is perpendicular to the geographic poles
      such that if D is the dipole position and S is the south pole
      Y=DxS. The X-axis completes a right-handed orthogonal set.

      The implementation of this frame is complicated in that the
      definition of the IGRF dipole is a function of time and the IGRF
      model cannot be directly incorporated into SPICE. However, SPICE
      does allow one to define time dependent Euler angles. Meaning, you
      can define a single Euler angle that rotates the Geocentric
      Equatorial Inertial (GEI) system to GMC for a given ephem time t:

         V     = r(t) * V
          GEI             GMC

      where r(t) is a time dependent Euler angle representation of a
      rotation. SPICE allows for the time dependence to be represented
      by a polynomial expansion. This expansion can be fit using the
      IGRF model, thus representing the IGRF dipole axis.

      IGRF-14 (the 14th version) was fit for the period of 1990-2035,
      which encompasses the mission and will also make this kernel
      useful for performing Magnetic dipole frame transformations for
      the 1990's and the 2000's. However, IGRF-14 is not as accurate for
      this entire time interval. The years between 1945-2020 are labeled
      definitive, although only back to 1990 was used in the polynomial
      fit. 2020-2025 is provisional, and may change with IGRF-15.
      2025-2030 was only a prediction. Beyond 2030, the predict is so
      far in the future as to not be valid. So to make the polynomials
      behave nicely in this region (in case someone does try to use this
      frame during that time), the 2030 prediction was extended until
      2035. So for low precision, this kernel can be used for the years
      2025-2035. Any times less than 1990 and greater than 2035 were not
      used in the fit, and therefore may be vastly incorrect as the
      polynomials may diverge outside of this region. These coefficients
      will be refit when IGRF-15 is released.

      Also, since the rest of the magnetic dipole frames are defined
      from this one, similar time ranges should be used for those frames

               Definitive         Provisional   Predict    Not Valid
      |--------------------------|+++++++++++|###########|???????????|
     1990                       2020        2025        2030        2035

      In addition to the error inherit in the model itself, the
      polynomial expansion cannot perfectly be fit the IGRF dipole. The
      maximum error on the fit is 0.2 milliradians, or 0.01 degrees,
      while the average error is 59 microradians or 0.003 degrees.

      The GMC frame is achieved by first rotating the IAU_EARTH frame
      about Z by the longitude degrees, and then rotating about the
      Y axis by the amount of latitude.

      NOTE: ITRF93 is much more accurate than IAU_EARTH, if precise
      Earth-Fixed coordinates are desired, then ITRF93 should be
      incorporated by changing RELATIVE of the IMAP_EARTHFIXED frame.

   \begindata

      FRAME_IMAP_GMC              = -43914
      FRAME_-43914_NAME           = 'IMAP_GMC'
      FRAME_-43914_CLASS          = 5
      FRAME_-43914_CLASS_ID       = -43914
      FRAME_-43914_CENTER         = 399
      FRAME_-43914_RELATIVE       = 'IMAP_EARTHFIXED'
      FRAME_-43914_DEF_STYLE      = 'PARAMETERIZED'
      FRAME_-43914_FAMILY         = 'EULER'
      FRAME_-43914_EPOCH          = @2010-JAN-1/00:00:00
      FRAME_-43914_AXES           = ( 3,  2,  1 )
      FRAME_-43914_UNITS          = 'DEGREES'
      FRAME_-43914_ANGLE_1_COEFFS = ( +72.21459071369075
                                      +2.5468902895893966E-9
                                      -9.716151847392007E-19
                                      -1.0433860683683533E-26
                                      +2.362766949492718E-36
                                      +3.3213862072412154E-44
                                      -3.5122239525813096E-54
                                      -4.264324158308002E-62
                                      +2.495064964115813E-72
                                      +1.8605789215176264E-80 )
      FRAME_-43914_ANGLE_2_COEFFS = ( -9.981781660857344
                                      +1.8136204417470554E-9
                                      +7.130241121790372E-19
                                      -2.215929597148403E-27
                                      -3.900143352851885E-36
                                      +6.599160686982152E-45
                                      +8.376429421972708E-54
                                      -1.07431639798394E-62
                                      -5.913960690205374E-72
                                      +6.775302680782905E-81 )
      FRAME_-43914_ANGLE_3_COEFFS = ( 0 )

   \begintext


   Geocentric Equatorial Inertial (GEI) Frame ([3],[6])
   ---------------------------------------------------------------------

      Alias for SPICE J2000 frame.

      The Geocentric Equatorial Inertial System (GEI) has its X-axis
      pointing from the Earth towards the first point of Aries (the
      position of the Sun at the vernal equinox). This direction is the
      intersection of the Earth's equatorial plane and the ecliptic
      plane and thus the X-axis lies in both planes. The Z-axis is
      parallel to the rotation axis of the Earth and Y completes the
      right-handed orthogonal set (Y = Z x X).

   \begindata

      FRAME_IMAP_GEI          = -43915
      FRAME_-43915_NAME       = 'IMAP_GEI'
      FRAME_-43915_CLASS      =  4
      FRAME_-43915_CLASS_ID   =  -43915
      FRAME_-43915_CENTER     =  399
      TKFRAME_-43915_RELATIVE = 'J2000'
      TKFRAME_-43915_SPEC     = 'MATRIX'
      TKFRAME_-43915_MATRIX   = ( 1  0  0
                                  0  1  0
                                  0  0  1 )

   \begintext


   Geocentric Solar Ecliptic (GSE) Frame ([3],[5])
   ---------------------------------------------------------------------

      Rotating geocentric frame in which Sun and Earth are fixed and the
      Z axis is the unit normal to the Ecliptic plane.

      The position of the Sun relative to the Earth is the primary
      vector: the X axis points from the Earth to the Sun.

      The northern surface normal to the mean ecliptic of date
      (IMAP_ECLIPDATE) is the secondary vector: the Z axis is the
      component of this vector orthogonal to the X axis.

      The Y axis is Z cross X, completing the right-handed frame.

      All vectors are geometric: no aberration corrections are used.

   \begindata

      FRAME_IMAP_GSE              = -43916
      FRAME_-43916_NAME           = 'IMAP_GSE'
      FRAME_-43916_CLASS          =  5
      FRAME_-43916_CLASS_ID       =  -43916
      FRAME_-43916_CENTER         =  399
      FRAME_-43916_RELATIVE       = 'J2000'
      FRAME_-43916_DEF_STYLE      = 'PARAMETERIZED'
      FRAME_-43916_FAMILY         = 'TWO-VECTOR'
      FRAME_-43916_PRI_AXIS       = 'X'
      FRAME_-43916_PRI_VECTOR_DEF = 'OBSERVER_TARGET_POSITION'
      FRAME_-43916_PRI_OBSERVER   = 'EARTH'
      FRAME_-43916_PRI_TARGET     = 'SUN'
      FRAME_-43916_PRI_ABCORR     = 'NONE'
      FRAME_-43916_SEC_AXIS       = 'Z'
      FRAME_-43916_SEC_VECTOR_DEF = 'CONSTANT'
      FRAME_-43916_SEC_FRAME      = 'IMAP_ECLIPDATE'
      FRAME_-43916_SEC_SPEC       = 'RECTANGULAR'
      FRAME_-43916_SEC_VECTOR     = ( 0, 0, 1 )

   \begintext


   Geocentric Solar Magnetospheric (GSM) Frame ([3],[5],[6])
   ---------------------------------------------------------------------

      Rotating geocentric frame in which Sun and Earth are fixed and the
      XZ plane contains Earth's magnetic dipole moment. Specifically,
      the dipole moment will vary in the XZ plane about the Z axis of
      this frame.

      The position of the Sun relative to the Earth is the primary
      vector: the X axis points from the Earth to the Sun.

      Earth's magnetic dipole axis (the +Z axis of IMAP_GMC) is the
      secondary vector: the Z axis is the component of this vector
      orthogonal to the X axis.

      The Y axis is Z cross X, completing the right-handed frame.

      All vectors are geometric: no aberration corrections are used.

   \begindata

      FRAME_IMAP_GSM              = -43917
      FRAME_-43917_NAME           = 'IMAP_GSM'
      FRAME_-43917_CLASS          =  5
      FRAME_-43917_CLASS_ID       =  -43917
      FRAME_-43917_CENTER         =  399
      FRAME_-43917_RELATIVE       = 'J2000'
      FRAME_-43917_DEF_STYLE      = 'PARAMETERIZED'
      FRAME_-43917_FAMILY         = 'TWO-VECTOR'
      FRAME_-43917_PRI_AXIS       = 'X'
      FRAME_-43917_PRI_VECTOR_DEF = 'OBSERVER_TARGET_POSITION'
      FRAME_-43917_PRI_OBSERVER   = 'EARTH'
      FRAME_-43917_PRI_TARGET     = 'SUN'
      FRAME_-43917_PRI_ABCORR     = 'NONE'
      FRAME_-43917_SEC_AXIS       = 'Z'
      FRAME_-43917_SEC_VECTOR_DEF = 'CONSTANT'
      FRAME_-43917_SEC_FRAME      = 'IMAP_GMC'
      FRAME_-43917_SEC_SPEC       = 'RECTANGULAR'
      FRAME_-43917_SEC_VECTOR     = (0, 0, 1)

   \begintext


   Solar Magnetic of Date (SMD) Frame ([3],[5],[6])
   ---------------------------------------------------------------------

      Rotating geocentric frame in which the Z axis is aligned with
      Earth's magnetic dipole moment, and the XZ plane contains the
      Earth-Sun vector. Specifically, the Earth-Sun vector will vary in
      the XZ plane about the X axis of this frame.

      Earth's magnetic dipole axis (the +Z axis of IMAP_GMC) is the
      primary vector and aligns with the Z axis of this frame.

      The position of the Sun relative to the Earth is the secondary
      vector: the X axis is the component of the Earth-Sun vector
      orthogonal to the Z axis.

      The Y axis is Z cross X, completing the right-handed frame.

      All vectors are geometric: no aberration corrections are used.

   \begindata

      FRAME_IMAP_SMD              = -43918
      FRAME_-43918_NAME           = 'IMAP_SMD'
      FRAME_-43918_CLASS          =  5
      FRAME_-43918_CLASS_ID       =  -43918
      FRAME_-43918_CENTER         =  399
      FRAME_-43918_RELATIVE       = 'J2000'
      FRAME_-43918_DEF_STYLE      = 'PARAMETERIZED'
      FRAME_-43918_FAMILY         = 'TWO-VECTOR'
      FRAME_-43918_PRI_AXIS       = 'Z'
      FRAME_-43918_PRI_VECTOR_DEF = 'CONSTANT'
      FRAME_-43918_PRI_FRAME      = 'IMAP_GMC'
      FRAME_-43918_PRI_SPEC       = 'RECTANGULAR'
      FRAME_-43918_PRI_VECTOR     = (0, 0, 1)
      FRAME_-43918_SEC_AXIS       = 'X'
      FRAME_-43918_SEC_VECTOR_DEF = 'OBSERVER_TARGET_POSITION'
      FRAME_-43918_SEC_OBSERVER   = 'EARTH'
      FRAME_-43918_SEC_TARGET     = 'SUN'
      FRAME_-43918_SEC_ABCORR     = 'NONE'

   \begintext


Sun Based Frames
========================================================================

   These dynamic frames are used for analyzing data in a reference
   frame tied to the dynamics of the Sun.


   Heliocentric Radial Tangential Normal (RTN) Frame ([3],[7])
   ---------------------------------------------------------------------

      The position of the spacecraft relative to the Sun is the primary
      vector: the X axis points from the Sun center to the spacecraft.

      The solar rotation axis is the secondary vector: the Z axis is
      the component of the solar north direction perpendicular to X.

      The Y axis is Z cross X, completing the right-handed reference
      frame.

      All vectors are geometric: no aberration corrections are used.

   \begindata

      FRAME_IMAP_RTN              = -43920
      FRAME_-43920_NAME           = 'IMAP_RTN'
      FRAME_-43920_CLASS          =  5
      FRAME_-43920_CLASS_ID       =  -43920
      FRAME_-43920_CENTER         =  10
      FRAME_-43920_RELATIVE       = 'J2000'
      FRAME_-43920_DEF_STYLE      = 'PARAMETERIZED'
      FRAME_-43920_FAMILY         = 'TWO-VECTOR'
      FRAME_-43920_PRI_AXIS       = 'X'
      FRAME_-43920_PRI_VECTOR_DEF = 'OBSERVER_TARGET_POSITION'
      FRAME_-43920_PRI_OBSERVER   = 'SUN'
      FRAME_-43920_PRI_TARGET     = 'IMAP'
      FRAME_-43920_PRI_ABCORR     = 'NONE'
      FRAME_-43920_PRI_FRAME      = 'IAU_SUN'
      FRAME_-43920_SEC_AXIS       = 'Z'
      FRAME_-43920_SEC_VECTOR_DEF = 'CONSTANT'
      FRAME_-43920_SEC_FRAME      = 'IAU_SUN'
      FRAME_-43920_SEC_SPEC       = 'RECTANGULAR'
      FRAME_-43920_SEC_VECTOR     = ( 0, 0, 1 )

   \begintext


   Heliocentric Inertial (HCI) Frame ([3],[5],[7])
   ---------------------------------------------------------------------

      Referred to as "Heliographic Inertial (HGI) frame at epoch J2000"
      in [3], but named as in [7] to avoid confusion with HGI of J1900.

      The X-Y Plane lies in the solar equator, +Z axis is parallel to
      the Sun's rotation vector.

      The solar rotation axis is the primary vector: the Z axis points
      in the solar north direction.

      The ascending node on the Earth ecliptic of J2000 of the solar
      equator forms the X axis. This is accomplished by using the +Z
      axis of the ecliptic of J2000 as the secondary vector and HCI +Y
      as the secondary axis.

      The Y axis is Z cross X, completing the right-handed reference
      frame.

   \begindata

      FRAME_IMAP_HCI              = -43921
      FRAME_-43921_NAME           = 'IMAP_HCI'
      FRAME_-43921_CLASS          = 5
      FRAME_-43921_CLASS_ID       = -43921
      FRAME_-43921_CENTER         = 10
      FRAME_-43921_RELATIVE       = 'J2000'
      FRAME_-43921_DEF_STYLE      = 'PARAMETERIZED'
      FRAME_-43921_FAMILY         = 'TWO-VECTOR'
      FRAME_-43921_PRI_AXIS       = 'Z'
      FRAME_-43921_PRI_VECTOR_DEF = 'CONSTANT'
      FRAME_-43921_PRI_FRAME      = 'IAU_SUN'
      FRAME_-43921_PRI_SPEC       = 'RECTANGULAR'
      FRAME_-43921_PRI_VECTOR     = ( 0, 0, 1 )
      FRAME_-43921_SEC_AXIS       = 'Y'
      FRAME_-43921_SEC_VECTOR_DEF = 'CONSTANT'
      FRAME_-43921_SEC_FRAME      = 'ECLIPJ2000'
      FRAME_-43921_SEC_SPEC       = 'RECTANGULAR'
      FRAME_-43921_SEC_VECTOR     = ( 0, 0, 1 )

   \begintext


   Heliocentric of Date (HCD) Frame ([3],[5],[7])
   ---------------------------------------------------------------------

      Referred to as "Heliographic Inertial (HGI) frame true to
      reference date" in [3], but named as in [7] without "inertial."

      The X-Y Plane lies in the solar equator, +Z axis is parallel to
      the Sun's rotation vector.

      The solar rotation axis is the primary vector: the Z axis points
      in the solar north direction.

      The ascending node on the Earth ecliptic of date of the solar
      equator forms the X axis. This is accomplished by using the +Z
      axis of the ecliptic of date as the secondary vector and HCD +Y
      as the secondary axis.

      The Y axis is Z cross X, completing the right-handed reference
      frame.

   \begindata

      FRAME_IMAP_HCD              = -43922
      FRAME_-43922_NAME           = 'IMAP_HCD'
      FRAME_-43922_CLASS          = 5
      FRAME_-43922_CLASS_ID       = -43922
      FRAME_-43922_CENTER         = 10
      FRAME_-43922_RELATIVE       = 'J2000'
      FRAME_-43922_DEF_STYLE      = 'PARAMETERIZED'
      FRAME_-43922_FAMILY         = 'TWO-VECTOR'
      FRAME_-43922_PRI_AXIS       = 'Z'
      FRAME_-43922_PRI_VECTOR_DEF = 'CONSTANT'
      FRAME_-43922_PRI_FRAME      = 'IAU_SUN'
      FRAME_-43922_PRI_SPEC       = 'RECTANGULAR'
      FRAME_-43922_PRI_VECTOR     = ( 0, 0, 1 )
      FRAME_-43922_SEC_AXIS       = 'Y'
      FRAME_-43922_SEC_VECTOR_DEF = 'CONSTANT'
      FRAME_-43922_SEC_FRAME      = 'IMAP_ECLIPDATE'
      FRAME_-43922_SEC_SPEC       = 'RECTANGULAR'
      FRAME_-43922_SEC_VECTOR     = ( 0, 0, 1 )

   \begintext


   Heliographic Coordinates (HGC) Frame ([3],[7])
   ---------------------------------------------------------------------

      Cartesian counterpart to the spherical coordinates defined in [3],
      "Heliographic Spherical (HGS) coordinate frame true to ref. date".

      Alias for SPICE IAU_SUN (Carrington heliographic coordinates)
      in which the frame rotates with the surface of the sun with a
      sidereal period of exactly 25.38 days.

      The Z axis is the solar rotation axis.

      The X axis is the intersection of the Carrington prime meridian
      and the heliographic equator.

      The Y axis is Z cross X, completing the right-handed reference
      frame.

   \begindata

      FRAME_IMAP_HGC          = -43923
      FRAME_-43923_NAME       = 'IMAP_HGC'
      FRAME_-43923_CLASS      = 4
      FRAME_-43923_CLASS_ID   = -43923
      FRAME_-43923_CENTER     = 10
      TKFRAME_-43923_RELATIVE = 'IAU_SUN'
      TKFRAME_-43923_SPEC     = 'MATRIX'
      TKFRAME_-43923_MATRIX   = ( 1  0  0
                                  0  1  0
                                  0  0  1 )
   \begintext


   Heliocentric Aries Ecliptic (HAE) Frame ([3],[7])
   ---------------------------------------------------------------------

      Alias for SPICE ECLIPJ2000.

      The Z axis is the normal to the mean ecliptic at J2000.

      The X axis is the unit vector from Earth to the first point of
      Aries at J2000.

      The Y axis is Z cross X, completing the right-handed reference
      frame.

   \begindata

      FRAME_IMAP_HAE          = -43924
      FRAME_-43924_NAME       = 'IMAP_HAE'
      FRAME_-43924_CLASS      = 4
      FRAME_-43924_CLASS_ID   = -43924
      FRAME_-43924_CENTER     = 10
      TKFRAME_-43924_RELATIVE = 'ECLIPJ2000'
      TKFRAME_-43924_SPEC     = 'MATRIX'
      TKFRAME_-43924_MATRIX   = ( 1  0  0
                                  0  1  0
                                  0  0  1 )
  \begintext


   Heliocentric Aries Ecliptic of Date (HAED) Frame ([3],[7])
   ---------------------------------------------------------------------

      Same orientation as IMAP_ECLIPDATE, but with Sun at the center
      instead of Earth.

      The Z axis is the normal to the mean ecliptic of date.

      The X axis is the unit vector from Earth to the first point of
      Aries of date.

      The Y axis is Z cross X, completing the right-handed reference
      frame.

   \begindata

      FRAME_IMAP_HAED         = -43925
      FRAME_-43925_NAME       = 'IMAP_HAED'
      FRAME_-43925_CLASS      = 4
      FRAME_-43925_CLASS_ID   = -43925
      FRAME_-43925_CENTER     = 10
      TKFRAME_-43925_RELATIVE = 'IMAP_ECLIPDATE'
      TKFRAME_-43925_SPEC     = 'MATRIX'
      TKFRAME_-43925_MATRIX   = ( 1  0  0
                                  0  1  0
                                  0  0  1 )
   \begintext


   Heliocentric Earth Ecliptic (HEE) Frame ([3],[7])
   ---------------------------------------------------------------------

      The position of the Earth relative to the Sun is the primary
      vector: the X axis points from the Sun to the Earth.

      The northern surface normal to the mean ecliptic of date is the
      secondary vector: the Z axis is the component of this vector
      orthogonal to the X axis.

      The Y axis is Z cross X, completing the right-handed reference
      frame.

      All vectors are geometric: no aberration corrections are used.

   \begindata

      FRAME_IMAP_HEE              = -43926
      FRAME_-43926_NAME           = 'IMAP_HEE'
      FRAME_-43926_CLASS          = 5
      FRAME_-43926_CLASS_ID       = -43926
      FRAME_-43926_CENTER         = 10
      FRAME_-43926_RELATIVE       = 'J2000'
      FRAME_-43926_DEF_STYLE      = 'PARAMETERIZED'
      FRAME_-43926_FAMILY         = 'TWO-VECTOR'
      FRAME_-43926_PRI_AXIS       = 'X'
      FRAME_-43926_PRI_VECTOR_DEF = 'OBSERVER_TARGET_POSITION'
      FRAME_-43926_PRI_OBSERVER   = 'SUN'
      FRAME_-43926_PRI_TARGET     = 'EARTH'
      FRAME_-43926_PRI_ABCORR     = 'NONE'
      FRAME_-43926_SEC_AXIS       = 'Z'
      FRAME_-43926_SEC_VECTOR_DEF = 'CONSTANT'
      FRAME_-43926_SEC_FRAME      = 'IMAP_ECLIPDATE'
      FRAME_-43926_SEC_SPEC       = 'RECTANGULAR'
      FRAME_-43926_SEC_VECTOR     = ( 0, 0, 1 )

   \begintext


   Heliospheric Ram Ecliptic (HRE) Frame ([3],[8],[9])
   ---------------------------------------------------------------------

      This is a heliocentric frame oriented with respect to the current,
      nominal ram direction of the Sun's motion relative to the local
      interstellar medium and the ecliptic plane, otherwise known as the
      heliospheric "nose" direction.

      The nose direction is the primary vector: the X axis points in the
      direction [-0.2477, -0.9647, 0.0896] in the ECLIPJ2000 (IMAP_HAE)
      frame.

      The northern surface normal to the mean ecliptic of J2000 is the
      secondary vector: the Z axis is the component of this vector
      orthogonal to the X axis.

      The Y axis is Z cross X, completing the right-handed reference
      frame.

   \begindata

      FRAME_IMAP_HRE               = -43927
      FRAME_-43927_NAME            = 'IMAP_HRE'
      FRAME_-43927_CLASS           = 5
      FRAME_-43927_CLASS_ID        = -43927
      FRAME_-43927_CENTER          = 10
      FRAME_-43927_RELATIVE        = 'J2000'
      FRAME_-43927_DEF_STYLE       = 'PARAMETERIZED'
      FRAME_-43927_FAMILY          = 'TWO-VECTOR'
      FRAME_-43927_PRI_AXIS        = 'X'
      FRAME_-43927_PRI_VECTOR_DEF  = 'CONSTANT'
      FRAME_-43927_PRI_FRAME       = 'ECLIPJ2000'
      FRAME_-43927_PRI_SPEC        = 'RECTANGULAR'
      FRAME_-43927_PRI_VECTOR      = ( -0.2477, -0.9647, 0.0896 )
      FRAME_-43927_SEC_AXIS        = 'Z'
      FRAME_-43927_SEC_VECTOR_DEF  = 'CONSTANT'
      FRAME_-43927_SEC_FRAME       = 'ECLIPJ2000'
      FRAME_-43927_SEC_SPEC        = 'RECTANGULAR'
      FRAME_-43927_SEC_VECTOR      = ( 0, 0, 1 )

      \begintext


   Heliospheric Nose Upfield (HNU) Frame ([3],[8],[9])
   ---------------------------------------------------------------------

      Heliocentric frame oriented with respect to the current nominal
      ram direction of the Sun's motion relative to the local
      interstellar medium and the current best estimate of the
      unperturbed magnetic field direction in the upstream local
      interstellar medium.

      The nominal upfield direction of the ISM B-field is the primary
      vector: the Z axis points in the direction
      [-0.5583, -0.6046, 0.5681] in the ECLIPJ2000 (IMAP_HAE) frame.

      The nose direction [-0.2477, -0.9647, 0.0896] in the ECLIPJ2000
      (IMAP_HAE) frame is the secondary vector: the X axis is the
      component of this vector orthogonal to the Z axis.

      The Y axis is Z cross X, completing the right-handed reference
      frame.

   \begindata

      FRAME_IMAP_HNU               = -43928
      FRAME_-43928_NAME            = 'IMAP_HNU'
      FRAME_-43928_CLASS           = 5
      FRAME_-43928_CLASS_ID        = -43928
      FRAME_-43928_CENTER          = 10
      FRAME_-43928_RELATIVE        = 'J2000'
      FRAME_-43928_DEF_STYLE       = 'PARAMETERIZED'
      FRAME_-43928_FAMILY          = 'TWO-VECTOR'
      FRAME_-43928_PRI_AXIS        = 'Z'
      FRAME_-43928_PRI_VECTOR_DEF  = 'CONSTANT'
      FRAME_-43928_PRI_FRAME       = 'ECLIPJ2000'
      FRAME_-43928_PRI_SPEC        = 'RECTANGULAR'
      FRAME_-43928_PRI_VECTOR      = ( -0.5583, -0.6046, 0.5681 )
      FRAME_-43928_SEC_AXIS        = 'X'
      FRAME_-43928_SEC_VECTOR_DEF  = 'CONSTANT'
      FRAME_-43928_SEC_FRAME       = 'ECLIPJ2000'
      FRAME_-43928_SEC_SPEC        = 'RECTANGULAR'
      FRAME_-43928_SEC_VECTOR      = ( -0.2477, -0.9647, 0.0896 )

   \begintext


   Galactic Coordinate System (GCS) Frame ([3])
   ---------------------------------------------------------------------

      Alias for SPICE galactic system II frame GALACTIC.

      The primary axis is the normal to the galactic equatorial plane:
      Z axis is this unit vector.

      The secondary axis is the vector from the Sun to the galatic
      center (represented by Sagittarious): X axis is the component of
      this vector orthogonal to the Z axis.

      The Y axis is Z cross X, completing the right-handed reference
      frame.

   \begindata

      FRAME_IMAP_GCS          = -43929
      FRAME_-43929_NAME       = 'IMAP_GCS'
      FRAME_-43929_CLASS      = 4
      FRAME_-43929_CLASS_ID   = -43929
      FRAME_-43929_CENTER     = 10
      TKFRAME_-43929_RELATIVE = 'GALACTIC'
      TKFRAME_-43929_SPEC     = 'MATRIX'
      TKFRAME_-43929_MATRIX   = ( 1  0  0
                                  0  1  0
                                  0  0  1 )
  \begintext


END OF FILE