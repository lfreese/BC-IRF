PROGRAM calculate_forcing_and_climate_response
!
! PURPOSE: calculate global annual-mean BC radiative forcings [W m-2] and
! the estimated global annual-mean temperature response [K] given the
! following information
! - emission area (lat,lon limits)
! - annual-mean emission rate OR time series of monthly emission rate
! These data are provided in the file "input_data.asc"
!
! Compiling (for Intel compiler)
!  module load netcdf-fortran
!  ifort calculate_forcing_and_climate_response.f90 -lnetcdff
!
! Petri Räisänen (FMI, 22 Feb 2022 / 8 April 2022)
!
!************************************************************************

  USE netcdf

  IMPLICIT NONE

  INTEGER, PARAMETER :: dp=SELECTED_REAL_KIND(12,307)

  INTEGER, PARAMETER :: nlon=12   ! number of longitude zones for BC emissions
  INTEGER, PARAMETER :: nlat=16   ! number of latitude zones fBC emissions
  INTEGER, PARAMETER :: nseas=5   ! number of seasons for BC emissions
                                  ! (ANN, DJF, MAM, JJA, SON)
  INTEGER, PARAMETER :: nmonth=12 ! Number of months per year

  INTEGER, PARAMETER :: nrecl=1 ! GrADS record length (1 for Intel, 4 for Cray)

  REAL(dp), PARAMETER :: UNDEF=-9999._dp
  REAL(dp), PARAMETER :: pi=4*ATAN(1._dp)
  REAL(dp), PARAMETER :: rearth=6.37122e6_dp ! Radius of Earth
  REAL(dp), PARAMETER :: area_earth=4*pi*rearth**2

!***************************************************************
! Variables read from the NetCDF file
! Coordinate limits for the BC emission regions
  REAL(dp), DIMENSION(nlon) :: longitude_west, longitude_east
  REAL(dp), DIMENSION(nlat) :: latitude_south, latitude_north

! Radiative forcings normalized by BC emissions [TJ kg^-1]
  REAL(dp), DIMENSION(nlon,nlat,nseas) :: dirsf_toa, snowsf_toa

! Estimated temperature response normalized by BC emissions [K (kg s^-1)^-1]
  REAL(dp), DIMENSION(nlon,nlat,nseas) :: dt_norm_drf, dt_norm_snowrf

!*********************************
! Input data provided by the user
!*********************************

  REAL(dp) :: lon_w, lon_e, lat_s, lat_n  ! coordinate limits
  REAL(dp), DIMENSION(nmonth) :: emissions_monthly
  REAL(dp) :: emissions_annual
  LOGICAL :: lmonthly_emissions

! Seasonal-mean emissions (1 = ANN, 2 = DJF, 3 = MAM, 4 = JJA, 5 = SON)
  REAL(dp), DIMENSION(nseas) :: emissions_seasonal

!****************************
! Output from this program
!****************************

! Global annual-mean radiative forcings in [W m-2]
  REAL(dp) :: drf_toa_out, snowrf_toa_out
! Global-mean temperature responses in [K] 
  REAL(dp) :: dt_drf_out, dt_snowrf_out

!*********************************************************
! Other variables

  INTEGER, DIMENSION(nmonth) :: ndays_per_month
  INTEGER, DIMENSION(nmonth,nseas) :: months_in_season

  REAL(dp), DIMENSION(nmonth) :: sec_per_month  
  REAL(dp), DIMENSION(nseas) :: sec_per_season  
  REAL(dp) :: sec_per_year

! The overlapping part of the user-defined region with each of the predefined
! emission regions (in m^2)
  REAL(dp), DIMENSION(nlon,nlat) :: overlap_area

  REAL(dp), DIMENSION(nlat) :: sin_south, sin_north
  REAL(dp) :: sin_s, sin_n, delta_lon, delta_sinlat

  INTEGER :: ilon, ilat, iseas, imonth
  LOGICAL:: ok

! Input file for reading in the radiative forcings and climate responses
! (normalized by emissions, as a function of emission latitude, longitude
! and season)
  CHARACTER (LEN=100) :: infile  

!*************************************************************************
! Days per month as in NorESM (no leap years)

  ndays_per_month = (/31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31/)
  sec_per_month(:) = 86400._dp*ndays_per_month(:)

! Months included in each season 
! (1 = ANN, 2 = DJF, 3 = MAM, 4 = JJA, 5 = SON)
  
  months_in_season(1:12,1) = (/1,1,1,1,1,1,1,1,1,1,1,1/)
  months_in_season(1:12,2) = (/1,1,0,0,0,0,0,0,0,0,0,1/)
  months_in_season(1:12,3) = (/0,0,1,1,1,0,0,0,0,0,0,0/)
  months_in_season(1:12,4) = (/0,0,0,0,0,1,1,1,0,0,0,0/)
  months_in_season(1:12,5) = (/0,0,0,0,0,0,0,0,1,1,1,0/)

  DO iseas=1,5
    sec_per_season(iseas)=SUM(months_in_season(:,iseas)*sec_per_month(:))
  ENDDO
  sec_per_year = sec_per_season(1)

!************************************************************************

  infile="BC_forcing_and_climate_response_normalized_by_emissions.nc"

  CALL read_netcdf_data(nlon,nlat,nseas,infile, &
    longitude_west, longitude_east, latitude_south, latitude_north, &
    dirsf_toa, snowsf_toa, dt_norm_drf, dt_norm_snowrf)

! Read in input data provided by the user in the file "input_data.asc" 

  CALL read_input_data(nmonth,lon_w, lon_e, lat_s, lat_n, lmonthly_emissions, &
            emissions_monthly, emissions_annual)  

!*****************************************************************************
! Determine seasonal-mean BC emissions. For convenience, the whole year
! is defined as the first "season". 
! (1 = ANN, 2 = DJF, 3 = MAM, 4 = JJA, 5 = SON)
! If the emissions are defined at monthly resolution, then only the
! seasonal indices 2-5 are used. And if they are defined at annual resolution,
! only the index 1 is used. This trick avoids the use of separate branches for 
! lmonthly_emissions=.T. and lmonthly_emissions=.F. later in the program.
!***************************************************************************

   IF (lmonthly_emissions) THEN
     emissions_seasonal(1)=0.
     DO iseas=2,5
       emissions_seasonal(iseas) = &
         SUM(months_in_season(:,iseas)*sec_per_month(:)*emissions_monthly(:)) &
        /SUM(months_in_season(:,iseas)*sec_per_month(:))
     ENDDO
   ELSE
     emissions_seasonal(1)=emissions_annual
     emissions_seasonal(2:5)=0.
   END IF

! Determine the how large part of the user-defined region (in m**2) overlaps
! with each of the predefined emission regions for which the radiative
! forcings and climate responses are available

! Sine of latitude, predefined emissions regions
  sin_south(:) = SIN(pi/180._dp * latitude_south(:))
  sin_north(:) = SIN(pi/180._dp * latitude_north(:))

! Sine of latitude, user-defined emission regions
  sin_s = SIN(pi/180._dp * lat_s)
  sin_n = SIN(pi/180._dp * lat_n)

  DO ilon=1,nlon
    delta_lon=common_lon(lon_w,lon_e,longitude_west(ilon),longitude_east(ilon))
    DO ilat=1,nlat
      delta_sinlat=MIN(sin_n, sin_north(ilat))-MAX(sin_s,sin_south(ilat))
      delta_sinlat=MAX(delta_sinlat,0._dp)
      overlap_area(ilon,ilat) = (delta_lon/360._dp)*(delta_sinlat/2) * area_earth
    ENDDO
  ENDDO

! Calculate global-annual mean radiative forcings (in W m-2)

  drf_toa_out = global_annual_mean_rf(nlon,nlat,nseas,dirsf_toa, &
                emissions_seasonal, overlap_area, &
                sec_per_season, sec_per_year, area_earth)

  snowrf_toa_out = global_annual_mean_rf(nlon,nlat,nseas,snowsf_toa, &
                   emissions_seasonal, overlap_area, &
                   sec_per_season, sec_per_year, area_earth)

! Calculate global-annual mean temperature response (in K)

  dt_drf_out = global_annual_mean_dt(nlon,nlat,nseas,dt_norm_drf, &
                emissions_seasonal, overlap_area, &
                sec_per_season, sec_per_year, area_earth)

  dt_snowrf_out = global_annual_mean_dt(nlon,nlat,nseas,dt_norm_snowrf, &
                emissions_seasonal, overlap_area, &
                sec_per_season, sec_per_year, area_earth)

!***********
! OUTPUT
!***********

  WRITE(*,*)
  WRITE(*,*) "Estimated global annual-mean radiative forcings [W m-2]"
  WRITE(*,'(" DRF_TOA   ", ES13.5)') drf_toa_out
  WRITE(*,'(" SNOWRF_TOA", ES13.5)') snowrf_toa_out
  WRITE(*,'(" SUM_RF_TOA", ES13.5)') drf_toa_out+snowrf_toa_out
  WRITE(*,*)
  WRITE(*,*) "Estimated global annual-mean temperature responses [K]"
  WRITE(*,'(" DT_DRF    ", ES13.5)') dt_drf_out
  WRITE(*,'(" DT_SNOWRF ", ES13.5)') dt_snowrf_out
  WRITE(*,'(" DT_SUM    ", ES13.5)') dt_drf_out + dt_snowrf_out

!*********
CONTAINS
!*********

!**************************************************************
  FUNCTION global_annual_mean_RF(nlon,nlat,nseas,rf, emissions, overlap_area, &
           sec_per_season, sec_per_year, area_earth) RESULT(avg_rf)
       
! Compute global annual-mean radiative forcing (in W m-2)

    IMPLICIT NONE
    INTEGER, INTENT(in) :: nlon  
    INTEGER, INTENT(in) :: nlat
    INTEGER, INTENT(in) :: nseas ! number of seasons (ANN, DJF, MAM, JJA, SON)

! global-annual mean radiative forcing associated with BC emissions in 
! a given region and season, in units [TJ kg^{-1}]
    REAL(dp), DIMENSION(nlon,nlat,nseas), INTENT(in) :: rf 
! Seasonal-mean BC emissions in the user-defined region [kg m^{-2} s^{-1}]
    REAL(dp), DIMENSION(nseas), INTENT(in) :: emissions
! The overlapping part of the user-defined region with each of the predefined
! emission regions (in m^2) 
    REAL(dp), DIMENSION(nlon,nlat), INTENT(in) :: overlap_area

! Season lengths in seconds 
    REAL(dp), DIMENSION(nseas), INTENT(in) :: sec_per_season
! Year length in seconds
    REAL(dp), INTENT(in) :: sec_per_year
! Area of earth [m^2]
    REAL(dp), INTENT(in) :: area_earth

! Local variables
    REAL(dp) :: avg_rf, summa
    INTEGER :: ilon, ilat, iseas

! Calculate radiative forcing first in units of [TJ]
    summa=0.
    DO ilon=1,nlon
      DO ilat=1,nlat
        DO iseas=1,nseas
          summa = summa + overlap_area(ilon,ilat)*emissions(iseas) &
                        * rf(ilon,ilat,iseas) * sec_per_season(iseas)
        ENDDO
      ENDDO
    ENDDO
! Convert to W m^-2
    avg_rf = 1.E+12 * summa / (area_earth*sec_per_year) 

    RETURN
  END FUNCTION global_annual_mean_RF
!****************************************************************************
  FUNCTION global_annual_mean_dt(nlon,nlat,nseas,dt, emissions, overlap_area, &
           sec_per_season, sec_per_year, area_earth) RESULT(avg_dt)
       
! Compute global annual-mean temperature response (in K)

    IMPLICIT NONE
    INTEGER, INTENT(in) :: nlon  
    INTEGER, INTENT(in) :: nlat
    INTEGER, INTENT(in) :: nseas ! number of seasons (ANN, DJF, MAM, JJA, SON)

! global-annual mean temperature response associated with BC emissions
! a given region and season, in units [K (kg^{s-1})^{-1}]
    REAL(dp), DIMENSION(nlon,nlat,nseas), INTENT(in) :: dt 
! Seasonal-mean BC emissions in the user-defined region [kg m^{-2} s^{-1}]
    REAL(dp), DIMENSION(nseas), INTENT(in) :: emissions
! The overlapping part of the user-defined region with each of the predefined
! emission regions (in m^2) 
    REAL(dp), DIMENSION(nlon,nlat), INTENT(in) :: overlap_area

! Season lengths in seconds 
    REAL(dp), DIMENSION(nseas), INTENT(in) :: sec_per_season
! Year length in seconds
    REAL(dp), INTENT(in) :: sec_per_year
! Area of earth [m^2]
    REAL(dp), INTENT(in) :: area_earth

! Local variables
    REAL(dp) :: avg_dt, summa
    INTEGER :: ilon, ilat, iseas

! Calculate global-mean temperature response
    summa=0.
    DO ilon=1,nlon
      DO ilat=1,nlat
        DO iseas=1,nseas
          summa = summa + overlap_area(ilon,ilat)*emissions(iseas) &
                        * dt(ilon,ilat,iseas) * sec_per_season(iseas)
        ENDDO
      ENDDO
    ENDDO
! Normalize by the length of the year
    avg_dt = summa / sec_per_year

    RETURN
  END FUNCTION global_annual_mean_dt
!****************************************************************************
!****************************************************************************
  FUNCTION regional_annual_mean_dt(nlon,nlat,nseas,dt, emissions, overlap_area, &
           sec_per_season, sec_per_year, area_earth) RESULT(avg_dt)
       
! Compute regional annual-mean temperature response (in K)

    IMPLICIT NONE
    INTEGER, INTENT(in) :: nlon  
    INTEGER, INTENT(in) :: nlat
    INTEGER, INTENT(in) :: nseas ! number of seasons (ANN, DJF, MAM, JJA, SON)

! regional-annual mean temperature response associated with BC emissions
! a given region and season, in units [K (kg^{s-1})^{-1}]
    REAL(dp), DIMENSION(nlon,nlat,nseas), INTENT(in) :: dt 
! Seasonal-mean BC emissions in the user-defined region [kg m^{-2} s^{-1}]
    REAL(dp), DIMENSION(nseas), INTENT(in) :: emissions
! The overlapping part of the user-defined region with each of the predefined
! emission regions (in m^2) 
    REAL(dp), DIMENSION(nlon,nlat), INTENT(in) :: overlap_area

! Season lengths in seconds 
    REAL(dp), DIMENSION(nseas), INTENT(in) :: sec_per_season
! Year length in seconds
    REAL(dp), INTENT(in) :: sec_per_year
! Area of earth [m^2]
    REAL(dp), INTENT(in) :: area_earth

! Local variables
    REAL(dp) :: avg_dt, summa
    INTEGER :: ilon, ilat, iseas

! Calculate global-mean temperature response
    summa=0.
    DO ilon=1,nlon
      DO ilat=1,nlat
        DO iseas=1,nseas
          summa = summa + overlap_area(ilon,ilat)*emissions(iseas) &
                        * dt(ilon,ilat,iseas) * sec_per_season(iseas)
        ENDDO
      ENDDO
    ENDDO
! Normalize by the length of the year
    avg_dt = summa / sec_per_year

    RETURN
  END FUNCTION regional_annual_mean_dt
!****************************************************************************
!****************************************************************************

  SUBROUTINE read_netcdf_data(nlon,nlat,nseas,infile, &
       longitude_west, longitude_east, latitude_south, latitude_north, &
       dirsf_toa, snowsf_toa, dt_norm_drf, dt_norm_snowrf)

! PURPOSE: Read in the precomputed radiative forcings and temperature responses
! as a function of BC emission region and season

    IMPLICIT NONE

! Dimensions
    INTEGER, INTENT(in) :: nlon  ! number of longitude zones for BC emissions
    INTEGER, INTENT(in) :: nlat  ! number of latitude zones fBC emissions
    INTEGER, INTENT(in) :: nseas ! number of seasons for BC emissions
                                 ! (ANN, DJF, MAM, JJA, SON)

    CHARACTER(LEN=*) :: infile ! NetCDF input file

! Coordinate limits for the BC emission regions
    REAL(dp), DIMENSION(nlon), INTENT(out) :: longitude_west, longitude_east
    REAL(dp), DIMENSION(nlat), INTENT(out) :: latitude_south, latitude_north
 
! Global-mean radiative forcings normalized by emissions [TJ kg^-1]
    REAL(dp), DIMENSION(nlon,nlat,nseas), INTENT(out) :: dirsf_toa, snowsf_toa

! Global-mean temperature responses normalized by emission rate [K (kg s^-1)^-1]
    REAL(dp), DIMENSION(nlon,nlat,nseas), INTENT(out) :: &
                                                 dt_norm_drf, dt_norm_snowrf

! NetCDF stuff
    INTEGER, PARAMETER :: nvarmax = 20 ! Maximum number of variables
    INTEGER :: ncid, ndims, nvars, nglobalatts, unlimdimid, iret, ivar
    CHARACTER (LEN=40), DIMENSION(nvarmax) :: varname

!********************************************************************
! Open the existing NetCDF file
    iret = nf90_open(path = infile, mode = nf90_nowrite, ncid = ncid)

! Find out what it contains
    
    iret = nf90_inquire(ncid, ndims, nvars, nglobalatts, unlimdimid)

! Get variable names 

    DO ivar=1,nvars
      iret = nf90_inquire_variable(ncid,ivar,name=varname(ivar))
!     write(*,'(3I6,1X,A)') iret,ncid,ivar,TRIM(varname(ivar))
    ENDDO 

! Read in the variables. This makes use of the fact that the variable names
! and dimensions are known beforehand

    DO ivar=1,nvars
      iret = nf90_inquire_variable(ncid,ivar,name=varname(ivar))
      SELECT CASE(TRIM(varname(ivar)))
        CASE("longitude_west")
          iret = nf90_get_var(ncid, ivar, longitude_west(1:nlon))
        CASE("longitude_east")
          iret = nf90_get_var(ncid, ivar, longitude_east(1:nlon))
        CASE("latitude_south")
          iret = nf90_get_var(ncid, ivar, latitude_south(1:nlat))
        CASE("latitude_north")
          iret = nf90_get_var(ncid, ivar, latitude_north(1:nlat))

! Radiative forcings normalized by BC emissions [TJ kg^-1]
        CASE("dirsf_toa")
          iret = nf90_get_var(ncid, ivar, dirsf_toa(1:nlon,1:nlat,1:nseas))
        CASE("snowsf_toa")
          iret = nf90_get_var(ncid, ivar, snowsf_toa(1:nlon,1:nlat,1:nseas)) 

! Estimated temperature response normalized by BC emission rate [K (kg s^-1)^-1]
       CASE("dt_norm_drf")
          iret = nf90_get_var(ncid, ivar, dt_norm_drf(1:nlon,1:nlat,1:nseas))
       CASE("dt_norm_snowrf")
          iret = nf90_get_var(ncid, ivar, dt_norm_snowrf(1:nlon,1:nlat,1:nseas))
      END SELECT
    ENDDO
    iret = nf90_close(ncid)

     RETURN
  END SUBROUTINE read_netcdf_data
!*****************************************************************
  SUBROUTINE read_input_data(nmonth, lon_w, lon_e, lat_s, lat_n, &
       lmonthly_emissions, emissions_monthly, emissions_annual)  

! PURPOSE: Read the following input data from the file input_data.asc
!
! - LON_W: longitude of the western boundary of the region considered (degrees)
! - LON_E: longitude of the eastern boundary of the region considered (degrees)
! - LAT_S: latitude of the southern boundary of the region considered (degrees)
! - LAT_N: latitude of the northern boundary of the region considered (degrees)
! - LMONTHLY_EMISSIONS: whether monthly (TRUE) or annual (FALSE) emission
!                       data is provided
! - EMISSIONS_MONTHLY: monthly BC emissions [kg m^{-2} s^{-1}]
!                      (defined if LMONTHLY_EMISSIONS = .TRUE.)
! - EMISSIONS_ANNUAL: annual BC emissions [kg m^{-2} s^{-1}]
!                      (defined if LMONTHLY_EMISSIONS = .FALSE.)

    IMPLICIT NONE
    INTEGER, INTENT(in) :: nmonth ! number of months per year
    REAL(dp), INTENT(out) :: lon_w, lon_e, lat_s, lat_n
    REAL(dp), DIMENSION(nmonth), INTENT(out) :: emissions_monthly
    REAL(dp), INTENT(out) :: emissions_annual
    LOGICAL, INTENT(out) :: lmonthly_emissions

    INTEGER, PARAMETER :: nval=99 ! Maximum number of values per line
                                  ! (arbitrarily large)
    INTEGER :: ind, i, n, nval_read
    REAL(dp), DIMENSION(nmonth) :: emissions
    REAL(dp), DIMENSION(nval) :: values
    LOGICAL:: ok, ok_all

    CHARACTER(LEN=160) :: rivi
    CHARACTER(LEN=30) :: string_tmp

! Initializate input data to undefined values

    lon_w = UNDEF
    lon_e = UNDEF
    lat_s = UNDEF
    lat_n = UNDEF
    lmonthly_emissions=.FALSE.
    emissions_monthly(:) = UNDEF
    emissions_annual = UNDEF

    ok_all = .TRUE.

! Open then input data file

    OPEN(11,FILE="input_data.asc", STATUS='OLD')

! Read in the longitude limits     
 10 CONTINUE
    READ(11,'(A160)',END=99,ERR=99) rivi      
    ind=INDEX(rivi,"LONGITUDE_LIMITS")
    IF (ind==0) GOTO 10
    READ(rivi,*,ERR=96) string_tmp,lon_w,lon_e

! Read in the latitude limits         
 20 CONTINUE
    READ(11,'(A160)',END=99,ERR=99) rivi      
    ind=INDEX(rivi,"LATITUDE_LIMITS")
    IF (ind==0) GOTO 20
    READ(rivi,*,ERR=97) string_tmp,lat_s,lat_n    

! Read in the emission rates
 30 CONTINUE
    READ(11,'(A160)',END=99,ERR=99) rivi      
    ind=INDEX(rivi,"EMISSION_RATE")
    IF (ind==0) GOTO 30
 
    n=0
 40 CONTINUE
    rivi=" "  
    nval_read=0  
    READ(11,'(A160)',ERR=99,end=50) rivi
    IF (TRIM(rivi)/=" ") THEN
      CALL read_line(nval,rivi,values,nval_read)
      DO i=1,nval_read
        n=n+1
        IF (n <= nmonth) emissions(n)=values(i)         
      ENDDO
      GOTO 40
    END IF

 50 CONTINUE

! If needed, modify the longitudes (by +-360 deg) so that LON_E > LON_W 
! and check that the values are OK.
    CALL modify_and_check_lon(lon_w,lon_e,ok)

    IF (.NOT.ok) THEN
      ok_all = .FALSE.
      WRITE(*,'(" Longitude limits not OK: ",2F11.4)') lon_w, lon_e
    END IF  
  
! 4) Check the latitude limits
    IF (.NOT.check_lat(lat_s,lat_n)) THEN
      ok_all = .FALSE.
      WRITE(*,'(" Latitude limits not OK: ",2F11.4)') lat_s, lat_n
    END IF

! 5) Check the number of emissions values
    IF ((n/=1).AND.(n/=12)) THEN
      ok_all = .FALSE.
      WRITE(*,'("Number of emission values given =",I3)') n
      WRITE(*,'("Give either a single annual-mean value or 12 monthly-mean values")')
      STOP
    ELSE 
      IF (n==12) THEN
        lmonthly_emissions=.TRUE.
        emissions_monthly(1:n) = emissions(1:n)
      ELSE IF (n==1) THEN
        lmonthly_emissions=.FALSE.
        emissions_annual = emissions(1)
      END IF
    END IF

! Check that the emission values are non-negative
    IF (lmonthly_emissions) THEN
      IF (MINVAL(emissions_monthly(1:nmonth)) < 0.) THEN
        ok_all = .FALSE.
        WRITE(*,*) " All 12 monthly emission rates must be non-negative!"
        WRITE(*,'(6ES12.4)') emissions_monthly(1:nmonth)
      END IF
    ELSE
      IF (emissions_annual < 0.) THEN
        ok_all = .FALSE.
        WRITE(*,*) " The annual-mean emission rate must be non-negative!"
        WRITE(*,'(6ES12.4)') emissions_annual
      END IF 
    END IF

    IF (.NOT.ok_all) STOP

! Write input data
    WRITE(*,*)
    WRITE(*,'(" Longitude limits:",2F11.4)') lon_w, lon_e
    WRITE(*,'(" Latitude limits: ",2F11.4)') lat_s, lat_n
    IF (lmonthly_emissions) THEN
      WRITE(*,*)
      WRITE(*,'(" Monthly-mean BC emissions [kg m-2 s-1]:")')
      WRITE(*,'(6ES12.4)') emissions_monthly(1:nmonth)
    ELSE
      WRITE(*,'(" Annual-mean BC emissions [kg m-2 s-1]: ",ES12.4)') &
           emissions_annual     
    END IF 

    RETURN

 96 WRITE(*,'(" Error in reading LONGITUDE_LIMITS")')
    STOP
 97 WRITE(*,'(" Error in reading LATITUDE_LIMITS")')
    STOP
 98 WRITE(*,'(" Error in reading EMISSIONS")')
    STOP
 99 WRITE(*,'(" Error in reading input_data.asc")')
    STOP

  END SUBROUTINE read_input_data
!***********************************************************************
  SUBROUTINE read_line(nval,rivi,values,nval_read)
! Try to read numerical values from a line
    IMPLICIT NONE
    INTEGER, INTENT(in) :: nval            ! Dimension of array values
    CHARACTER(LEN=*), INTENT(in) :: rivi   ! A row of text

    INTEGER, INTENT(out) :: nval_read   ! Number of values actually read
    REAL(dp), DIMENSION(nval), INTENT(out) :: values

    INTEGER :: length, length2,i,j,n,iostat
    INTEGER, DIMENSION(nval+1) :: indsep

    CHARACTER(LEN=10000) :: rivi2
    CHARACTER(LEN=100) :: word
    CHARACTER :: char1

    LOGICAL :: lsep,lprevious_is_space, lprevious_is_sep,ok
 
! Initialize to undefined

    values(:)=UNDEF

! First, make a copy of the row, in which extra spaces are omitted.
    j=0
    lprevious_is_space=.TRUE.
    length=LEN_TRIM(rivi)
    rivi2=""

    DO i=1,length
      char1=rivi(i:i)
      IF ((char1/=" ").OR.(.NOT.lprevious_is_space)) THEN
        j=j+1
        rivi2(j:j)=char1
        IF (char1==" ") THEN 
          lprevious_is_space=.TRUE. 
        ELSE
          lprevious_is_space=.FALSE.
        END IF
      END IF 
    ENDDO
    length2=j

! Second, find all value separators in the line: these include
! - commas (,) and semicolons (;)
! - spaces ( ), when the previous character is not a separator

    n=1
    indsep(1)=0
    lprevious_is_sep=.TRUE.
    DO i=1,length2
      lsep=.FALSE.
      char1=rivi2(i:i)
      SELECT CASE(char1)
! Commas (,) and semicolons(;) are always separators
        CASE(",",";")
          lsep=.TRUE.
! Spaces are separators, if not preceeded by a separator
        CASE(" ")
          IF (.NOT.lprevious_is_sep) lsep=.TRUE.
      END SELECT
      IF (lsep) THEN
        n=n+1
        indsep(n)=i
      END IF    
      lprevious_is_sep=lsep
    ENDDO
    indsep(n+1)=length2+1
! Extra check: ignore the last separator, if it ends the line.
    IF (indsep(n)==length2) n=n-1

! Finally, read in the values from the substrings between separators
     DO i=1,n
       word=rivi2(indsep(i)+1:indsep(i+1)-1)//" "
       READ(word,*,IOSTAT=iostat) values(i)
       IF (iostat /=0) values(i)=UNDEF
     ENDDO
     nval_read=n

    RETURN
  END SUBROUTINE read_line
!*************************************************************
  FUNCTION check_lat(rlat1,rlat2) RESULT(ok)
! Checks that 
!  - rlat1 and rlat2 are between -90 and 90 deg!
!  - rlat2 >= rlat1
   
    IMPLICIT NONE
    REAL(dp), INTENT(in) :: rlat1,rlat2

    LOGICAL :: ok

    ok =.TRUE.
    IF (rlat2 - rlat1 <   0.) ok =.FALSE.
    IF (rlat1 < -90._dp) ok =.FALSE.
    IF (rlat2 >  90._dp) ok =.FALSE.

  END FUNCTION check_lat
!*************************************************************************
  SUBROUTINE modify_and_check_lon(rlon1, rlon2,ok)
! Inputs a pair of longitudes RLON1 and RLON2 given by the user,
! possibly modifies them to avoid problems with the dateline and Greenwich
! median, and check that the values are OK.
!
    IMPLICIT NONE
    REAL(dp), INTENT(inout) :: rlon1, rlon2
    LOGICAL, INTENT(out) :: ok

! 1) It is assumed that RLON2 is located east of RLON1. However,
! we may have the following cases in which RLON2 < RLON1
! - the user gives the longitudes between -180 and 180 deg
!   and the longitude band of interest covers the dateline 180W/E
! - the user gives the longitudes between 0 and 360 deg 
!   and the longitude band of interest covers the Greenwich meridian 0 W/E
! In such cases, modify the longitudes so that RLON2 > RLON1
!

    IF (rlon2 < rlon1) THEN
      IF (rlon2 < 0.) THEN
        rlon2=rlon2+360.
      ELSE  IF (rlon1 > 0.) THEN 
        rlon1=rlon1-360.         
      END IF
    END IF
! 2) Check that, after the previous step, 
!  - rlon1 and rlon2 are between -360 and 360 deg!
!  - 0.  <= rlon2-rlon1 <= 360. 

    ok =.TRUE.
    IF (rlon2 - rlon1 <   0.) ok =.FALSE.
    IF (rlon2 - rlon1 > 360.) ok =.FALSE.
    IF (rlon1 < -360.) ok =.FALSE.
    IF (rlon2 >  360.) ok =.FALSE.

    RETURN
  END SUBROUTINE modify_and_check_lon
!**********************************************************************
  FUNCTION common_lon (xlon1, xlon2, ylon1, ylon2) RESULT(delta_lon)

    REAL(dp), INTENT(in) :: xlon1 ! lower longitude limit for the first grid
    REAL(dp), INTENT(in) :: xlon2 ! upper longitude limit for the first grid

    REAL(dp), INTENT(in) :: ylon1 ! lower longitude limit for the second grid
    REAL(dp), INTENT(in) :: ylon2 ! upper longitude limit for the second grid

    REAL(dp) :: delta_lon

    REAL(dp) :: xlon1a, xlon1b, xlon2a, xlon2b, &
                ylon1a, ylon1b, ylon2a, ylon2b, &
                delta1, delta2, delta3, delta4

    LOGICAL :: ok

! Check that the input values are adequate

! Split both the first grid (xlon1...xlon2) and the second grid
! (ylon1...ylon2) in two parts that lie between 0 and 360 deg

    CALL split_lon(xlon1,xlon2, xlon1a, xlon2a, xlon1b, xlon2b)
    CALL split_lon(ylon1,ylon2, ylon1a, ylon2a, ylon1b, ylon2b)

    delta1 = MAX(MIN(xlon2a,ylon2a)-MAX(xlon1a,ylon1a),0.)
    delta2 = MAX(MIN(xlon2a,ylon2b)-MAX(xlon1a,ylon1b),0.)
    delta3 = MAX(MIN(xlon2b,ylon2a)-MAX(xlon1b,ylon1a),0.)
    delta4 = MAX(MIN(xlon2b,ylon2b)-MAX(xlon1b,ylon1b),0.)

    delta_lon = delta1+delta2+delta3+delta4
  END FUNCTION common_lon
!*******************************************************************
  SUBROUTINE split_lon(rlon1,rlon2, rlon1a, rlon2a, rlon1b, rlon2b)

! Assumes that rlon1 and rlon2 follow the rules defined in "check_lon"
! If "rlon1" is negative, then the region is split in two parts 
! in the range 0 ... 360 deg 

    IMPLICIT NONE
    
    REAL(dp), INTENT(in)  :: rlon1, rlon2
    REAL(dp), INTENT(out) :: rlon1a, rlon1b, rlon2a, rlon2b

    IF (rlon1 < 0.) THEN
      rlon1a = 0.
      rlon2a = MAX(rlon2,0.)
      rlon1b = rlon1 + 360.  
      rlon2b = MIN(rlon2,0.) +360. 
    ELSE
      rlon1a = rlon1
      rlon2a = rlon2
      rlon1b = 0.
      rlon2b = 0.
    END IF 

    RETURN
  END SUBROUTINE split_lon

!*********************************************************************

END PROGRAM calculate_forcing_and_climate_response
