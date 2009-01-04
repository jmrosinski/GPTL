subroutine gptlprocess_namelist (filename, unitno, outret)
!
! $Id: gptlprocess_namelist.f90,v 1.6 2009-01-04 21:14:41 rosinski Exp $
!
! Author: Jim Rosinski
!
! Utility subroutine processes namelist group &gptlnl and makes appropriate 
! calls to gptlsetoption() and/or gptlsetutr().
!
! To follow GPTL conventions this should be a function not a subroutine.
! But 'include ./gptl.inc' and then setting function gptlprocess_namelist
! to a return value causes compiler to barf because the function is declared 
! 'external' in the header. So set return value in output arg 'outret' instead.
!
  implicit none

  character(len=*), intent(in) :: filename  ! Input file containing &gptlnl
  integer, intent(in) :: unitno             ! Fortran unit number to open
  integer, intent(out) :: outret            ! Output return code

  include './gptl.inc'

  integer :: j    ! loop index
  integer :: ios  ! status return from file open
  integer :: code ! event code
  integer :: ret  ! return value
  integer, parameter :: maxevents = 99 ! space to hold more than enough events

! Default values for namelist variables

  logical, parameter :: def_wall            = .true.
  logical, parameter :: def_cpu             = .false.
  logical, parameter :: def_abort_on_error  = .false.
  logical, parameter :: def_overhead        = .true.
  integer, parameter :: def_depthlimit      = 99999    ! Effectively unlimited
  logical, parameter :: def_verbose         = .false.
  logical, parameter :: def_narrowprint     = .true.
  logical, parameter :: def_percent         = .false.
  logical, parameter :: def_persec          = .true.
  logical, parameter :: def_multiplex       = .false.
  logical, parameter :: def_dopr_preamble   = .true.
  logical, parameter :: def_dopr_threadsort = .true.
  logical, parameter :: def_dopr_multparent = .true.
  logical, parameter :: def_dopr_collision  = .true.
  character(len=16), parameter :: def_print_method = 'most_frequent   '
  character(len=16), parameter :: def_utr          = 'gettimeofday    '

! Namelist values: initialize to defaults

  logical :: wall            = def_wall
  logical :: cpu             = def_cpu
  logical :: abort_on_error  = def_abort_on_error
  logical :: overhead        = def_overhead
  integer :: depthlimit      = def_depthlimit
  logical :: verbose         = def_verbose
  logical :: narrowprint     = def_narrowprint
  logical :: percent         = def_percent
  logical :: persec          = def_persec
  logical :: multiplex       = def_multiplex
  logical :: dopr_preamble   = def_dopr_preamble
  logical :: dopr_threadsort = def_dopr_threadsort
  logical :: dopr_multparent = def_dopr_multparent
  logical :: dopr_collision  = def_dopr_collision
  character(len=16) :: print_method    = def_print_method
  character(len=16) :: utr             = def_utr
  character(len=16) :: eventlist(maxevents) = (/('                ',j=1,maxevents)/)
  
  namelist /gptlnl/ wall, cpu, abort_on_error, overhead, depthlimit, &
                    verbose, narrowprint, percent, persec, multiplex, &
                    dopr_preamble, dopr_threadsort, dopr_multparent, dopr_collision, &
                    print_method, eventlist, utr

  open (unit=unitno, file=filename, status='old', iostat=ios)
  if (ios /= 0) then
    write(6,*)'gptlprocess_namelist: cannot open namelist file ', filename
    outret = -1
    return
  end if

  read (unitno, gptlnl, iostat=ios)
  if (ios /= 0) then
    write(6,*)'gptlprocess_namelist: failure reading namelist'
    outret = -1
    return
  end if

! Set options for user-defined values which are not default.
! Do verbose and abort_on_error first

  if (verbose .neqv. def_verbose) then
    if (verbose) then
      ret = gptlsetoption (gptlverbose, 1)
    else
      ret = gptlsetoption (gptlverbose, 0)
    end if
  end if

  if (abort_on_error .neqv. def_abort_on_error) then
    if (abort_on_error) then
      ret = gptlsetoption (gptlabort_on_error, 1)
    else
      ret = gptlsetoption (gptlabort_on_error, 0)
    end if
  end if

  if (wall .neqv. def_wall) then
    if (wall) then
      ret = gptlsetoption (gptlwall, 1)
    else
      ret = gptlsetoption (gptlwall, 0)
    end if
  end if

  if (cpu .neqv. def_cpu) then
    if (cpu) then
      ret = gptlsetoption (gptlcpu, 1)
    else
      ret = gptlsetoption (gptlcpu, 0)
    end if
  end if

  if (overhead .neqv. def_overhead) then
    if (overhead) then
      ret = gptlsetoption (gptloverhead, 1)
    else
      ret = gptlsetoption (gptloverhead, 0)
    end if
  end if

  if (depthlimit /= def_depthlimit) then
    ret = gptlsetoption (gptldepthlimit, depthlimit)
  end if

  if (narrowprint .neqv. def_narrowprint) then
    if (narrowprint) then
      ret = gptlsetoption (gptlnarrowprint, 1)
    else
      ret = gptlsetoption (gptlnarrowprint, 0)
    end if
  end if

  if (percent .neqv. def_percent) then
    if (percent) then
      ret = gptlsetoption (gptlpercent, 1)
    else
      ret = gptlsetoption (gptlpercent, 0)
    end if
  end if

  if (persec .neqv. def_persec) then
    if (persec) then
      ret = gptlsetoption (gptlpersec, 1)
    else
      ret = gptlsetoption (gptlpersec, 0)
    end if
  end if

  if (multiplex .neqv. def_multiplex) then
    if (multiplex) then
      ret = gptlsetoption (gptlmultiplex, 1)
    else
      ret = gptlsetoption (gptlmultiplex, 0)
    end if
  end if

  if (dopr_preamble .neqv. def_dopr_preamble) then
    if (dopr_preamble) then
      ret = gptlsetoption (gptldopr_preamble, 1)
    else
      ret = gptlsetoption (gptldopr_preamble, 0)
    end if
  end if

  if (dopr_threadsort .neqv. def_dopr_threadsort) then
    if (dopr_threadsort) then
      ret = gptlsetoption (gptldopr_threadsort, 1)
    else
      ret = gptlsetoption (gptldopr_threadsort, 0)
    end if
  end if

  if (dopr_multparent .neqv. def_dopr_multparent) then
    if (dopr_multparent) then
      ret = gptlsetoption (gptldopr_multparent, 1)
    else
      ret = gptlsetoption (gptldopr_multparent, 0)
    end if
  end if

  if (dopr_collision .neqv. def_dopr_collision) then
    if (dopr_collision) then
      ret = gptlsetoption (gptldopr_collision, 1)
    else
      ret = gptlsetoption (gptldopr_collision, 0)
    end if
  end if
!
! Character-based variables
!
  if (utr /= def_utr) then
    if (trim(utr) == 'gettimeofday') then
      ret = gptlsetutr (gptlgettimeofday)
    else if (trim(utr) == 'nanotime') then
      ret = gptlsetutr (gptlnanotime)
    else if (trim(utr) == 'rtc') then
      ret = gptlsetutr (gptlrtc)
    else if (trim(utr) == 'mpiwtime') then
      ret = gptlsetutr (gptlmpiwtime)
    else if (trim(utr) == 'clockgettime') then
      ret = gptlsetutr (gptlclockgettime)
    else if (trim(utr) == 'papitime') then
      ret = gptlsetutr (gptlpapitime)
    else
      write(6,*)'gptlprocess_namelist: Underlying timing routine not available: ', utr
    end if
  end if
!
! Print method: use characters for namelist variables to avoid magic numbers
! in namelist
!
  if (print_method /= def_print_method) then
    if (trim(print_method) == 'first_parent') then
      ret = gptlsetoption (gptlprint_method, gptlfirst_parent)
    else if (trim(print_method) == 'last_parent') then
      ret = gptlsetoption (gptlprint_method, gptllast_parent)
    else if (trim(print_method) == 'most_frequent') then
      ret = gptlsetoption (gptlprint_method, gptlmost_frequent)
    else if (trim(print_method) == 'full_tree') then
      ret = gptlsetoption (gptlprint_method, gptlfull_tree)
    else
      write(6,*)'gptlprocess_namelist: print_method not available: ', print_method
    end if
  end if
!
! PAPI-based events
!
  do j=1,maxevents
    if (eventlist(j) /= '                ') then
      ret = gptlevent_name_to_code (trim (eventlist(j)), code)
      if (ret == 0) then
        ret = gptlsetoption (code, 1)
      else
        write(6,*)'gptlprocess_namelist: no code found for event ', eventlist(j)
      end if
    end if
  end do

  close (unit=unitno)
  outret = 0
  return
end subroutine gptlprocess_namelist
