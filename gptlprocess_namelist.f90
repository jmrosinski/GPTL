subroutine gptlprocess_namelist (filename, unitno, outret)
!
! To follow GPTL conventions this should be a function not a subroutine.
! But 'include ./gptl.inc' and then setting function gptlprocess_namelist
! to a return value causes compiler to barf. So set return value in
! output arg 'outret' instead.
!
  implicit none

  character(len=*), intent(in) :: filename
  integer, intent(in) :: unitno
  integer, intent(out) :: outret

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
  logical, parameter :: def_overhead        = .false.
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
  integer, parameter :: def_print_method    = gptlmost_frequent
  integer, parameter :: def_utr             = gptlgettimeofday

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
  integer :: print_method    = def_print_method    
  character(len=16) :: eventlist(maxevents) = (/('                ',j=1,maxevents)/)
  integer :: utr             = def_utr
  
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

  if (utr /= def_utr) then
    ret = gptlsetutr (utr)
  end if

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
