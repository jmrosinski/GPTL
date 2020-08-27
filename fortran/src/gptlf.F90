module gptl
! GPTL module file for user code. Parameter values match their counterparts
! in gptl.h. This file also contains an interface block for parameter checking.
! Also: Some F90-only subroutines after the interface block

  implicit none
  public

! User-accessible integers 

  integer, parameter :: GPTLsync_mpi       = 0
  integer, parameter :: GPTLwall           = 1
  integer, parameter :: GPTLcpu            = 2
  integer, parameter :: GPTLabort_on_error = 3
  integer, parameter :: GPTLoverhead       = 4
  integer, parameter :: GPTLdepthlimit     = 5
  integer, parameter :: GPTLverbose        = 6
  integer, parameter :: GPTLpercent        = 9
  integer, parameter :: GPTLpersec         = 10
  integer, parameter :: GPTLmultiplex      = 11
  integer, parameter :: GPTLdopr_preamble  = 12
  integer, parameter :: GPTLdopr_threadsort= 13
  integer, parameter :: GPTLdopr_multparent= 14
  integer, parameter :: GPTLdopr_collision = 15
  integer, parameter :: GPTLdopr_memusage  = 27
  integer, parameter :: GPTLprint_method   = 16
  integer, parameter :: GPTLtablesize      = 50
  integer, parameter :: GPTLmaxthreads     = 51
  integer, parameter :: GPTLonlyprint_rank0= 52
  integer, parameter :: GPTLmem_growth     = 53
  integer, parameter :: GPTLmaxwarps_gpu   = 54
  integer, parameter :: GPTLmaxtimers_gpu  = 55

  integer, parameter :: GPTL_IPC           = 17
  integer, parameter :: GPTL_LSTPI         = 21
  integer, parameter :: GPTL_DCMRT         = 22
  integer, parameter :: GPTL_LSTPDCM       = 23
  integer, parameter :: GPTL_L2MRT         = 24
  integer, parameter :: GPTL_LSTPL2M       = 25
  integer, parameter :: GPTL_L3MRT         = 26

  integer, parameter :: GPTLgettimeofday   = 1
  integer, parameter :: GPTLnanotime       = 2
  integer, parameter :: GPTLmpiwtime       = 4
  integer, parameter :: GPTLclockgettime   = 5
  integer, parameter :: GPTLplacebo        = 7
  integer, parameter :: GPTLread_real_time = 3
					                
  integer, parameter :: GPTLfirst_parent   = 1
  integer, parameter :: GPTLlast_parent    = 2
  integer, parameter :: GPTLmost_frequent  = 3
  integer, parameter :: GPTLfull_tree      = 4

! Function prototypes

  interface
     subroutine gptlprocess_namelist (filename, unitno, outret)
       character(len=*) :: filename
       integer :: unitno
       integer :: outret
     end subroutine gptlprocess_namelist

     integer function gptlinitialize ()
     end function gptlinitialize

     integer function gptlfinalize ()
     end function gptlfinalize

     integer function gptlpr (procid)
       integer :: procid
     end function gptlpr

     integer function gptlpr_file (file)
       character(len=*) :: file
     end function gptlpr_file

#ifdef HAVE_LIBMPI
     integer function gptlpr_summary (fcomm)
       integer :: fcomm
     end function gptlpr_summary

     integer function gptlpr_summary_file (fcomm, name)
       integer :: fcomm
       character(len=*) :: name
     end function gptlpr_summary_file

     integer function gptlbarrier (fcomm, name)
       integer :: fcomm
       character(len=*) :: name
     end function gptlbarrier
#endif     

     integer function gptlreset ()
     end function gptlreset

     integer function gptlreset_timer (name)
       character(len=*) :: name
     end function gptlreset_timer

     integer function gptlstamp (wall, usr, sys)
       real(8) :: wall, usr, sys
     end function gptlstamp

     integer function gptlstart (name)
       character(len=*) :: name
     end function gptlstart

     integer function gptlinit_handle (name, handle)
       character(len=*) :: name
       integer :: handle
     end function gptlinit_handle

     integer function gptlstart_handle (name, handle)
       character(len=*) :: name
       integer :: handle
     end function gptlstart_handle

     integer function gptlstop (name)
       character(len=*) :: name
     end function gptlstop

     integer function gptlstop_handle (name, handle)
       character(len=*) :: name
       integer :: handle
     end function gptlstop_handle

     integer function gptlsetoption (option, val)
       integer :: option, val
     end function gptlsetoption

     integer function gptlenable ()
     end function gptlenable

     integer function gptldisable ()
     end function gptldisable

     integer function gptlsetutr (option)
       integer :: option
     end function gptlsetutr

     integer function gptlquery (name, t, count, onflg, wallclock, &
                                 usr, sys, papicounters_out, maxcounters)
       character(len=*) :: name
       integer :: t, count
       integer :: onflg
       real(8) :: wallclock, usr, sys
       integer(8) :: papicounters_out
       integer :: maxcounters
     end function gptlquery

     integer function gptlget_wallclock (name, t, value)
       character(len=*) :: name
       integer :: t
       real(8) :: value
     end function gptlget_wallclock

     integer function gptlget_wallclock_latest (name, t, value)
       character(len=*) :: name
       integer :: t
       real(8) :: value
     end function gptlget_wallclock_latest

     integer function gptlget_threadwork (name, maxwork, imbal)
       character(len=*) :: name
       real(8) :: maxwork
       real(8) :: imbal
     end function gptlget_threadwork

     integer function gptlstartstop_val (name, value)
       character(len=*) :: name
       real(8) :: value
     end function gptlstartstop_val

     integer function gptlget_eventvalue (timername, eventname, t, value)
       character(len=*) :: timername
       character(len=*) :: eventname
       integer :: t
       real(8) :: value
     end function gptlget_eventvalue

     integer function gptlget_nregions (t, nregions)
       integer :: t
       integer :: nregions
     end function gptlget_nregions

     integer function gptlget_regionname (t, region, name)
       integer :: t
       integer :: region
       character(len=*) :: name
     end function gptlget_regionname

     integer function gptlget_memusage (rss)
       real :: rss
     end function gptlget_memusage

     integer function gptlprint_memusage (str)
       character(len=*) :: str
     end function gptlprint_memusage

     integer function gptlget_procsiz (procsiz, rss)
       real :: procsiz, rss
     end function gptlget_procsiz

     integer function gptlnum_errors ()
     end function gptlnum_errors

     integer function gptlnum_warn ()
     end function gptlnum_warn

     integer function gptlget_count (name, t, count)
       character(len=*) :: name
       integer :: t
       integer :: count
     end function gptlget_count

#ifdef HAVE_PAPI
     integer function gptl_papilibraryinit ()
     end function gptl_papilibraryinit

     integer function gptlevent_name_to_code (str, code)
       character(len=*) :: str
       integer :: code
     end function gptlevent_name_to_code

     integer function gptlevent_code_to_name (code, str)
       integer :: code
       character(len=*) :: str
     end function gptlevent_code_to_name
#endif

#ifdef ENABLE_CUDA
     integer function gptlget_gpu_props (khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu)
       integer, intent(out) :: khz
       integer, intent(out) :: warpsize
       integer, intent(out) :: devnum
       integer, intent(out) :: smcount
       integer, intent(out) :: cores_per_sm
       integer, intent(out) :: cores_per_gpu
     end function gptlget_gpu_props

     integer function gptlcudadevsync ()
     end function gptlcudadevsync
#endif

   end interface
  
contains

  ! These routines are available only in the F90 interface (not C either)

! gptlstart_threadohd_outer: 
!   Start a timer with "_OUT" appended
!   Reset the "_FNL" inner timer to zero in preparation for a threaded loop
  integer function gptlstart_threadohd_outer (name)
    character(len=*), intent(in) :: name
    
    character(len=len(name)+4) :: outername  ! "_OUT" is 4 characters long
    character(len=len(name)+4) :: innerlast  ! "_FNL" is 4 characters long
    integer :: ret

    gptlstart_threadohd_outer = -1
    ! Append special tags for outer and and most recent inner timers
    outername = name//'_OUT'
    innerlast = name//'_FNL'

    ret = gptlstart (outername)
    if (ret /= 0) then
      write(6,*)'gptlstart_threadohd_outer: Failure from gptlstart* name=',outername
      return
    end if

    !TODO: Check the return code from gptlreset_timer
    ret = gptlreset_timer (innerlast)
    gptlstart_threadohd_outer = 0
  end function gptlstart_threadohd_outer

! gptlstart_threadohd_inner: 
!   Start a timer with '_INN" appended
!   Start a timer with "_FNL" appended for subsequent use in overhead calcs.
  integer function gptlstart_threadohd_inner (name)
    character(len=*), intent(in) :: name
    
    character(len=len(name)+4) :: innername  ! "_INN" is 4 characters long
    character(len=len(name)+4) :: innerlast  ! "_FNL" is 4 characters long
    integer :: ret
    
    gptlstart_threadohd_inner = -1
    innername = name//'_INN'
    innerlast = name//'_FNL'

    ret = gptlstart (innername)
    if (ret /= 0) then
      write(6,*)'gptlstart_threadohd_inner: Failure from gptlstart* name=',innername
      return
    end if

    ret = gptlstart (innerlast)
    if (ret /= 0) then
      write(6,*)'gptlstart_threadohd_inner: Failure from gptlstart* name=',innerlast
      return
    end if
    gptlstart_threadohd_inner = 0
  end function gptlstart_threadohd_inner
  
! gptlstop_threadohd_inner: 
!   Stop a timer with '_INN" appended
!   Stop a timer with "_FNL" appended for subsequent use in overhead calcs.
  integer function gptlstop_threadohd_inner (name)
    character(len=*), intent(in) :: name
    
    character(len=len(name)+4) :: innername  ! "_INN" is 4 characters long
    character(len=len(name)+4) :: innerlast  ! "_FNL" is 4 characters long
    integer :: ret
    
    gptlstop_threadohd_inner = -1
    innername = name//'_INN'
    innerlast = name//'_FNL'

    ret = gptlstop (innerlast)
    if (ret /= 0) then
      write(6,*)'gptlstop_threadohd_inner: Failure from gptlstop* name=',innerlast
      return
    end if

    ret = gptlstop (innername)
    if (ret /= 0) then
      write(6,*)'gptlstop_threadohd_inner: Failure from gptlstop* name=',innername
      return
    end if
    gptlstop_threadohd_inner = 0
  end function gptlstop_threadohd_inner
  
! gptlstop_threadohd_outer:
!   Estimate threading overhead and threading load imbalance, using the formulae:
!   overhead = time wrapping do/enddo minus slowest thread inside the loop
!   imbalance = slowest thread minus mean time taken by all threads,
!     where "mean time taken by all threads" is taken as best possible time
!   Two pseudo-timers are invoked, with "_imbal" and "_thdovr" appended to be
!     tracked and reported by the main GPTL library
  integer function gptlstop_threadohd_outer (name)
    character(len=*), intent(in) :: name
    
    real(8) :: innermax   ! Max time across threads
    real(8) :: imbal      ! Computed imbalance
    real(8) :: outertime  ! thread 0 time outside loop
    real(8) :: threadover ! Computed overhead due to threading overhead
    
    character(len=len(name)+4) :: outername  ! "_OUT" is 4 characters long
    character(len=len(name)+4) :: innerlast  ! "_FNL" is 4 characters long
    integer :: ret
    
    gptlstop_threadohd_outer = -1  ! Init to bad value
    outername = name//'_OUT'
    innerlast = name//'_FNL'

    ret = gptlstop (outername)
    if (ret /= 0) then
      write(6,*)'gptlstop_threadohd_outer: Failure from gptlstop* name=',outername
      return
    end if
    
    if (gptlget_threadwork (innerlast, innermax, imbal) /= 0) then
      write(6,*)'gptlstop_threadohd_outer: Failure from gptlget_threadwork name=',innerlast
      return
    end if

    if (gptlget_wallclock_latest (outername, 0, outertime) /= 0) then
      write(6,*)'gptlstop_threadohd_outer: Failure from gptlget_wallclock name=',outername
      return
    end if

    if (gptlstartstop_val (name//'_imbal', imbal) /= 0) then
      write(6,*)'gptlstop_threadohd_outer: Failure from gptlstartstop_val name=',name//'_imbal'
      return
    end if

    threadover = outertime - innermax
    if (gptlstartstop_val (name//'_thdovr', threadover) /= 0) then
      write(6,*)'gptlstop_threadohd_outer: Failure from gptlstartstop_val name=',name//'_thdovr'
      return
    end if
    gptlstop_threadohd_outer = 0
  end function gptlstop_threadohd_outer
end module gptl
