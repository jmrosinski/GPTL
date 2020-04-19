program utrtest
  use gptl
#ifdef THREADED_OMP
  use omp_lib
#endif
  implicit none

  logical :: enable_name = .true.       ! true means include non-handle start/stop routines
  logical :: enable_handle = .true.     ! true means include handle start/stop routines
  logical :: enable_nullterm = .true.   ! true means include passing null-terminated vars
  logical :: enable_autoprof = .true.   ! true means include auto-profiling
  logical :: enable_expensive = .false. ! true means order auto-prof calls with collisions badly
  double precision :: sum = 0.
  integer :: ret
  integer :: handle1 = 0
  integer :: handle2 = 0
  integer :: handle3 = 0
  integer :: handle4 = 0
  integer :: handle5 = 0
  integer :: handle6 = 0
  integer :: handle7 = 0
  integer :: handle8 = 0

  integer :: n                 ! iterator through argument list
  integer :: narg              ! number of cmd-line args
  character(len=256) :: arg    ! cmd-line arg
  character(len=13) :: av_timers(5) = (/'gettimeofday ', &
                                        'nanotime     ', &
                                        'mpi_wtime    ', &
                                        'clock_gettime', &
                                        'placebo      '/)
  integer, parameter :: n_av_timers = size(av_timers)
  
#ifdef THREADED_OMP
  call omp_set_num_threads (1)
#endif

  ret = gptlsetutr (gptlnanotime)  ! set the default underlying timing routine
  ret = gptlsetoption (gptldopr_collision, 1)

  narg = command_argument_count ()
  n = 1
  do while (n <= narg)
    call get_command_argument (n, arg)
    if (trim(arg) == '-n') then
      enable_name = .false.
      n = n + 1
    else if (trim(arg) == '-h') then
      enable_handle = .false.
      n = n + 1
    else if (trim(arg) == '-0') then
      enable_nullterm = .false.
      n = n + 1
    else if (trim(arg) == '-a') then
      enable_autoprof = .false.
      n = n + 1
    else if (trim(arg) == '-e') then
      enable_expensive = .true.
      n = n + 1
    else
      call get_command_argument (n, arg)
      if (trim (arg) == '-t') then
        if (n+1 > narg) then
          call printusemsg_exit ()
        end if
        call get_command_argument (n+1, arg)
        if (trim(arg) == 'gettimeofday') then
          ret = gptlsetutr (gptlgettimeofday)
        else if (trim(arg) == 'nanotime') then
          if (gptlsetutr (gptlnanotime) /= 0) then
            write(6,*) 'nanotime not available on this arch'
            stop 1
          end if
        else if (trim(arg) == 'mpi_wtime') then
          if (gptlsetutr (gptlmpiwtime) /= 0) then
            write(6,*) 'MPI was not enabled at build time so mpi_wtime not available'
            stop 1
          end if
        else if (trim(arg) == 'clock_gettime') then
          if (gptlsetutr (gptlclockgettime) /= 0) then
            write(6,*) 'clock_gettime was not enabled at build time so clock_gettime not available'
            stop 1
          end if
        else if (trim(arg) == 'placebo') then
          ret = gptlsetutr (gptlplacebo)
        else
          write(6,*)'Unknown argument ', trim(arg)
          call printusemsg_exit ()
          stop 1
        end if
        n = n + 2   ! for '-t <timer>'
      else
        write(6,*)'Unknown flag ',trim(arg),' Only -n -h -0 -a and -t [timer] are known'
        stop 1
      end if
    end if
  end do

  write(6,*) 'Purpose: estimate overhead of GPTL underlying timing routine (UTR)'
  
  ret = gptlinitialize ()

  if (enable_name) then
    ret = gptlstart ('total_startstop')
    if (enable_expensive) then
      call sub (1, 10000000, '1x1e7', sum)
      call sub (10, 1000000, '10x1e6', sum)
      call sub (100, 100000, '100x1e5', sum)
      call sub (1000, 10000, '1000x1e4', sum)
      call sub (10000, 1000, '1e4x1000', sum)
      call sub (100000, 100, '1e5x100', sum)
      call sub (1000000, 10, '1e6x10', sum)
      call sub (10000000, 1, '1e7x1', sum)
    else
      call sub (10000000, 1, '1e7x1', sum)
      call sub (1000000, 10, '1e6x10', sum)
      call sub (100000, 100, '1e5x100', sum)
      call sub (10000, 1000, '1e4x1000', sum)
      call sub (1000, 10000, '1000x1e4', sum)
      call sub (100, 100000, '100x1e5', sum)
      call sub (10, 1000000, '10x1e6', sum)
      call sub (1, 10000000, '1x1e7', sum)
    end if
    ret = gptlstop ('total_startstop')
  end if

  if (enable_handle) then
    ret = gptlinit_handle ('1e7x1_handle', handle8)
    ret = gptlinit_handle ('1e6x10_handle', handle7)
    ret = gptlinit_handle ('1e5x100_handle', handle6)
    ret = gptlinit_handle ('1e4x1000_handle', handle5)
    ret = gptlinit_handle ('1000x1e4_handle', handle4)
    ret = gptlinit_handle ('100x1e5_handle', handle3)
    ret = gptlinit_handle ('10x1e6_handle', handle2)
    ret = gptlinit_handle ('1x1e7_handle', handle1)
    
    ret = gptlstart ('total_handle')
    if (enable_expensive) then
      call sub_handle (1, 10000000, '1x1e7_handle', sum, handle1)
      call sub_handle (10, 1000000, '10x1e6_handle', sum, handle2)   ! collides
      call sub_handle (100, 100000, '100x1e5_handle', sum, handle3)
      call sub_handle (1000, 10000, '1000x1e4_handle', sum, handle4)
      call sub_handle (10000, 1000, '1e4x1000_handle', sum, handle5)
      call sub_handle (100000, 100, '1e5x100_handle', sum, handle6)
      call sub_handle (1000000, 10, '1e6x10_handle', sum, handle7)   ! collides
      call sub_handle (10000000, 1, '1e7x1_handle', sum, handle8)
    else
      call sub_handle (10000000, 1, '1e7x1_handle', sum, handle8)
      call sub_handle (1000000, 10, '1e6x10_handle', sum, handle7)
      call sub_handle (100000, 100, '1e5x100_handle', sum, handle6)
      call sub_handle (10000, 1000, '1e4x1000_handle', sum, handle5)
      call sub_handle (1000, 10000, '1000x1e4_handle', sum, handle4)
      call sub_handle (100, 100000, '100x1e5_handle', sum, handle3)
      call sub_handle (10, 1000000, '10x1e6_handle', sum, handle2)
      call sub_handle (1, 10000000, '1x1e7_handle', sum, handle1)
    end if
    ret = gptlstop ('total_handle')
  end if

  if (enable_nullterm) then
    ret = gptlinit_handle ('1e7x1_handle0'//char(0), handle8)
    ret = gptlinit_handle ('1e6x10_handle0'//char(0), handle7)
    ret = gptlinit_handle ('1e5x100_handle0'//char(0), handle6)
    ret = gptlinit_handle ('1e4x1000_handle0'//char(0), handle5)
    ret = gptlinit_handle ('1000x1e4_handle0'//char(0), handle4)
    ret = gptlinit_handle ('100x1e5_handle0'//char(0), handle3)
    ret = gptlinit_handle ('10x1e6_handle0'//char(0), handle2)
    ret = gptlinit_handle ('1x1e7_handle0'//char(0), handle1)
    
    ret = gptlstart ('total_handle_nullterm'//char(0))
    if (enable_expensive) then
      call sub_handle (1, 10000000, '1x1e7_handle0'//char(0), sum, handle1)    ! collides
      call sub_handle (10, 1000000, '10x1e6_handle0'//char(0), sum, handle2)   ! collides
      call sub_handle (100, 100000, '100x1e5_handle0'//char(0), sum, handle3)  ! collides
      call sub_handle (1000, 10000, '1000x1e4_handle0'//char(0), sum, handle4)
      call sub_handle (10000, 1000, '1e4x1000_handle0'//char(0), sum, handle5)
      call sub_handle (100000, 100, '1e5x100_handle0'//char(0), sum, handle6)  ! collides
      call sub_handle (1000000, 10, '1e6x10_handle0'//char(0), sum, handle7)   ! collides
      call sub_handle (10000000, 1, '1e7x1_handle0'//char(0), sum, handle8)    ! collides
    else
      call sub_handle (10000000, 1, '1e7x1_handle0'//char(0), sum, handle8)    ! collides
      call sub_handle (1000000, 10, '1e6x10_handle0'//char(0), sum, handle7)   ! collides
      call sub_handle (100000, 100, '1e5x100_handle0'//char(0), sum, handle6)  ! collides
      call sub_handle (10000, 1000, '1e4x1000_handle0'//char(0), sum, handle5)
      call sub_handle (1000, 10000, '1000x1e4_handle0'//char(0), sum, handle4)
      call sub_handle (100, 100000, '100x1e5_handle0'//char(0), sum, handle3)  ! collides
      call sub_handle (10, 1000000, '10x1e6_handle0'//char(0), sum, handle2)   ! collides
      call sub_handle (1, 10000000, '1x1e7_handle0'//char(0), sum, handle1)    ! collides
    end if
    ret = gptlstop ('total_handle_nullterm'//char(0))
  end if

  if (enable_autoprof) then
    ret = gptlstart ('total_autoprof'//char(0))
    call sub_autoprof (10000000, 1, sum)
    call sub_autoprof (1000000, 10, sum)
    call sub_autoprof (100000, 100, sum)
    call sub_autoprof (10000, 1000, sum)
    call sub_autoprof (1000, 10000, sum)
    call sub_autoprof (100, 100000, sum)
    call sub_autoprof (10, 1000000, sum)
    call sub_autoprof (1, 10000000, sum)
    ret = gptlstop ('total_autoprof'//char(0))
  end if

  ret = gptlpr (-1)  ! negative number means write to stderr
  stop 0

CONTAINS

  subroutine printusemsg_exit
    implicit none
    integer :: n
    write(6,*)'Usage: utrtest [-n] [-h] [-0] [-a] [-t ',(av_timers(n),n=1,n_av_timers-1),' | ', &
         av_timers(n_av_timers), ']'
    write(6,*)'where -t <timer> utilizes the specified underlying timer'
    write(6,*)'where -n disables basic named timers'
    write(6,*)'      -h disables handle-based timers'
    write(6,*)'      -0 disables handle-based null-terminated timers'
    write(6,*)'      -a disables auto-profiled timers'
    stop 1
  end subroutine printusemsg_exit
end program utrtest

subroutine sub (outer, inner, name, sum)
  use gptl
  implicit none
  integer, intent(in) :: outer
  integer, intent(in) :: inner
  character(len=*), intent(in) :: name
  double precision, intent(inout) :: sum
  integer :: i, j, ret
  do i=0,outer-1
    ret = gptlstart (name)
    do j=0,inner-1
      sum = sum + j
    end do
    ret = gptlstop (name)
  end do
end subroutine sub

subroutine sub_handle (outer, inner, name, sum, handle)
  use gptl
  implicit none
  integer, intent(in) :: outer, inner
  character(len=*), intent(in) :: name
  double precision, intent(inout) :: sum
  integer, intent(inout) :: handle
  integer :: i, j, ret
  do i=0,outer-1
    ret = gptlstart_handle (name, handle)
    do j=0,inner-1
      sum = sum + j
    end do
    ret = gptlstop_handle (name, handle)
  end do
end subroutine sub_handle

! Begin auto-profiled routines
subroutine sub_autoprof (outer, inner, sum)
  implicit none
  integer, intent(in) :: outer, inner
  integer :: i
  double precision, intent(inout) :: sum
  select case (outer)
  case (1)
    do i=0,outer-1
      call innersub1 (inner, sum)
    end do
  case (10)
    do i=0,outer-1
      call innersub10 (inner, sum)
    end do
  case (100)
    do i=0,outer-1
      call innersub100 (inner, sum)
    end do
  case (1000)
    do i=0,outer-1
      call innersub1000 (inner, sum)
    end do
  case (10000)
    do i=0,outer-1
      call innersub10000 (inner, sum)
    end do
  case (100000)
    do i=0,outer-1
      call innersub100000 (inner, sum)
    end do
  case (1000000)
    do i=0,outer-1
      call innersub1000000 (inner, sum)
    end do
  case (10000000)
    do i=0,outer-1
      call innersub10000000 (inner, sum)
    end do
  case default
    write(6,*)'outer=',outer,' not known'
  end select
end subroutine sub_autoprof
