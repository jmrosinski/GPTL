program utrtest
  use gptl
#ifdef THREADED_OMP
  use omp_lib
#endif

  implicit none

  external :: sub

  logical :: enable_handle = .true.  ! true means include handle start/stop routines
  logical :: enable_name = .true.    ! true means include non-handle start/stop routines
  double precision :: sum = 0.
  integer :: ret
  integer :: handle1
  integer :: handle2
  integer :: handle3
  integer :: handle4
  integer :: handle5
  integer :: handle6
  integer :: handle7
  integer :: handle8

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

  ret = gptlsetutr (gptlgettimeofday)  ! set the default underlying timing routine

  narg = command_argument_count ()
  call get_command_argument (1, arg)
  n = 1
  do while (n < narg)
    select case (trim(arg))
    case ('-h')
      enable_handle = .false.
      n = n + 1
    case ('-n')
      enable_name = .false.
      n = n + 1
    case default
      call get_command_argument (n, arg)
      if (trim (arg) == '-t') then
        if (n+1 > narg) then
          call printusemsg_exit ()
        end if
        call get_command_argument (n+1, arg)
        select case (trim(arg))
        case ('gettimeofday')
          ret = gptlsetutr (gptlgettimeofday)
        case ('nanotime')
          if (gptlsetutr (gptlnanotime) /= 0) then
            write(6,*) 'nanotime not available on this arch'
            stop 1
          end if
        case ('mpi_wtime')
          if (gptlsetutr (gptlmpiwtime) /= 0) then
            write(6,*) 'MPI was not enabled at build time so mpi_wtime not available'
            stop 1
          end if
        case ('clock_gettime')
          if (gptlsetutr (gptlclockgettime) /= 0) then
            write(6,*) 'clock_gettime was not enabled at build time so clock_gettime not available'
            stop 1
          end if
        case ('placebo')
          ret = gptlsetutr (gptlplacebo)
        case default
          write(6,*)'Unknown argument ', trim(arg)
          call printusemsg_exit ()
          stop 1
        end select
        n = n + 2   ! for '-t <timer>'
      else
        write(6,*)'Unknown flag ',trim(arg),' Only -h -n and -t [timer] are known'
        stop 1
      end if
    end select
  end do

  write(6,*) 'Purpose: estimate overhead of GPTL underlying timing routine (UTR)'
  
  ret = gptlinitialize ()

  if (enable_name) then
    ret = gptlstart ('total_startstop')
    call sub (1, 10000000, "1x1e7", sum)
    call sub (10, 1000000, "10x1e6", sum)
    call sub (100, 100000, "100x1e5", sum)
    call sub (1000, 10000, "1000x1e4", sum)
    call sub (10000, 1000, "1e4x1000", sum)
    call sub (100000, 100, "1e5x100", sum)
    call sub (1000000, 10, "1e6x10", sum)
    call sub (10000000, 1, "1e7x1", sum)
    ret = gptlstop ("total_startstop")
  end if

  if (enable_name) then
    ret = gptlinit_handle ('1x1e7_handle', handle1)
    ret = gptlinit_handle ('10x1e6_handle', handle2)
    ret = gptlinit_handle ('100x1e5_handle', handle3)
    ret = gptlinit_handle ('1000x1e4_handle', handle4)
    ret = gptlinit_handle ('1e4x1000_handle', handle5)
    ret = gptlinit_handle ('1e5x100_handle', handle6)
    ret = gptlinit_handle ('1e6x10_handle', handle7)
    ret = gptlinit_handle ('1e7x1_handle', handle8)
  
    ret = gptlstart ('total_handle')
    call sub_handle (1, 10000000, "1x1e7_handle", sum, handle1)
    call sub_handle (10, 1000000, "10x1e6_handle", sum, handle2)
    call sub_handle (100, 100000, "100x1e5_handle", sum, handle3)
    call sub_handle (1000, 10000, "1000x1e4_handle", sum, handle4)
    call sub_handle (10000, 1000, "1e4x1000_handle", sum, handle5)
    call sub_handle (100000, 100, "1e5x100_handle", sum, handle6)
    call sub_handle (1000000, 10, "1e6x10_handle", sum, handle7)
    call sub_handle (10000000, 1, "1e7x1_handle", sum, handle8)
    ret = gptlstop ("total_handle")
  end if

  ret = gptlpr (-1)  ! negative number means write to stderr
  stop 0

CONTAINS

  subroutine printusemsg_exit
    implicit none

    integer :: n
    write(6,*)'Usage: utrtest [-t ',(av_timers(n),n=1,n_av_timers-1),' | ', &
                                     av_timers(n_av_timers), ']'
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

  integer, intent(in) :: outer
  integer, intent(in) :: inner
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

