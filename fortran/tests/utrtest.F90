program utrtest
  use gptl

  implicit none

  external :: sub

  double precision :: sum = 0.
  integer :: ret
  integer :: n
  integer :: handle1
  integer :: handle2
  integer :: handle3
  integer :: handle4
  integer :: handle5
  integer :: handle6
  integer :: handle7
  integer :: handle8

  write(6,*) 'Purpose: estimate overhead of GPTL underlying timing routine (UTR)'
  write(6,*) 'Enter 1 for gettimeofday (slow, coarse grained, works everywhere)'
#ifdef HAVE_NANOTIME
  write(6,*) 'Enter 2 for nanotime (fast, fine grained, requires x86, counts cycles not seconds)'
#endif
#ifdef HAVE_LIBMPI    
  write(6,*) 'Enter 3 for MPI_Wtime (requires MPI)'
#endif
#ifdef HAVE_LIBRT
  write(6,*) 'Enter 4 for clock_gettime (hardly ever use this one)'
#endif
  write(6,*) 'Enter 5 for a do-nothing placebo (potentially useful if run under "time" for overhead)'
  read (5,*) n
  select case (n)
  case (1)
    ret = gptlsetutr (gptlgettimeofday)
#ifdef HAVE_NANOTIME
  case (2)
    ret = gptlsetutr (gptlnanotime)
#endif
#ifdef HAVE_LIBMPI    
  case (3)
    ret = gptlsetutr (gptlmpiwtime)
#endif
#ifdef HAVE_LIBRT
  case (4)
    ret = gptlsetutr (gptlclockgettime)
#endif
  case (5)
    ret = gptlsetutr (gptlplacebo)
  case default
    write(6,*)'Input value (',n,') is not between 1 and 5'
    stop 1
  end select
  
  ret = gptlinitialize ()

  ret = gptlinit_handle ('1x1e7', handle1)
  ret = gptlinit_handle ('10x1e6', handle2)
  ret = gptlinit_handle ('100x1e5', handle3)
  ret = gptlinit_handle ('1000x1e4', handle4)
  ret = gptlinit_handle ('1e4x1000', handle5)
  ret = gptlinit_handle ('1e5x100', handle6)
  ret = gptlinit_handle ('1e6x10', handle7)
  ret = gptlinit_handle ('1e7x1', handle8)
  
  ret = gptlstart ('total')
  !      ret = GPTLdisable ()
  call sub (1, 10000000, "1x1e7", sum, handle1)
  call sub (10, 1000000, "10x1e6", sum, handle2)
  call sub (100, 100000, "100x1e5", sum, handle3)
  call sub (1000, 10000, "1000x1e4", sum, handle4)
  call sub (10000, 1000, "1e4x1000", sum, handle5)
  call sub (100000, 100, "1e5x100", sum, handle6)
  call sub (1000000, 10, "1e6x10", sum, handle7)
  call sub (10000000, 1, "1e7x1", sum, handle8)
  !      ret = gptlenable ()
  ret = gptlstop ("total")

  ret = gptlpr (-1)  ! negative number means write to stderr
  stop 0
end program utrtest

subroutine sub (outer, inner, name, sum, handle)
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
  
  return
end subroutine sub
