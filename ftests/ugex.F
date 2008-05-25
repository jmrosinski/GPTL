      program ugex
      implicit none

#include "../gptl.inc"
#ifdef HAVE_PAPI
#include "f90papi.h"
#endif

      integer, parameter :: nompiter = 128 ! iteration count for threaded loop
      integer, parameter :: ny = 9         ! iteration count for middle loop
      integer, parameter :: nx = 1000000   ! iteration count for inner loop
      
      integer :: i, j, iter        ! loop indices
      integer :: ret               ! return code
      integer*8 :: papicounters(3) ! PAPI counter values
      
      real*8 :: sums(nompiter)     ! summation array
      
      if (gptlsetoption (gptlverbose, 0) < 0) call exit (1)       ! turn off verbosity
      if (gptlsetoption (gptlabort_on_error, 1) < 0) call exit(2) ! abort on error
      
#ifdef HAVE_PAPI
      ret = gptlsetoption (PAPI_FP_INS, 1)     ! count FP instructions
      ret = gptlsetoption (PAPI_TOT_CYC, 1)    ! count total cycles
#endif
      ret = gptlsetoption (gptloverhead, 0)    ! don't print overhead stats
      ret = gptlsetoption (gptlnarrowprint, 1) ! print fewer sig figs
      
      ret = gptlinitialize ()   ! initialize GPTL
      ret = gptlstart ('total') ! start a timer for the entire program
      
      ret = gptlstart ('init')  ! start a timer
      do i=1,nompiter
         sums(i) = 0.
      end do
      ret = gptlstop ('init')   ! stop a timer
      
! Invoke a threaded loop, and gather timing info
      
!$OMP PARALLEL DO PRIVATE (i, j, iter, ret)
      
      do iter=1,nompiter
         ret = gptlstart ('Jloop')
         do j=1,ny
            ret = gptlstart ('Iloop1')
            do i=1,nx
               sums(iter) = sums(iter) + 0.0001*i
            end do
            ret = gptlstop ('Iloop1')
            
            ret = gptlstart ('Iloop2')
            do i=1,nx
               sums(iter) = sums(iter) + i
            end do
            ret = gptlstop ('Iloop2')
         end do
         ret = gptlstop ('Jloop') ! stop timer
      end do
      
      ret = gptlstop ('total')  ! stop the timer for the entire program
      
! Retrieve the PAPI counters for timer 'total' and print them
      
      ret = gptlquerycounters ('total', -1, papicounters)
      write(6,*)'total PAPI_FP_INS=    ', papicounters(1)
      write(6,*)'total PAPI_TOT_CYC=   ', papicounters(2)
      
      ret = gptlpr (0)          ! print the timing results to timing.0
      ret = gptlfinalize ()     ! clean up
      
      stop 0
      end program ugex
      
