      program omptest
      implicit none

#include "../gpt.inc"

      integer ret, iter
      
      if (gptsetoption (gptcpu, 1) .lt. 0) call exit(99)
      if (gptsetoption (gptwall, 1) .lt. 0) call exit(99)
      if (gptsetoption (gptabort_on_error, 1) .lt. 0) call exit(99)
!      if (gpt_setoption (pcl_l1dcache_miss, 0) .lt. 0) call exit(99)
!      if (gpt_setoption (pcl_l2cache_miss, 0) .lt. 0) call exit(99)
!      if (gpt_setoption (pcl_cycles, 1) .lt. 0) call exit(99)
!      if (gpt_setoption (pcl_elapsed_cycles, 0) .lt. 0) call exit(99)
!      if (gpt_setoption (pcl_fp_instr, 1) .lt. 0) call exit(99)
!      if (gpt_setoption (pcl_loadstore_instr, 0) .lt. 0) call exit(99)
!      if (gpt_setoption (pcl_instr, 1) .lt. 0) call exit(99)
!      if (gpt_setoption (pcl_stall, 0) .lt. 0) call exit(99)
!      if (gpt_setoption (pcl_l2cache_miss, 0) .lt. 0) call exit(99)
!      call gpt_setoption (cpu, .false.)

      if (gptinitialize () .lt. 0) stop 99

      if (gptstart ('outside_nothing') .lt. 0) stop 99

!!$OMP PARALLEL DO PRIVATE (iter)

      do iter=1,2
        if (gptstart ('inside_nothing') .lt. 0) stop 99
        call do_nothing (iter)
        if (gptstop ('inside_nothing') .lt. 0) stop 99
      end do

      if (gptstop ('outside_nothing') .lt. 0) stop 99

!$OMP PARALLEL DO PRIVATE (iter)

      do iter=1,64
        call sub (iter)
      end do

!$OMP PARALLEL DO PRIVATE (iter)

      do iter=1,64
        call sub2 (iter)
      end do

      if (gptpr (0) .lt. 0) call exit (99)
      if (gptfinalize () .lt. 0) call exit (99)
      stop
      end program
          
      subroutine sub(iter)
      implicit none

#include "../gpt.inc"

      character*7 name
      integer iter, ret
      real sum1, sum2, x

      integer i

      write(name,'(a4,i3.3)') 'iter',iter

      sum1 = 0
      sum2 = 0

      ret = gptstart (name)

      do i=1,iter*100000
        x = i
        sum1 = sum1 + 2.1*x
        sum2 = sum2 + 3.2*x
      end do

      ret = gptstop (name)

      write(6,*)'iter, sum=', iter, sum1, sum2
      return
      end subroutine sub

      subroutine do_nothing (iter)
      integer iter
      return
      end

      subroutine sub2(iter)
      implicit none

#include "../gpt.inc"

      integer iter, ret
      real sum1, sum2, x

      integer i

      sum1 = 0
      sum2 = 0

      ret = gptstart ('sub2')

      do i=1,iter*100000
        x = i
        sum1 = sum1 + 2.1*x
        sum2 = sum2 + 3.2*x
      end do

      ret = gptstop ('sub2')

      write(6,*)'iter, sum=', iter, sum1, sum2
      return
      end subroutine sub2
