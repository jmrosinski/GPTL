      program omptest
      implicit none

#include "../gptl.inc"

      integer ret, iter
      
      if (gptlsetoption (gptlcpu, 1) .lt. 0) call exit(99)
      if (gptlsetoption (gptlwall, 1) .lt. 0) call exit(99)
      if (gptlsetoption (gptlabort_on_error, 1) .lt. 0) call exit(99)
!      if (gptl_setoption (pcl_l1dcache_miss, 0) .lt. 0) call exit(99)
!      if (gptl_setoption (pcl_l2cache_miss, 0) .lt. 0) call exit(99)
!      if (gptl_setoption (pcl_cycles, 1) .lt. 0) call exit(99)
!      if (gptl_setoption (pcl_elapsed_cycles, 0) .lt. 0) call exit(99)
!      if (gptl_setoption (pcl_fp_instr, 1) .lt. 0) call exit(99)
!      if (gptl_setoption (pcl_loadstore_instr, 0) .lt. 0) call exit(99)
!      if (gptl_setoption (pcl_instr, 1) .lt. 0) call exit(99)
!      if (gptl_setoption (pcl_stall, 0) .lt. 0) call exit(99)
!      if (gptl_setoption (pcl_l2cache_miss, 0) .lt. 0) call exit(99)
!      call gptl_setoption (cpu, .false.)

      if (gptlinitialize () .lt. 0) stop 99

      if (gptlstart ('outside_nothing') .lt. 0) stop 99

!!$OMP PARALLEL DO PRIVATE (iter)

      do iter=1,2
        if (gptlstart ('inside_nothing') .lt. 0) stop 99
        call do_nothing (iter)
        if (gptlstop ('inside_nothing') .lt. 0) stop 99
      end do

      if (gptlstop ('outside_nothing') .lt. 0) stop 99

!$OMP PARALLEL DO PRIVATE (iter)

      do iter=1,64
        call sub (iter)
      end do

!$OMP PARALLEL DO PRIVATE (iter)

      do iter=1,64
        call sub2 (iter)
      end do

      if (gptlpr (0) .lt. 0) call exit (99)
      if (gptlfinalize () .lt. 0) call exit (99)
      stop
      end program
          
      subroutine sub(iter)
      implicit none

#include "../gptl.inc"

      character*7 name
      integer iter, ret
      real sum1, sum2, x

      integer i

      write(name,'(a4,i3.3)') 'iter',iter

      sum1 = 0
      sum2 = 0

      ret = gptlstart (name)

      do i=1,iter*100000
        x = i
        sum1 = sum1 + 2.1*x
        sum2 = sum2 + 3.2*x
      end do

      ret = gptlstop (name)

      write(6,*)'iter, sum=', iter, sum1, sum2
      return
      end subroutine sub

      subroutine do_nothing (iter)
      integer iter
      return
      end

      subroutine sub2(iter)
      implicit none

#include "../gptl.inc"

      integer iter, ret
      real sum1, sum2, x

      integer i

      sum1 = 0
      sum2 = 0

      ret = gptlstart ('sub2')

      do i=1,iter*100000
        x = i
        sum1 = sum1 + 2.1*x
        sum2 = sum2 + 3.2*x
      end do

      ret = gptlstop ('sub2')

      write(6,*)'iter, sum=', iter, sum1, sum2
      return
      end subroutine sub2
