      program omptest
      implicit none

      include '../gpt.inc'

      integer gptsetoption
      integer gptinitialize
      integer gptstart
      integer gptstop
      integer gptpr
      
      external gptsetoption
      external gptinitialize
      external gptstart
      external gptstop
      external gptpr

      integer ret, iter
      
      if (gptsetoption (usrsys, 1) < 0) call exit(99)
      if (gptsetoption (wall, 1) < 0) call exit(99)
      if (gptsetoption (pcl_l1dcache_miss, 0) < 0) call exit(99)
      if (gptsetoption (pcl_l2cache_miss, 0) < 0) call exit(99)
      if (gptsetoption (pcl_cycles, 0) < 0) call exit(99)
      if (gptsetoption (pcl_elapsed_cycles, 0) < 0) call exit(99)
      if (gptsetoption (pcl_fp_instr, 0) < 0) call exit(99)
      if (gptsetoption (pcl_loadstore_instr, 0) < 0) call exit(99)
      if (gptsetoption (pcl_instr, 0) < 0) call exit(99)
      if (gptsetoption (pcl_stall, 0) < 0) call exit(99)
      if (gptsetoption (pcl_l2cache_miss, 0) < 0) call exit(99)
      if (gptinitialize() < 0) then
        write(6,*)'gptinitialize failed'
        call exit(99)
      end if
!      call gptsetoption (usrsys, .false.)

      if (gptstart ('outside_nothing') < 0) stop 99

!!$OMP PARALLEL DO PRIVATE (iter)

      do iter=1,2
        if (gptstart ('inside_nothing') < 0) stop 99
        call do_nothing (iter)
        if (gptstop ('inside_nothing') < 0) stop 99
      end do

      if (gptstop ('outside_nothing') < 0) stop 99

!$OMP PARALLEL DO PRIVATE (iter)

      do iter=1,32
        call sub(iter)
      end do

      if (gptpr(0) < 0) call exit(99)
      stop
      end program
          
      subroutine sub(iter)
      implicit none

      character*7 name
      integer iter, ret
      real sum1, sum2, x

      integer gptstart
      integer gptstop
      
      external gptstart
      external gptstop

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
      integer :: iter
      return
      end
