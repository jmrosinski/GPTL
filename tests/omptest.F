      program omptest
      implicit none

      include '../header.inc'

      integer t_setoptionf
      integer t_initializef
      integer t_startf
      integer t_stopf
      integer t_prf
      
      external t_setoptionf
      external t_initializef
      external t_startf
      external t_stopf
      external t_prf

      integer ret, iter
      
      if (t_setoptionf (usrsys, 0) < 0) call exit(99)
      if (t_setoptionf (wall, 0) < 0) call exit(99)
      if (t_setoptionf (pcl_l1dcache_miss, 0) < 0) call exit(99)
      if (t_setoptionf (pcl_l2cache_miss, 0) < 0) call exit(99)
      if (t_setoptionf (pcl_cycles, 1) < 0) call exit(99)
      if (t_setoptionf (pcl_elapsed_cycles, 0) < 0) call exit(99)
      if (t_setoptionf (pcl_fp_instr, 1) < 0) call exit(99)
      if (t_setoptionf (pcl_loadstore_instr, 0) < 0) call exit(99)
      if (t_setoptionf (pcl_instr, 0) < 0) call exit(99)
      if (t_setoptionf (pcl_stall, 0) < 0) call exit(99)
      if (t_setoptionf (pcl_l2cache_miss, 0) < 0) call exit(99)
      if (t_initializef() < 0) then
        write(6,*)'t_initializef failed'
        call exit(99)
      end if
!      call t_setoptionf (usrsys, .false.)

      if (t_startf ('outside_nothing') < 0) stop 99

!!$OMP PARALLEL DO PRIVATE (iter)

      do iter=1,2
        if (t_startf ('inside_nothing') < 0) stop 99
        call do_nothing (iter)
        if (t_stopf ('inside_nothing') < 0) stop 99
      end do

      if (t_stopf ('outside_nothing') < 0) stop 99

!$OMP PARALLEL DO PRIVATE (iter)

      do iter=1,64
        call sub(iter)
      end do

      if (t_prf(0) < 0) call exit(99)
      stop
      end program
          
      subroutine sub(iter)
      implicit none

      character*7 name
      integer iter, ret
      real sum1, sum2, x

      integer t_startf
      integer t_stopf
      
      external t_startf
      external t_stopf

      integer i

      write(name,'(a4,i3.3)') 'iter',iter

      sum1 = 0
      sum2 = 0

      ret = t_startf (name)

      do i=1,iter*100000
        x = i
        sum1 = sum1 + 2.1*x
        sum2 = sum2 + 3.2*x
      end do

      ret = t_stopf (name)

      write(6,*)'iter, sum=', iter, sum1, sum2
      return
      end subroutine sub

      subroutine do_nothing (iter)
      integer :: iter
      return
      end
