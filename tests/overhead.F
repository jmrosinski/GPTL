      program overhead
      implicit none

      include '../gpt.inc'

      integer ret, iter
      character*3 ans
      
      if (gptsetoption (gptwall, 1) < 0) call exit(99)

 1    write(6,*) 'Include CPU timings?'
      read (5,*) ans
      if (ans(1:1) == 'y' .or. ans(1:1) == 'Y') then
        write(6,*)'CPU timings will be included'
        if (gptsetoption (gptcpu, 1) < 0) call exit(99)
      else if (ans(1:1) == 'n' .or. ans(1:1) == 'N') then
        write(6,*)'CPU timings will not be included'
        if (gptsetoption (gptcpu, 0) < 0) call exit(99)
      else
        goto 1
      end if

      if (gptinitialize () < 0) stop 99

      if (gptstart('total') .lt. 0) stop 999
!JR Threaded
!$OMP PARALLEL DO PRIVATE (iter)

      do iter=1,1000
        call sub(iter,'threaded')
        call sub(iter,'threaded')
      end do

!JR Unthreaded
      do iter=1,1000
        call sub(iter,'unthreaded')
        call sub(iter,'unthreaded')
      end do

      if (gptstop ('total') .lt. 0) stop 999
      if (gptpr (0) < 0) call exit(99)
      stop
      end program overhead
          
      subroutine sub (iter, str)
      implicit none
      include '../gpt.inc'

      integer iter
      character*(*) str
      
      character*16 name
      integer ret
      real sum1, sum2, x

      integer i

      write(name,'(a,i4.4)') str, iter

      sum1 = 0
      sum2 = 0

      if (gptstart (name) .lt. 0) stop 999

      do i=1,iter*100
        x = i
        sum1 = sum1 + 2.1*x
        sum2 = sum2 + 3.2*x
      end do

      if (gptstop (name) .lt. 0) stop 999

      write(6,*)'iter, sum=', iter, sum1, sum2
      return
      end subroutine sub
