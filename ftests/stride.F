      program stride
      implicit none
      save

#include "../gptl.inc"
#ifdef HAVE_PAPI
#include <f77papi.h>
#endif

      integer len
      integer nstride
      integer looplen

      parameter (len=100000000)
      parameter (nstride=64)
      parameter (looplen=len/nstride)

      integer id
      integer n

      real arr(len)
      
      write(6,*)'Test strided reads/writes to memory'

      if (gptlsetoption (gptlabort_on_error, 1) .lt. 0) stop 99

      if (gptlsetoption (gptlcpu, 0) .lt. 0) stop 99
      if (gptlsetoption (gptlwall, 1) .lt. 0) stop 99

#ifdef HAVE_PAPI
! Test gptl_papiname2id
      id = gptl_papiname2id ('PAPI_TOT_CYC')
      if (id .ge. 0) call exit (1)
      if (gptlsetoption (id, 1) .lt. 0) stop 99
#endif

      if (gptlinitialize () .lt. 0) stop 99
      if (gptlstart ('total') .lt. 0) stop 99

      do n=1,nstride
        call writesub (arr,n,looplen)
        call readsub (arr,n,looplen)
      end do

      if (gptlstop ('total') .lt. 0) stop 99
      if (gptlpr (0) .lt. 0) stop 99

      stop 0
      end

      subroutine writesub (arr,n,looplen)
      implicit none
#include "../gptl.inc"
      integer n
      integer looplen
      real arr(n,looplen)

      integer i
      integer ret
      character*128 string

      if (n .gt. 8 .and. mod (n,2) .ne. 0) then
        ret = gptldisable ()
      end if

      write(string,'(a,i7.7,a,i3.3)')'write_len',looplen,'stride',n
      if (gptlstart (string) .lt. 0) stop 99
      do i=1,looplen
        arr(1,i) = i
      end do
      if (gptlstop (string) .lt. 0) stop 99

      ret = gptlenable ()

      return
      end

      subroutine readsub (arr,n,looplen)
      implicit none
#include "../gptl.inc"
      integer n
      integer looplen
      real arr(n,looplen)

      integer i
      integer ret
      real x
      character*128 string
      
      if (n .gt. 8 .and. mod (n,2) .ne. 0) then
        ret = gptldisable ()
      end if

      write(string,'(a,i7.7,a,i3.3)')'read_len',looplen,'stride',n

      if (gptlstart (string) .lt. 0) stop 99
      do i=1,looplen
        x = x + arr(1,i)
      end do
      if (gptlstop (string) .lt. 0) stop 99
      arr(1,1) = x

      ret = gptlenable ()

      return
      end
