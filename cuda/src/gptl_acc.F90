module gptl_acc
! GPTL module file for user code. This file contains an interface block for 
! parameter checking.

  implicit none
  public

! Function prototypes

  interface
    integer function gptlinit_handle_gpu (name, handle)
      character(len=*) :: name
      integer :: handle
!$acc routine seq
    end function gptlinit_handle_gpu

    integer function gptlstart_gpu (handle)
      integer :: handle
!$acc routine seq
    end function gptlstart_gpu

    integer function gptlstop_gpu (handle)
      integer :: handle
!$acc routine seq
    end function gptlstop_gpu

    integer function gptlmy_sleep (seconds)
      real :: seconds
!$acc routine seq                                                               
    end function gptlmy_sleep

    subroutine gptldummy_gpu () bind(C,name="GPTLdummy_gpu")
      use iso_c_binding
!$acc routine seq
    end subroutine gptldummy_gpu

    integer function gptlget_wallclock_gpu (handle, accum, maxval, minval)
      use iso_c_binding, only: c_double
      integer :: handle
      real(c_double) :: accum
      real(c_double) :: maxval
      real(c_double) :: minval
!$acc routine seq
    end function gptlget_wallclock_gpu
    
    integer function gptlget_warp_thread (warp, thread)
      integer :: warp, thread
!$acc routine seq                                                               
    end function gptlget_warp_thread

    integer function gptlsliced_up_how (txt)
      character(len=*) :: txt
!$acc routine seq                                                               
    end function gptlsliced_up_how

    integer(c_int) function gptlcuprofilerstart () bind(C,name="GPTLcuProfilerStart")
      use iso_c_binding
!$acc routine seq                                                               
    end function gptlcuprofilerstart

    integer(c_int) function gptlcuprofilerstop () bind(C,name="GPTLcuProfilerStop")
      use iso_c_binding

!$acc routine seq                                                               
    end function gptlcuprofilerstop
  end interface
end module gptl_acc
