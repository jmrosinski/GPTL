module gptl_acc
! GPTL module file for user code. This file contains an interface block for 
! parameter checking.

  implicit none
  public

! Function prototypes

  interface
    integer function gptlstart_gpu (name)
      character(len=*) :: name
!$acc routine seq
    end function gptlstart_gpu

    integer function gptlinit_handle_gpu (name, handle)
      character(len=*) :: name
      integer :: handle
!$acc routine seq
    end function gptlinit_handle_gpu

    integer function gptlstart_handle_gpu (name, handle)
      character(len=*) :: name
      integer :: handle
!$acc routine seq
    end function gptlstart_handle_gpu

    integer function gptlstop_gpu (name)
      character(len=*) :: name
!$acc routine seq
    end function gptlstop_gpu

    integer function gptlstop_handle_gpu (name, handle)
      character(len=*) :: name
      integer :: handle
!$acc routine seq
    end function gptlstop_handle_gpu

    integer function gptlmy_sleep (seconds)
      real :: seconds
!$acc routine seq                                                               
    end function gptlmy_sleep

    subroutine gptldummy_gpu (num) bind(C,name="GPTLdummy_gpu")
      use iso_c_binding
      integer(c_int), intent(in), value :: num
!$acc routine seq
    end subroutine gptldummy_gpu

    integer function gptlget_wallclock_gpu (name, accum, maxval, minval)
      use iso_c_binding, only: c_double
      character(len=*) :: name
      real(c_double) :: accum
      real(c_double) :: maxval
      real(c_double) :: minval
!$acc routine seq
    end function gptlget_wallclock_gpu
  end interface
end module gptl_acc
