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

     integer function gptlstart_gpu_c (name)
       character(len=*) :: name
!$acc routine seq
     end function gptlstart_gpu_c

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

     integer function gptlstart_handle_gpu_c (name, handle)
       character(len=*) :: name
       integer :: handle
!$acc routine seq
     end function gptlstart_handle_gpu_c

     integer function gptlstop_gpu (name)
       character(len=*) :: name
!$acc routine seq
     end function gptlstop_gpu

     integer function gptlstop_gpu_c (name)
       character(len=*) :: name
!$acc routine seq
     end function gptlstop_gpu_c

     integer function gptlstop_handle_gpu (name, handle)
       character(len=*) :: name
       integer :: handle
!$acc routine seq
     end function gptlstop_handle_gpu

     integer function gptlstop_handle_gpu_c (name, handle)
       character(len=*) :: name
       integer :: handle
!$acc routine seq
     end function gptlstop_handle_gpu_c

     integer function gptlmy_sleep (seconds) bind(C,name="GPTLmy_sleep")
       use iso_c_binding
       real(c_float), intent(in), value :: seconds
!$acc routine seq                                                               
     end function gptlmy_sleep

     subroutine gptldummy_gpu (num) bind(C,name="GPTLdummy_gpu")
       use iso_c_binding
       integer(c_int), intent(in), value :: num
!$acc routine seq
     end subroutine gptldummy_gpu
  end interface
end module gptl_acc
