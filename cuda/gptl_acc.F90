module gptl_acc
! GPTL module file for user code. This file contains an interface block for 
! parameter checking.

  implicit none
  public

! Function prototypes

  interface
     integer function gptlstart_gpu (name)
!$acc routine seq
       character(len=*) :: name
     end function gptlstart_gpu

     integer function gptlstart_gpu_c (name)
!$acc routine seq
       character(len=*) :: name
     end function gptlstart_gpu_c

     integer function gptlinit_handle_gpu (name, handle)
!$acc routine seq
       character(len=*) :: name
       integer :: handle
     end function gptlinit_handle_gpu

     integer function gptlstart_handle_gpu (name, handle)
!$acc routine seq
       character(len=*) :: name
       integer :: handle
     end function gptlstart_handle_gpu

     integer function gptlstart_handle_gpu_c (name, handle)
!$acc routine seq
       character(len=*) :: name
       integer :: handle
     end function gptlstart_handle_gpu_c

     integer function gptlstop_gpu (name)
!$acc routine seq
       character(len=*) :: name
     end function gptlstop_gpu

     integer function gptlstop_gpu_c (name)
!$acc routine seq
       character(len=*) :: name
     end function gptlstop_gpu_c

     integer function gptlstop_handle_gpu (name, handle)
!$acc routine seq
       character(len=*) :: name
       integer :: handle
     end function gptlstop_handle_gpu
     integer function gptlstop_handle_gpu_c (name, handle)
!$acc routine seq
       character(len=*) :: name
       integer :: handle
     end function gptlstop_handle_gpu_c
  end interface
end module gptl_acc
