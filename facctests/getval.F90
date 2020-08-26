module getval
  implicit none

CONTAINS

  subroutine getval_int (arg, str)
    implicit none
    
    integer, intent(inout) :: arg
    character(len=*), intent(in) :: str
    
    integer :: ans

    write(6,'(a,a,a,i9,a)')'Enter ',str,' or -1 to accept default (',arg,')'
    read(5,*) ans
    if (ans /= -1) then
      arg = ans
    end if
    write(6,*) 'arg=',arg
  end subroutine getval_int

  subroutine getval_real (arg, str)
    implicit none

    real, intent(inout) :: arg
    character(len=*), intent(in) :: str
    
    real :: ans

    write(6,'(a,a,a,f5.3,a)')'Enter ',str,' or -1 to accept default (',arg,')'
    read(5,*) ans
    if (ans /= -1.) then
      arg = ans
    end if
    write(6,*) 'arg=',arg
  end subroutine getval_real
end module getval
