module subs
  use gptl
  use gptl_acc

  implicit none

  private
  public :: doalot_log, doalot_log_inner, doalot_sqrt, doalot_sqrt_double

CONTAINS
  
  real function doalot_log (n, innerlooplen) result (sum)
    integer, intent(in) :: n, innerlooplen
    integer :: i, iter
    real :: sum
!$acc routine seq

    sum = 0.
    do iter=1,innerlooplen
      do i=1,n
        sum = sum + log (real (iter*i))
      end do
    end do
  end function doalot_log

! doalot_log_inner: Same computations as doalot_log, but add a timer inside "innerlooplen"
  real function doalot_log_inner (n, innerlooplen, doalot_log_inner_handle) result (sum)
    integer, intent(in) :: n, innerlooplen, doalot_log_inner_handle
    integer :: i, iter
    integer :: ret
    real :: sum
!$acc routine seq

    sum = 0.
    do iter=1,innerlooplen
      ret = gptlstart_gpu (doalot_log_inner_handle)
      do i=1,n
        sum = sum + log (real (iter*i))
      end do
      ret = gptlstop_gpu (doalot_log_inner_handle)
    end do
  end function doalot_log_inner

  real function doalot_sqrt (n, innerlooplen) result (sum)
    integer, intent(in) :: n, innerlooplen
    integer :: i, iter
    real :: sum
!$acc routine seq

    sum = 0.
    do iter=1,innerlooplen
      do i=1,n
        sum = sum + sqrt (float (iter*i))
      end do
    end do
  end function doalot_sqrt

  real*8 function doalot_sqrt_double (n, innerlooplen) result (sum)
    integer, intent(in) :: n, innerlooplen
    integer :: i, iter
    real*8 :: sum
!$acc routine seq

    sum = 0.
    do iter=1,innerlooplen
      do i=1,n
        sum = sum + sqrt (dble (iter*i))
      end do
    end do
  end function doalot_sqrt_double
end module subs
