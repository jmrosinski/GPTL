program persist
  use gptl
  use gptl_gpu
  implicit none
!$acc routine (sub2) seq
!$acc routine (gptlstart_gpu) seq
!$acc routine (gptlstop_gpu) seq
!$acc routine (doalot) seq

  integer :: ret
  integer :: n
  integer :: pret(1000)
  real :: vals(1000)
  integer, external :: sub1, sub2
!  integer, external :: gptlstart_gpu, gptlstop_gpu
  real, external :: doalot

  write(6,*) 'calling sub1'
  ret = sub1 ()
  write(6,*) 'calling sub2'
!$acc kernels copyout (ret)
  ret = sub2 ('zzz')
!$acc end kernels

  write(6,*)'persist: calling gptlinitialize 1'
  ret = gptlinitialize ()
  ret = gptlstart ('doalot_cpu')
!$acc parallel loop copyout(ret, vals)
  do n=1,1000
    ret = gptlstart_gpu ('doalot_perwarp')
    vals(n) = doalot (n)
    ret = gptlstop_gpu ('doalot_perwarp')
  end do
!$acc end parallel
  ret = gptlstop ('doalot_cpu')
  ret = gptlpr (0)
end program persist

real function doalot (n) result (sum)
  implicit none
  integer, intent(in) :: n
  integer :: i, iter
  real :: sum
!$acc routine seq

  sum = 0.
  do iter=1,10000
    do i=1,n
      sum = sum + log (real (iter*i))
    end do
  end do
end function doalot
