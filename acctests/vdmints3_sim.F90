subroutine vdmints3_sim (nz, ips, ipe)
  use gptl
  use gptl_acc

  integer, intent(in) :: nz, ips, ipe

  integer :: k, isn, ipn
  integer :: ret, ret2
  integer :: handle, handle2

!$acc routine (solveithls3_sim) vector
!$acc routine (burn_time) vector

!$acc parallel private(ret) copyout(handle,handle2)
  ret = gptlinit_handle_gpu ('vdmints3', handle)
  ret = gptlinit_handle_gpu ('vdmints3_ipn', handle2)
  ret = gptlstart_gpu ('vdmints3_gpu')
!$acc end parallel

!$acc parallel private(ret) num_workers(1) vector_length(96) copyin(handle,handle2)
  ret = gptlstart_gpu_c ('vdmints3'//char(0))
!$acc loop gang worker private(ret2)
  do ipn=ips,ipe
    ret2 = gptlstart_gpu_c('vdmints3_ipn'//char(0))
!$acc loop vector
    do k=1,NZ-1
      call burn_time(ipn,k)
    end do

    call solveithls3_sim(ipn)

    do isn=1,6
!$acc loop vector
      do k=1,NZ-1
        call burn_time(1,k)
      end do
    end do
!$acc loop vector
    do k=1,NZ-1
      call burn_time(ipn,k)
    end do
    call burn_time(ipn,1)
    ret2 = gptlstop_gpu_c('vdmints3_ipn'//char(0))
  end do
  ret = gptlstop_gpu_c('vdmints3'//char(0))
!$acc end parallel

!$acc parallel private(ret)
  ret = gptlstop_gpu ('vdmints3_gpu')
!$acc end parallel
end subroutine vdmints3_sim

subroutine burn_time (ipn,k)
  use gptl_acc
  integer, intent(in) :: ipn,k
  real :: sleep_duration
  real :: summ
  integer :: i,kk

!$acc routine seq

  summ = 0.
  do kk=1,k
    do i=1,ipn
      summ = summ + i/kk
    end do
  end do
!  ret = gptlmy_sleep (sleep_duration)
end subroutine burn_time

subroutine solveithls3_sim(ipn)
  use gptl_acc
  integer, intent(in) :: ipn
  real :: sleep_duration
  real :: summ
  integer :: i,kk

!$acc routine seq
  sleep_duration = 0.01
  summ = 0.
  do i=1,ipn
    summ = summ + i
  end do
!  ret = gptlmy_sleep (sleep_duration)
end subroutine solveithls3_sim
