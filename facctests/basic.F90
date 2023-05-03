program basic
  use gptl
  use gptl_acc
  
  implicit none

  ! command argument parsing
  integer :: ret
  integer :: khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu
  integer :: niter, n
  integer :: nwarps
  integer :: total_gputime
  
  ret = gptlget_gpu_props (khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu)
  niter = cores_per_gpu
  nwarps = niter / warpsize

  ! Initialize the GPTL library on CPU and GPU
  ret = gptlinitialize ()

  ! Define handles
!$acc parallel private(ret) copyout(total_gputime)
  ret = gptlinit_handle_gpu ('total_gputime'//char(0), total_gputime)
!$acc end parallel

  ret = gptlstart ('total')
!$acc parallel loop private(ret) copyin(total_gputime)
  do n=0,niter-1
    ret = gptlstart_gpu (total_gputime)
    ret = gptlstop_gpu (total_gputime)
  end do
!$acc end parallel
  ret = gptlstop ('total')
  ret = gptlpr (0)
  stop 0
end program basic
