program driver_vdmints3
  use getval, only: getval_int
  use gptl
  use gptl_acc

  implicit none

  integer :: maxwarps_gpu
  integer :: ips = 1
  integer :: ipe = 10242
  integer :: oversub
  integer :: cores_per_sm
  integer :: cores_per_gpu
  integer :: khz, warpsize, devnum, smcount
  integer :: n
  integer :: ret

  ret = gptlget_gpu_props (khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu)
  write(6,*)'cores_per_gpu=', cores_per_gpu

  maxwarps_gpu = cores_per_gpu / warpsize
  call getval_int (maxwarps_gpu, 'maxwarps_gpu')
  write(6,*)'maxwarps_gpu=',maxwarps_gpu
  ret = gptlsetoption (gptlmaxwarps_gpu, maxwarps_gpu)

  call getval_int (ips, 'ips')
  call getval_int (ipe, 'ipe')

  oversub = (ipe - ips + 1 + (cores_per_gpu-1)) / cores_per_gpu
  call getval_int (oversub, 'oversub')

  ret = gptlsetoption (gptlverbose, 1)
!ret = gptlsetoption (gptlmaxthreads_gpu, 262208)
!ret = gptlsetoption (gptlmaxwarps_gpu, 448)   ! for flatten
! This one is the highest number that works

  write(6,*)'Calling gptlinitialize'
  ret = gptlinitialize()
  do n=1,10
    ret = gptlstart('vdmints3_sim')
    call vdmints3_sim (ips, ipe, oversub)
    ret = gptlstop('vdmints3_sim')
  end do
  ret = gptlpr(0)
end program driver_vdmints3
