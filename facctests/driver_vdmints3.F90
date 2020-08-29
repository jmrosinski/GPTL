program driver_vdmints3
  use getval, only: getval_int
  use gptl
  use gptl_acc

  implicit none

  integer :: ips = 1
  integer :: ipe = 10242
  integer :: oversub = 1    ! oversubsubscription factor (default 1)
  integer :: khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu
  integer :: warps_per_gpu
  integer :: nworkers
  integer :: chunksize
  integer :: maxwarps_gpu
  integer, parameter :: nz = 96
  integer, parameter :: vector_dim = nz - 1
  integer :: n
  integer :: ret
  integer :: cores_used

  ret = gptlget_gpu_props (khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu)
  warps_per_gpu = cores_per_gpu / warpsize
  write(6,*)'cores_per_gpu=', cores_per_gpu
  write(6,*)'warps_per_gpu=', warps_per_gpu

  call getval_int (oversub, 'oversubscription factor (int)')
  write(6,*)'oversub=', oversub

  nworkers = nz / warpsize
  call getval_int (nworkers, 'nworkers')

  chunksize = (oversub * cores_per_gpu) / (nz / nworkers)
  write(6,*)'chunksize=', chunksize,' based on (oversub*cores_per_gpu)/(nz/nworkers)'

  maxwarps_gpu = oversub * warps_per_gpu;
  call getval_int (maxwarps_gpu, 'max warp number to examine: default will cover all')
  write(6,*) 'maxwarps_gpu=', maxwarps_gpu
  ret = gptlsetoption (gptlmaxwarps_gpu, maxwarps_gpu)

  ret = gptlsetoption (gptlverbose, 1)
!ret = gptlsetoption (gptlmaxthreads_gpu, 262208)
!ret = gptlsetoption (gptlmaxwarps_gpu, 448)   ! for flatten
! This one is the highest number that works

  write(6,*)'Calling gptlinitialize'
  ret = gptlinitialize()
  do n=1,10
    write(6,*)'calling vdmints3_sim iter=',n
    ret = gptlstart('vdmints3_sim')
    call vdmints3_sim (nz, ips, ipe, chunksize, warpsize, nworkers)
    ret = gptlstop('vdmints3_sim')
  end do
  ret = gptlpr(0)
end program driver_vdmints3
