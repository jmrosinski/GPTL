program driver_vdmints3
  use getval, only: getval_int
  use gptl
  use gptl_acc

  implicit none

  integer :: ips = 1
  integer :: ipe = 10242
  integer :: oversub = 1
  integer :: n
  integer :: ret

  call getval_int (ips, 'ips')
  call getval_int (ipe, 'ipe')
  call getval_int (oversub, 'oversub')

  ret = gptlsetoption (gptlverbose, 1)
!ret = gptlsetoption (gptlmaxthreads_gpu, 262208)
!ret = gptlsetoption (gptlmaxwarps_gpu, 448)   ! for flatten
! This one is the highest number that works
  ret = gptlsetoption (gptlmaxwarps_gpu, 33600 / oversub)
  ret = gptlsetoption (gptlmaxtimers_gpu, 10)
  ret = gptlinitialize()
  do n=1,10
    ret = gptlstart('vdmints3_sim')
    call vdmints3_sim (ips, ipe, oversub)
    ret = gptlstop('vdmints3_sim')
  end do
  ret = gptlpr(0)
end program driver_vdmints3
