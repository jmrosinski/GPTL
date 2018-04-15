program driver_vdmints3
  use getval, only: getval_int
  use gptl
  use gptl_acc

  implicit none

  integer :: nz = 96
  integer :: ips = 1
  integer :: ipe = 10242
  integer :: n
  integer :: ret

  call getval_int (nz, 'nz')
  call getval_int (ips, 'ips')
  call getval_int (ipe, 'ipe')

  ret = gptlsetoption (gptlverbose, 1)
!ret = gptlsetoption (gptlmaxthreads_gpu, 262208)
!ret = gptlsetoption (gptlmaxwarps_gpu, 448)   ! for flatten
! This one is the highest number that works
  ret = gptlsetoption (gptlmaxwarps_gpu, 33600)
  ret = gptlsetoption (gptlmaxtimers_gpu, 10)
  ret = gptlinitialize()
  do n=1,10
    ret = gptlstart('vdmints3_sim')
    call vdmints3_sim (nz, ips, ipe)
    ret = gptlstop('vdmints3_sim')
  end do
  ret = gptlpr(0)
end program driver_vdmints3
