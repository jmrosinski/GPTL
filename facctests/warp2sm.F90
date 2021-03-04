program warp2sm
  use gptl
  use gptl_acc
  use openacc

  integer :: khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu
  integer :: nwarps
  integer :: mywarp
  integer :: inner = 96
  integer :: outer = 10242
  integer :: totaliters
  integer :: n, nn, k
  integer, allocatable :: smarr(:)
  integer :: ret
  integer :: total_gputime, inner_loop, outer_loop

  ret = gptlget_gpu_props (khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu)

  totaliters = inner*outer
  nwarps = totaliters / warpsize
  if (mod (totaliters,warpsize) /= 0) then
    nwarps = nwarps + 1
  end if

  ret = gptlsetoption (gptlmaxwarps_gpu, nwarps)
  ret = gptlinitialize ()

  allocate (smarr(0:nwarps-1))
  smarr(:) = -1

!$acc parallel private(ret) copyout(total_gputime, inner_loop, outer_loop)
  ret = gptlinit_handle_gpu ('total_gputime'//char(0), total_gputime)
  ret = gptlinit_handle_gpu ('inner_loop'//char(0),    inner_loop)
  ret = gptlinit_handle_gpu ('outer_loop'//char(0),    outer_loop)
!$acc end parallel
  ret = gptlcudadevsync ()

  ret = gptlstart ('total')
!$acc parallel private(ret) copyin (total_gputime)
  ret = gptlstart_gpu (total_gputime)
!$acc end parallel

!$acc parallel loop private(n,k,ret,mywarp) copyin(outer, inner) copy(smarr)
  do n=0,outer-1
    ret = gptlstart_gpu (outer_loop)
!$acc loop vector
    do k=0,inner-1
      ret = gptlstart_gpu (inner_loop)
      mywarp = gptlget_sm_thiswarp (smarr)
      ret = gptlstop_gpu (inner_loop)
    end do
    ret = gptlstop_gpu (outer_loop)
  end do
  ret = gptlstart ('devsync')
  ret = gptlcudadevsync ()
  ret = gptlstop ('devsync')

!$acc parallel private(ret) copyin (total_gputime)
  ret = gptlstop_gpu (total_gputime)
!$acc end parallel
  ret = gptlstop ('total')

  do n=0,nwarps-1
    if (smarr(n) > -1) then
      write(6,*) 'warp ', n, 'sm ', smarr(n)
    end if
  end do
  ret = gptlpr (0)
  stop 0
end program warp2sm
