program verify_results
  use gptl
  use gptl_acc
  use openacc
  
  integer, save :: vdmints3_handle, ipn_handle, k_handle
  integer :: ipn
  integer, parameter :: ips=1
!  integer :: ipe=10242
  integer, parameter :: ipe=10242
  integer, parameter :: NZ=96
  integer, parameter :: maxtimers = 3  ! max number of USER timers
  integer, parameter :: warpsize= 32
  integer :: maxwarps = ((ipe-ips+1)*NZ)/warpsize
  integer :: ret
  real :: sleepsec = 0.1

  
  ret = gptlsetoption (gptlmaxwarps_gpu, maxwarps) 
  ret = gptlsetoption (gptlmaxtimers_gpu, 3)
  ret = gptlinitialize ()
  ret = gptlstart ('total')

!$acc parallel private(ret) copyout(vdmints3_handle, ipn_handle, k_handle)
  ret = gptlinit_handle_gpu ('vdmints3',     vdmints3_handle)
  ret = gptlinit_handle_gpu ('vdmints3_ipn', ipn_handle)
  ret = gptlinit_handle_gpu ('vdmints3_k',   k_handle)
!$acc end parallel
  ret = gptlcudadevsync ()

!$acc parallel private(ret) copyin(vdmints3_handle)
  ret = gptlstart_gpu (vdmints3_handle)
!$acc end parallel

!$acc parallel private(ret) num_workers(3) vector_length(32), copyin(ipn_handle,k_handle,sleepsec)
!$acc loop gang
  do ipn=ips,ipe
    ret = gptlstart_gpu (ipn_handle)
!$acc loop vector
    do k=1,NZ-1
      ret = gptlstart_gpu (k_handle)
      ret = gptlmy_sleep (sleepsec)
      ret = gptlstop_gpu (k_handle)
    end do
    ret = gptlstop_gpu (ipn_handle)
  end do
!$acc end parallel

!$acc parallel private(ret)
  ret = gptlstop_gpu (vdmints3_handle)
!$acc end parallel
  ret = gptlcudadevsync ()

  ret = gptlstop ('total')
  ret = gptlpr (0)
  stop 0
end program verify_results
