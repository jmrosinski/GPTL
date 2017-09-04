program simple
  implicit none
!$acc routine (sub2) seq
!$acc routine (gptlstart_gpu) seq

  integer :: ret
  integer :: n
  integer, external :: sub1, sub2
  integer, external :: gptlinitialize, gptlstart_gpu
  character(len=3) :: zzz = 'zzz'

  write(6,*) 'calling sub1'
  ret = sub1 ()
  write(6,*) 'calling sub2'
!$acc kernels copyin(zzz) copyout(ret)
  ret = sub2 (zzz)
!$acc end kernels

  write(6,*)'simple: calling gptlinitialize 1'
  ret = gptlinitialize ()
  write(6,*)'calling gptlstart_gpu'
!$acc kernels copyout (ret)
  ret = gptlstart_gpu ('zzz')
!$acc end kernels
end program simple
