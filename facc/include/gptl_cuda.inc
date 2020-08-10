integer, external :: gptlstart_gpu
integer, external :: gptlstart_gpu_c
integer, external :: gptlinit_handle_gpu
integer, external :: gptlstart_handle_gpu
integer, external :: gptlstart_handle_gpu_c
integer, external :: gptlstop_gpu
integer, external :: gptlstop_gpu_c
integer, external :: gptlstop_handle_gpu
integer, external :: gptlstop_handle_gpu_c
integer, external :: gptlmy_sleep
integer, external :: gptlget_wallclock_gpu

!$acc routine (gptlinit_handle_gpu) seq
!$acc routine (gptlstart_gpu) seq
!$acc routine (gptlstart_gpu_c) seq
!$acc routine (gptlstart_handle_gpu) seq
!$acc routine (gptlstart_handle_gpu_c) seq
!$acc routine (gptlstop_gpu) seq
!$acc routine (gptlstop_gpu_c) seq
!$acc routine (gptlstop_handle_gpu) seq
!$acc routine (gptlstop_handle_gpu_c) seq
!$acc routine (gptlmy_sleep) seq
!$acc routine (gptlget_wallclock_gpu) seq

