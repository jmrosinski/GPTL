integer, external :: gptlinit_handle_gpu
integer, external :: gptlstart_gpu
integer, external :: gptlstop_gpu
integer, external :: gptlmy_sleep
integer, external :: gptlget_wallclock_gpu
integer, external :: gptlget_warp_thread
integer, external :: gptlsliced_up_how

!$acc routine (gptlinit_handle_gpu) seq
!$acc routine (gptlstart_gpu) seq
!$acc routine (gptlstop_gpu) seq
!$acc routine (gptlmy_sleep) seq
!$acc routine (gptlget_wallclock_gpu) seq
!$acc routine (gptlget_warp_thread) seq
!$acc routine (gptlsliced_up_how) seq

! These last 2 GPU-related routines run on CPU
integer, external :: gptlget_gpu_props
integer, external :: gptlcudadevsync
