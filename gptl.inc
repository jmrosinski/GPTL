!
! $Id: gptl.inc.template,v 1.3 2011-03-28 20:55:19 rosinski Exp $
!
! Author: Jim Rosinski
!
! GPTL header file to be included in user code. Values match
! their counterparts in gptl.h. See that file or man pages
! or web-based documenation for descriptions of each value
!
      integer, parameter :: GPTLsync_mpi        = 0
      integer, parameter :: GPTLwall            = 1
      integer, parameter :: GPTLcpu             = 2
      integer, parameter :: GPTLabort_on_error  = 3
      integer, parameter :: GPTLoverhead        = 4
      integer, parameter :: GPTLdepthlimit      = 5
      integer, parameter :: GPTLverbose         = 6
      integer, parameter :: GPTLpercent         = 9
      integer, parameter :: GPTLdopr_preamble   = 12
      integer, parameter :: GPTLdopr_threadsort = 13
      integer, parameter :: GPTLdopr_multparent = 14
      integer, parameter :: GPTLdopr_collision  = 15
      integer, parameter :: GPTLdopr_memusage   = 27
      integer, parameter :: GPTLprint_method    = 16
      integer, parameter :: GPTLtablesize       = 50
      integer, parameter :: GPTLmaxthreads      = 51
      integer, parameter :: GPTLmaxwarps_gpu    = 52
      integer, parameter :: GPTLtablesize_gpu   = 53
      integer, parameter :: GPTLmaxtimers_gpu   = 54

      integer, parameter :: GPTLnanotime        = 2
      integer, parameter :: GPTLmpiwtime        = 4
      integer, parameter :: GPTLclockgettime    = 5
      integer, parameter :: GPTLgettimeofday    = 1
      integer, parameter :: GPTLplacebo         = 7
      integer, parameter :: GPTLread_real_time  = 3

      integer, parameter :: GPTLfirst_parent    = 1
      integer, parameter :: GPTLlast_parent     = 2
      integer, parameter :: GPTLmost_frequent   = 3
      integer, parameter :: GPTLfull_tree       = 4

! Externals

      integer, external :: gptlsetoption
      integer, external :: gptlinitialize
      integer, external :: gptlstart
      integer, external :: gptlstart_handle
      integer, external :: gptlinit_handle
      integer, external :: gptlstop
      integer, external :: gptlstop_handle
      integer, external :: gptlstamp 
      integer, external :: gptlpr
      integer, external :: gptlpr_file
      integer, external :: gptlpr_summary
      integer, external :: gptlpr_summary_file
      integer, external :: gptlbarrier
      integer, external :: gptlreset 
      integer, external :: gptlreset_timer
      integer, external :: gptlfinalize
      integer, external :: gptlget_memusage
      integer, external :: gptlprint_memusage
      integer, external :: gptlprint_rusage
      integer, external :: gptlenable
      integer, external :: gptldisable
      integer, external :: gptlsetutr
      integer, external :: gptlquery
      integer, external :: gptlget_wallclock
      integer, external :: gptlget_wallclock_latest
      integer, external :: gptlget_threadwork
      integer, external :: gptlstartstop_val
      integer, external :: gptlget_eventvalue
      integer, external :: gptlget_nregions
      integer, external :: gptlget_regionname
      integer, external :: gptlevent_name_to_code
      integer, external :: gptlevent_code_to_name
      integer, external :: gptlnum_errors
      integer, external :: gptlnum_warn
      integer, external :: gptlget_count
      integer, external :: gptlget_gpu_props
      integer, external :: gptlcudadevsync
