      integer GPTLwall
      integer GPTLcpu
      integer GPTLabort_on_error
      integer GPTLoverhead
      integer GPTLdepthlimit
      integer GPTLverbose
      integer GPTLnarrowprint
      integer GPTLparentchild
      integer GPTLpercent
      integer GPTLpersec
      integer GPTLmultiplex

      integer GPTLnanotime
      integer GPTLrtc
      integer GPTLmpiwtime
      integer GPTLclockgettime
      integer GPTLgettimeofday
      integer GPTLpapitime

      parameter (GPTLwall           = 1)
      parameter (GPTLcpu            = 2)
      parameter (GPTLabort_on_error = 3)
      parameter (GPTLoverhead       = 4)
      parameter (GPTLdepthlimit     = 5)
      parameter (GPTLverbose        = 6)
      parameter (GPTLnarrowprint    = 7)
      parameter (GPTLparentchild    = 8)
      parameter (GPTLpercent        = 9)
      parameter (GPTLpersec         = 10)
      parameter (GPTLmultiplex      = 11)

      parameter (GPTLgettimeofday   = 12)
      parameter (GPTLnanotime       = 13)
      parameter (GPTLrtc            = 14)
      parameter (GPTLmpiwtime       = 15)
      parameter (GPTLclockgettime   = 16)
      parameter (GPTLpapitime       = 17)

! Externals

      integer gptlsetoption
      integer gptlinitialize
      integer gptlstart
      integer gptlstop
      integer gptlstamp 
      integer gptlpr
      integer gptlpr_summary
      integer gptlreset 
      integer gptlfinalize
      integer gptlget_memusage
      integer gptlprint_memusage
      integer gptlenable
      integer gptldisable
      integer gptlsetutr
      integer gptlquery
      integer gptlquerycounters
      integer gptlget_nregions
      integer gptlget_regionname
      integer gptl_papiprinttable
      integer gptl_papiname2id

      external gptlsetoption
      external gptlinitialize
      external gptlstart
      external gptlstop
      external gptlstamp 
      external gptlpr
      external gptlpr_summary
      external gptlreset 
      external gptlfinalize
      external gptlget_memusage
      external gptlprint_memusage
      external gptlenable
      external gptldisable
      external gptlsetutr
      external gptlquery
      external gptlquerycounters
      external gptlget_nregions
      external gptlget_regionname
      external gptl_papiprinttable
      external gptl_papiname2id
