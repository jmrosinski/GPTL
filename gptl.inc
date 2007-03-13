      integer GPTLwall
      integer GPTLcpu
      integer GPTLabort_on_error
      integer GPTLoverhead
      integer GPTLdepthlimit
      integer GPTLverbose
      integer GPTLnarrowprint

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

      parameter (GPTLgettimeofday   = 8)
      parameter (GPTLnanotime       = 9)
      parameter (GPTLrtc            = 10)
      parameter (GPTLmpiwtime       = 11)
      parameter (GPTLclockgettime   = 12)
      parameter (GPTLpapitime       = 13)

! Externals

      integer gptlsetoption
      integer gptlinitialize
      integer gptlstart
      integer gptlstop
      integer gptlstamp 
      integer gptlpr
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
