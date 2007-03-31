      integer GPTLwall
      integer GPTLcpu
      integer GPTLabort_on_error
      integer GPTLoverhead
      integer GPTLdepthlimit
      integer GPTLverbose
      integer GPTLnarrowprint
      integer GPTLparentchild

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

      parameter (GPTLgettimeofday   = 9)
      parameter (GPTLnanotime       = 10)
      parameter (GPTLrtc            = 11)
      parameter (GPTLmpiwtime       = 12)
      parameter (GPTLclockgettime   = 13)
      parameter (GPTLpapitime       = 14)

! Externals

      integer gptlsetoption
      integer gptlinitialize
      integer t_startf
      integer t_stopf
      integer t_stampf
      integer gptlpr
      integer gptlreset 
      integer gptlfinalize
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
      external t_startf
      external t_stopf
      external t_stampf
      external gptlpr
      external gptlreset 
      external gptlfinalize
      external gptlenable
      external gptldisable
      external gptlsetutr
      external gptlquery
      external gptlquerycounters
      external gptlget_nregions
      external gptlget_regionname
      external gptl_papiprinttable
      external gptl_papiname2id
