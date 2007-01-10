      integer GPTLwall
      integer GPTLcpu
      integer GPTLabort_on_error
      integer GPTLoverhead
      integer GPTLdepthlimit

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

      parameter (GPTLnanotime       = 6)
      parameter (GPTLrtc            = 7)
      parameter (GPTLmpiwtime       = 8)
      parameter (GPTLclockgettime   = 9)
      parameter (GPTLgettimeofday   = 10)
      parameter (GPTLpapitime       = 11)

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
      external gptl_papiprinttable
      external gptl_papiname2id
