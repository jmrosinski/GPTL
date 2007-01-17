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

      parameter (GPTLgettimeofday   = 6)
      parameter (GPTLnanotime       = 7)
      parameter (GPTLrtc            = 8)
      parameter (GPTLmpiwtime       = 9)
      parameter (GPTLclockgettime   = 10)
      parameter (GPTLpapitime       = 11)

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
      external gptl_papiprinttable
      external gptl_papiname2id
