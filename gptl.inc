      integer GPTLwall
      integer GPTLcpu
      integer GPTLabort_on_error
      integer GPTLoverhead

      integer GPTLnanotime
      integer GPTLrtc
      integer GPTLmpiwtime
      integer GPTLclockgettime
      integer GPTLgettimeofday

      parameter (GPTLwall           = 1)
      parameter (GPTLcpu            = 2)
      parameter (GPTLabort_on_error = 3)
      parameter (GPTLoverhead       = 4)

      parameter (GPTLnanotime       = 5)
      parameter (GPTLrtc            = 6)
      parameter (GPTLmpiwtime       = 7)
      parameter (GPTLclockgettime   = 8)
      parameter (GPTLgettimeofday   = 9)

! for compatibility with CAM entries

      integer wall
      integer usrsys
      parameter (wall = GPTLwall)
      parameter (usrsys = GPTLcpu)

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
      integer gptlsetutr
      integer gptlenable
      integer gptldisable

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
      external gptlsetutr
      external gptlenable
      external gptldisable
