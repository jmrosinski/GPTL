      integer GPTwall
      integer GPTcpu

      parameter (GPTwall = 1)
      parameter (GPTcpu  = 2)

! for compatibility with CAM entries

      integer wall
      integer usrsys
      parameter (wall = GPTwall)
      parameter (usrsys = GPTcpu)

! Externals

      integer gptsetoption
      integer gptinitialize
      integer gptstart
      integer gptstop
      integer gptstamp 
      integer gptpr
      integer gptreset 

      external gptsetoption
      external gptinitialize
      external gptstart
      external gptstop
      external gptstamp 
      external gptpr
      external gptreset 

