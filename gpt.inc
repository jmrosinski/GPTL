      integer GPTwall
      integer GPTcpu

      parameter (GPTwall = 1)
      parameter (GPTcpu  = 2)

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

