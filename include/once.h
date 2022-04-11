#include "gptl.h"

namespace once {
  extern int depthlimit;
  extern long ticks_per_sec;      // clock ticks per second
#ifdef HAVE_NANOTIME
  extern float cpumhz;
  extern char *clock_source;
#endif
  extern int funcidx;
  extern bool verbose;
  extern bool onlyprint_rank0;
  
  // Wrap in extern "C" due to functions. Not sure if this is necessary
  extern "C" {
    typedef struct {    
      const GPTLFuncoption option;
      double (*func)(void);
      int (*funcinit)(void);
      const char *name;
    } Funcentry;
    extern Funcentry funclist[];
  }
}
