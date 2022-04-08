#include "gptl.h"

namespace gptl_once {
  extern int depthlimit;
  extern bool percent;
  extern bool dopr_preamble;
  extern bool dopr_threadsort;
  extern bool dopr_multparent;
  extern bool dopr_collision;
  typedef struct {    
    const GPTLFuncoption option;
    double (*func)(void);
    int (*funcinit)(void);
    const char *name;
  } Funcentry;
  extern Funcentry funclist[];
}
