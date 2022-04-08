#include "private.h"

namespace gptlmain {
  extern volatile bool initialized;
  extern volatile bool disabled;
  extern int tablesize;
  extern int tablesizem1;
#ifdef HAVE_NANOTIME
  extern double cyc2sec;
#endif
  extern Hashentry **hashtable;
  extern Nofalse *stackidx;
  extern Timer ***callstack;
  extern bool imperfect_nest;
  extern bool dopr_memusage;

  extern "C" {
    double (*ptr2wtimefunc)();
    unsigned int genhashidx (const char *, const int);
    Timer *getentry (const Hashentry *, const char *, unsigned int);
    int preamble_start (int *, const char *);
    int preamble_stop (int *, double *, long *, long *, const char *);
    int update_ll_hash (Timer *, int, unsigned int);
    int update_parent_info (Timer *, Timer **, int);
    int update_stats (Timer *, const double, const long, const long, const int);
    int update_ptr (Timer *, const int);
   }
}
