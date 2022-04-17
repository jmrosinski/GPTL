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
  extern Timer **timers;
  extern Timer ***callstack;
  extern Timer **last;
  extern bool imperfect_nest;
  extern bool dopr_memusage;
  extern Settings cpustats;
  extern Settings wallstats;
  extern int depthlimit;
  extern bool dousepapi;
#ifdef HAVE_GETTIMEOFDAY
  extern time_t ref_gettimeofday;
#endif
#ifdef HAVE_LIBRT
  extern time_t ref_clock_gettime;
#endif
#ifdef _AIX
  extern time_t ref_read_real_time;
#endif

  extern "C" {
    // Not sure why the following extern is needed but it seems to be
    extern double (*ptr2wtimefunc)();
    unsigned int genhashidx (const char *, const int);
    Timer *getentry (const Hashentry *, const char *, unsigned int);
    int preamble_start (int *, const char *);
    int preamble_stop (int *, double *, long *, long *, const char *);
    int update_ll_hash (Timer *, int, unsigned int);
    int update_parent_info (Timer *, Timer **, int);
    int update_stats (Timer *, const double, const long, const long, const int);
    int update_ptr (Timer *, const int);
#ifdef ENABLE_NESTEDOMP
    void get_nested_thread_nums (int *, int *);
#endif
    // These are the (possibly) supported underlying wallclock timers
#ifdef HAVE_NANOTIME
    double utr_nanotime (void);
#endif
#ifdef HAVE_LIBMPI
    double utr_mpiwtime (void);
#endif
#ifdef _AIX
    double utr_read_real_time (void);
#endif
#ifdef HAVE_LIBRT
    double utr_clock_gettime (void);
#endif
#ifdef HAVE_GETTIMEOFDAY
    double utr_gettimeofday (void);
#endif
    double utr_placebo (void);
  }
}
