#ifndef ONCE_H
#define ONCE_H

#include "private.h"
#include "gptl.h"

#include <sys/time.h>

namespace gptl_once {
  using namespace gptl_private;
  extern float cpumhz;
  extern volatile bool initialized;
  extern bool percent;
  extern bool dopr_preamble;
  extern bool dopr_threadsort;
  extern bool dopr_multparent;
  extern bool dopr_collision;
  extern bool dopr_memusage;
  extern bool verbose;
  extern long ticks_per_sec;
  extern GPTL_Method method;
  extern int depthlimit;
  extern bool sync_mpi;
  extern bool onlypr_rank0;
  extern time_t ref_gettimeofday;
  extern time_t ref_clock_gettime;
  extern int funcidx;
#ifdef HAVE_NANOTIME
  extern double cyc2sec;
  extern char *clock_source;
  extern "C" {
    float get_clockfreq (void);
  }
#endif
  extern "C" {
    typedef struct {
      const GPTL_Funcoption option;   // GPTL_Funcoption is user-visible
      double (*func)(void);
      int (*funcinit)(void);
      const char *name;
    } Funcentry;

    extern Funcentry funclist[];
#ifdef HAVE_GETTIMEOFDAY
    extern int init_gettimeofday (void);
#endif
#ifdef HAVE_NANOTIME
    extern int init_nanotime (void);
    extern float get_clockfreq (void);         // cycles/sec
#endif
#ifdef HAVE_LIBMPI
    extern int init_mpiwtime (void);
#endif
#ifdef HAVE_LIBRT
    extern int init_clock_gettime (void);
#endif
#ifdef _AIX
    extern int init_read_real_time (void);
#endif
    extern int init_placebo (void);
  }
}
#endif

