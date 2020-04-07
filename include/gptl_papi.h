#ifndef GPTL_PAPI_H
#define GPTL_PAPI_H

#include "private.h"
#include <stdio.h>

namespace gptl_papi {
  using namespace gptl_private;
  extern int npapievents;
  extern int nevents;
  extern int *EventSet;
  extern long long **papicounters;
  extern bool is_multiplexed;

  typedef struct {
    int counter;         // PAPI or Derived counter
    char namestr[13];    // PAPI or Derived counter as string
    char str8[9];        // print string for output timers (8 chars)
  } Entry;

  // Functions called from anywhere in GPTL
  extern "C" {
    int PAPIsetoption (const int, const int);
    int PAPIlibraryinit (void);
    int PAPIinitialize (const int, const bool, int *, Entry *);
    int create_and_start_events (const int);
    int PAPIstart (const int, Papistats *);
    int PAPIstop (const int, Papistats *);
    void PAPIprstr (FILE *);
    void PAPIpr (FILE *fp, const Papistats *, const int, const int, const double);
    void PAPIprintenabled (FILE *);
    void PAPIadd (Papistats *, const Papistats *);
    void PAPIfinalize (int);
    void PAPIquery (const Papistats *, long long *, int);
    int PAPIget_eventvalue (const char *, const Papistats *, double *);
    void read_counters1000 (void);
  }
}
#endif
