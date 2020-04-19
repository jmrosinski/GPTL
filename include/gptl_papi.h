#ifndef GPTL_PAPI_H
#define GPTL_PAPI_H

#include "private.h"
#include <stdio.h>

typedef struct {
  int counter;      // PAPI or Derived counter
  char namestr[13]; // PAPI or Derived counter as string
  char str8[9];     // print string for output timers (8 chars)
} GPTLEntry;

typedef struct {
  GPTLEntry event;
  int numidx;       // derived event: PAPI counter array index for numerator
  int denomidx;     // derived event: PAPI counter array index for denominator
} GPTLPr_event;

extern GPTLEntry GPTLeventlist[]; // list of PAPI-based events to be counted
extern int GPTLnevents;           // number of PAPI events (init to 0)

// These are all the GPTL-private PAPI functions
extern int GPTL_PAPIsetoption (const int, const int);
extern int GPTL_PAPIinitialize (const bool);
extern int GPTL_PAPIstart (const int, Papistats *);
extern int GPTL_PAPIstop (const int, Papistats *);
extern void GPTL_PAPIprstr (FILE *);
extern void GPTL_PAPIpr (FILE *, const Papistats *, const int, const int, const double);
extern void GPTL_PAPIadd (Papistats *, const Papistats *);
extern void GPTL_PAPIfinalize (void);
extern void GPTL_PAPIquery (const Papistats *, long long *, int);
extern int GPTL_PAPIget_eventvalue (const char *, const Papistats *, double *);
extern bool GPTL_PAPIis_multiplexed (void);
extern void GPTL_PAPIprintenabled (FILE *);
extern void GPTLread_counters1000 (void);
extern int GPTLget_npapievents (void);
extern int GPTLcreate_and_start_events (const int);

#endif
