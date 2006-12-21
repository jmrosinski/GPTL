/*
$Id: gptl.h,v 1.8 2006-12-21 02:40:03 rosinski Exp $
*/
#ifndef GPTL_H
#define GPTL_H
typedef enum {
  GPTLwall           = 1,
  GPTLcpu            = 2,
  GPTLabort_on_error = 3,
  GPTLoverhead       = 4
} Option;

typedef enum {
  GPTLnanotime       = 5,
  GPTLrtc            = 6,
  GPTLmpiwtime       = 7,
  GPTLclockgettime   = 8,
  GPTLgettimeofday   = 9
} Funcoption;

/*
** Function prototypes
*/

extern int GPTLsetoption (const int, const int);
extern int GPTLsetutr (const Funcoption);
extern int GPTLinitialize (void);
extern int GPTLfinalize (void);
#ifdef NUMERIC_TIMERS
extern inline int GPTLstart (const unsigned long);
extern inline int GPTLstop (const unsigned long);
#else
extern int GPTLstart (const char *);
extern int GPTLstop (const char *);
#endif
extern int GPTLstamp (double *, double *, double *);
extern int GPTLpr (const int);
extern int GPTLreset (void);
#ifdef HAVE_PAPI
extern void GPTLPAPIprinttable (void);
#endif

extern int GPTLprint_memusage (const char *);
extern int GPTLget_memusage (int *, int *, int *, int *, int *);

#endif
