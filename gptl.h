/*
$Id: gptl.h,v 1.3 2005-08-23 02:21:27 rosinski Exp $
*/
#ifndef GPTL_H
#define GPTL_H
typedef enum {
  GPTLwall           = 1,
  GPTLcpu            = 2,
  GPTLabort_on_error = 3,
  GPTLoverhead       = 4
} Option;

/*
** Function prototypes
*/

extern int GPTLsetoption (const int, const int);
extern int GPTLinitialize (void);
extern int GPTLfinalize (void);
#ifdef NUMERIC_TIMERS
extern int GPTLstart (const unsigned long);
extern int GPTLstop (const unsigned long);
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

#endif
