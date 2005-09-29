/*
$Id: gptl.h,v 1.5 2005-09-29 02:45:43 rosinski Exp $
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

extern int print_memusage (char *);
extern int get_memusage (int *, int *, int *, int *, int *);

#endif
