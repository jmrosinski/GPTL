/*
$Id: gptl.h,v 1.12 2006-12-31 23:52:20 rosinski Exp $
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
extern int GPTLinitialize (void);

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
extern int GPTLfinalize (void);
extern int GPTLprint_memusage (const char *);
extern int GPTLget_memusage (int *, int *, int *, int *, int *);
extern int GPTLsetutr (const int);
extern int GPTLenable (void);
extern int GPTLdisable (void);

/* 
** These are defined in gptl_papi.c 
*/

extern void GPTL_PAPIprinttable (void);
extern int GPTL_PAPIname2id (const char *);
#endif
