/*
$Id: gptl.h,v 1.14 2007-01-01 04:36:20 rosinski Exp $
*/
#ifndef GPTL_H
#define GPTL_H
typedef enum {
  GPTLwall           = 1,
  GPTLcpu            = 2,
  GPTLabort_on_error = 3,
  GPTLoverhead       = 4,
  GPTLdepthlimit     = 5
} Option;

typedef enum {
  GPTLnanotime       = 6,
  GPTLrtc            = 7,
  GPTLmpiwtime       = 8,
  GPTLclockgettime   = 9,
  GPTLgettimeofday   = 10
} Funcoption;

/*
** Function prototypes
*/

extern int GPTLsetoption (const int, const int);
extern int GPTLinitialize (void);
extern int GPTLstart (const char *);
extern int GPTLstop (const char *);
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
