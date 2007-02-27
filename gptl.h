/*
$Id: gptl.h,v 1.22 2007-02-27 20:26:41 rosinski Exp $
*/
#ifndef GPTL_H
#define GPTL_H
typedef enum {
  GPTLwall           = 1,
  GPTLcpu            = 2,
  GPTLabort_on_error = 3,
  GPTLoverhead       = 4,
  GPTLdepthlimit     = 5,
  GPTLverbose        = 6,

  GPTLnarrowprint    = -2 /* PAPI options are negative */
} Option;

typedef enum {
  GPTLgettimeofday   = 7,
  GPTLnanotime       = 8,
  GPTLrtc            = 9,
  GPTLmpiwtime       = 10,
  GPTLclockgettime   = 11,
  GPTLpapitime       = 12
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
extern int GPTLquery (const char *, int, int *, int *, double *, double *, double *,
		      long long *, const int);
extern int GPTLquerycounters (const char *, int, long long *);
extern int GPTLget_nregions (int, int *);
extern int GPTLget_regionname (int, int, char *, int);

/* 
** These are defined in gptl_papi.c 
*/

extern void GPTL_PAPIprinttable (void);
extern int GPTL_PAPIname2id (const char *);
#endif
