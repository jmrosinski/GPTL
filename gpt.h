/*
$Id: gpt.h,v 1.9 2004-11-08 23:58:58 rosinski Exp $
*/

typedef enum {
  GPTwall           = 1,
  GPTcpu            = 2,
  GPTabort_on_error = 3,
  GPToverhead       = 4
} Option;

/*
** Function prototypes
*/

extern int GPTsetoption (const int, const int);
extern int GPTinitialize (void);
extern int GPTfinalize (void);
extern int GPTstart (const char *);
extern int GPTstop (const char *);
extern int GPTstamp (double *, double *, double *);
extern int GPTpr (const int);
extern int GPTreset (void);
#ifdef HAVE_PAPI
extern void GPTPAPIprinttable (void);
#endif

