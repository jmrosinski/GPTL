/*
$Id: gpt.h,v 1.3 2004-10-14 19:25:54 rosinski Exp $
*/

typedef enum {
  GPTcpu   = 1,
  GPTwall  = 2,
  GPTother = 3
} Option;

/*
** Function prototypes
*/

extern int GPTsetoption (Option, int);
extern int GPTinitialize (void);
extern int GPTstart (char *);
extern int GPTstop (char *);
extern int GPTstamp (double *, double *, double *);
extern int GPTpr (int);
extern int GPTreset (void);
