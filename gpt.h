/*
$Id: gpt.h,v 1.4 2004-10-14 22:02:18 rosinski Exp $
*/

typedef enum {
  GPTwall  = 1,
  GPTcpu   = 2,
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
