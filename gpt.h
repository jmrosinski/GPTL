/*
$Id: gpt.h,v 1.5 2004-10-15 04:56:27 rosinski Exp $
*/

typedef enum {
  GPTwall           = 1,
  GPTcpu            = 2,
  GPTabort_on_error = 3,
  GPTother          = 4
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
